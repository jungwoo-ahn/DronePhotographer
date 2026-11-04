// Renders /api/state and offers exactly two orders: run the move the policy proposed,
// or stop. Every decision still lives in the supervisor and the gate; this file draws
// them and posts the two clicks.
//
// `renderPanel` is kept separate from the polling driver at the foot of the file so the
// simulated-flight demo can drive the same markup without a control loop behind it.
const $ = (s) => document.querySelector(s);
const POLL_MS = 150;
const FRAME_MS = 250;

const CELLS = [
  { k: 'vx', label: 'Forward', unit: 'm/s', d: 2 },
  { k: 'vy', label: 'Left', unit: 'm/s', d: 2 },
  { k: 'vz', label: 'Up', unit: 'm/s', d: 2 },
  { k: 'yaw_rate', label: 'Yaw left', unit: '°/s', d: 0 },
];

const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : v.toFixed(d);
const esc = (t) => String(t).replace(/[<>&]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]));
const utc = (wall) => (wall ? new Date(wall * 1000).toISOString().slice(11, 19) : '—');

function renderLinks(s) {
  const m = s.links.mavlink, v = s.links.video, p = s.links.policy;
  $('#l-mavlink').dataset.state = m.ok ? 'up' : 'down';
  $('#v-mavlink').textContent = m.ok ? 'receiving' : 'no telemetry';
  $('#l-video').dataset.state = !v.present ? 'off' : (v.ok ? 'up' : 'down');
  $('#v-video').textContent = !v.present ? 'not in use'
    : v.age_ms === null ? 'no frames' : `${v.age_ms.toFixed(0)} ms`;
  $('#l-policy').dataset.state = !p.present ? 'off' : (p.ok ? 'up' : 'down');
  $('#v-policy').textContent = !p.present ? 'not in use' : `${p.calls} asked`;
}

function renderCamera(s) {
  const v = s.links.video, T = s.telemetry || {};
  const state = !v.present ? 'off' : (v.age_ms === null ? 'waiting' : (v.ok ? 'live' : 'stale'));
  $('#cam').dataset.state = state;
  const over = $('#camover');
  over.hidden = state === 'live';
  over.textContent = state === 'off' ? 'No camera on this run.'
    : state === 'waiting' ? 'Waiting for the first frame.'
    : 'Signal lost, so the policy is not being asked.';
  $('#f-alt').textContent = `${num(T.altitude, 1)} m`;
  $('#f-home').textContent = `${num(T.radius, 1)} m`;
  $('#f-age').textContent = !v.present || v.age_ms === null ? '—' : `${v.age_ms.toFixed(0)} ms`;
}

function cells(cmd) {
  return CELLS.map((c) => `<div class="cell"><span class="k">${c.label}</span>
      <span class="v">${num(cmd[c.k], c.d)} ${c.unit}</span></div>`).join('');
}

function fine(sent) {
  return `MAVSDK forward ${num(sent.forward)}, right ${num(sent.right)}, `
       + `down ${num(sent.down)}, yawspeed ${num(sent.yawspeed, 0)}`;
}

// The card has four states and draws all of them: stopped, running, on offer, waiting.
function renderEntry(s) {
  const move = s.move, offer = move && move.offer, running = move && move.running;
  const set = (id, v) => { $(id).textContent = v; };
  const show = (id, on) => { $(id).hidden = !on; };
  const stopped = s.rung === 'estop';

  if (stopped) {
    set('#e-kind', 'Stopped'); set('#e-when', utc(s.wall)); set('#e-big', '—'); set('#e-bigu', '');
    set('#e-say', 'The stop is latched. The aircraft is holding.');
    show('#e-cells', false); show('#e-foot', false); show('#e-wait', true);
    set('#e-wait', 'Clear the stop before running another move.');
  } else if (running) {
    set('#e-kind', 'Running'); set('#e-when', utc(running.wall));
    set('#e-big', num(running.ends_in_s, 1)); set('#e-bigu', 's left');
    set('#e-say', running.text + '.');
    $('#e-cells').innerHTML = cells(running.command);
    show('#e-cells', true); show('#e-foot', false); show('#e-wait', false);
  } else if (offer) {
    set('#e-kind', 'Next move'); set('#e-when', utc(offer.wall));
    set('#e-big', num(offer.hold_s, 1)); set('#e-bigu', 's');
    set('#e-say', offer.text + '.' + (offer.note ? ' ' + offer.note : ''));
    $('#e-cells').innerHTML = cells(offer.command);
    const chip = $('#e-chip');
    chip.dataset.tone = offer.clamped.length ? 'clamped' : 'ok';
    chip.textContent = offer.clamped.length
      ? 'The envelope will hold it: ' + offer.clamped.join('; ')
      : 'Inside the envelope';
    set('#e-fine', fine(offer.sent));
    show('#e-cells', true); show('#e-foot', true); show('#e-wait', true);
    set('#e-wait', `Withdrawn in ${num(offer.expires_in_s, 1)} s if you leave it.`);
  } else if (!move) {
    // No approval gate on this run: the policy is driving the aircraft directly.
    set('#e-kind', 'Flying'); set('#e-when', utc(s.wall));
    set('#e-big', num(s.hz, 1)); set('#e-bigu', 'Hz');
    set('#e-say', 'The policy is flying directly, without approval.');
    $('#e-cells').innerHTML = cells(s.command);
    show('#e-cells', true); show('#e-foot', false); show('#e-wait', false);
  } else {
    set('#e-kind', 'Next move'); set('#e-when', '—'); set('#e-big', '—'); set('#e-bigu', '');
    set('#e-say', 'Waiting for the policy to suggest a move.');
    show('#e-cells', false); show('#e-foot', false); show('#e-wait', false);
  }
  $('#run').disabled = !offer || stopped;
  $('#skip').disabled = !offer;
}

function renderBand(s) {
  const band = $('#band'), move = s.move;
  const running = move && move.running;
  let tone = 'holding', say, fig = '';
  if (s.rung === 'estop') {
    tone = 'stopped';
    say = 'Stopped. The aircraft is hovering and will land when the run ends.';
  } else if (running) {
    tone = 'running';
    say = `Flying: ${running.text.toLowerCase()}.`;
    fig = `${num(running.ends_in_s, 1)} s left`;
  } else if (!move) {
    say = s.reason || 'Flying.';
  } else {
    say = 'Holding. The aircraft moves only when you run a move.';
    if (s.reason && s.rung === 'envelope') say = `Holding. ${s.reason}.`;
  }
  band.dataset.tone = tone;
  $('#band-say').textContent = (s.mode ? s.mode + ': ' : '') + say;
  $('#band-num').textContent = fig;
}

function renderRecord(s) {
  const rows = ((s.move && s.move.record) || []).map((r) => `<tr data-outcome="${esc(r.outcome)}">
      <td class="t">${utc(r.wall)}</td><td>${esc(r.text)}</td>
      <td class="o">${esc(r.outcome)}</td></tr>`).join('');
  $('#record').innerHTML = rows
    || '<tr><td class="t">—</td><td>Nothing has been run yet.</td><td class="o">—</td></tr>';
}

function renderPanel(s) {
  if (!s) return;
  if (s.status === 'starting') {
    $('#band-say').textContent = (s.mode ? s.mode + ': ' : '') + (s.phase || 'Starting.');
    return;
  }
  renderLinks(s); renderCamera(s); renderBand(s); renderEntry(s); renderRecord(s);
  const latched = s.rung === 'estop';
  $('#estop').dataset.latched = latched ? '1' : '0';
  $('#estop').textContent = latched ? 'Clear the stop' : 'E-STOP';
}

// --- driver: the live panel polls the pilot process -------------------------
if (typeof HALLMARK_DEMO === 'undefined') {
  const ONLINE = 'The aircraft holds unless you run a move. Enter runs the one on offer, Esc skips it, the space bar stops the aircraft.';
  const OFFLINE = 'The pilot process is not answering, so nothing here is live. Any move already running will end on its own clock.';

  const post = (path) => fetch(path, { method: 'POST' }).then(poll, () => {});

  async function poll() {
    try {
      const r = await fetch('/api/state', { cache: 'no-store' });
      renderPanel(await r.json());
      $('#foot').dataset.state = 'up';
      $('#foot-say').textContent = ONLINE;
    } catch (e) {
      // The pilot process going away is itself information worth showing.
      $('#foot').dataset.state = 'down';
      $('#foot-say').textContent = OFFLINE;
    }
  }

  const pollFrame = () => { $('#frame').src = '/api/frame.jpg?t=' + Date.now(); };

  $('#run').addEventListener('click', () => post('/api/accept'));
  $('#skip').addEventListener('click', () => post('/api/skip'));
  $('#estop').addEventListener('click', () =>
    post($('#estop').dataset.latched === '1' ? '/api/clear_estop' : '/api/estop'));

  addEventListener('keydown', (e) => {
    if (e.target !== document.body) return;
    // Space is the keyboard stop: the mouse is the slow path when something is wrong.
    if (e.code === 'Space') { e.preventDefault(); post('/api/estop'); }
    if (e.code === 'Enter' && !$('#run').disabled) { e.preventDefault(); post('/api/accept'); }
    if (e.code === 'Escape' && !$('#skip').disabled) { e.preventDefault(); post('/api/skip'); }
  });

  poll(); pollFrame();
  setInterval(poll, POLL_MS);
  setInterval(pollFrame, FRAME_MS);
}
