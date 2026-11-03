// Renders /api/state. Deliberately dumb: every decision lives in the supervisor and
// this file only draws it. Nothing here may ever command the aircraft except the stop.
//
// `renderPanel` is kept separate from the polling driver at the foot of the file so the
// simulated-flight demo can drive the same markup without a control loop behind it.
const $ = (s) => document.querySelector(s);
const POLL_MS = 150;        // about 7 Hz, well under the 20 Hz loop and plenty for an eye
const FRAME_MS = 250;

const AXES = [
  { ours: 'vx', name: 'Forward', sent: 'forward', sentName: 'MAVSDK forward', flips: false, unit: 'm/s', max: 3, d: 2 },
  { ours: 'vy', name: 'Left', sent: 'right', sentName: 'MAVSDK right', flips: true, unit: 'm/s', max: 3, d: 2 },
  { ours: 'vz', name: 'Up', sent: 'down', sentName: 'MAVSDK down', flips: true, unit: 'm/s', max: 1, d: 2 },
  { ours: 'yaw_rate', name: 'Yaw left', sent: 'yawspeed', sentName: 'MAVSDK yawspeed, clockwise', flips: true, unit: '°/s', max: 60, d: 0 },
];

const RUNGS = { estop: 'stop', envelope: 'envelope', stale: 'stale', policy: 'policy', idle: 'hover' };

const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : v.toFixed(d);
const esc = (t) => String(t).replace(/[<>&]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]));
const utc = (wall) => (wall ? new Date(wall * 1000).toISOString().slice(11, 19) : '—');

function buildAxes() {
  $('#axes').innerHTML = AXES.map((a) => `<li class="axis" data-a="${a.ours}">
      <div class="axis__n">${a.name}${a.flips ? ' <span class="axis__flip">sign inverts</span>' : ''}</div>
      <div class="axis__v measure"><span class="val">0.00</span><span class="unit">${a.unit}</span></div>
      <div class="axis__bar"><i></i></div>
      <div class="axis__sent">${a.sentName} <span class="measure sv">0.00</span> ${a.unit}</div>
    </li>`).join('');
}

function renderAxes(s) {
  const clamped = (s.clamped || []).length > 0;
  AXES.forEach((a) => {
    const row = $(`.axis[data-a="${a.ours}"]`);
    const ours = s.command[a.ours] || 0, sent = s.sent[a.sent] || 0;
    row.querySelector('.val').textContent = num(ours, a.d);
    row.querySelector('.sv').textContent = num(sent, a.d);
    // The bar grows from the centre, so a flipped sign shows up as a bar on the wrong
    // side: the bug class this repo keeps shipping, visible before takeoff.
    const frac = Math.max(-1, Math.min(1, ours / a.max));
    const bar = row.querySelector('.axis__bar');
    bar.classList.toggle('is-neg', frac < 0);
    bar.classList.toggle('is-clamped', clamped);
    bar.querySelector('i').style.transform = `scaleX(${Math.abs(frac)})`;
  });
}

function envRow(label, value, bound, unit, d) {
  const frac = bound ? Math.min(1, Math.abs(value) / bound) : 0;
  return `<tr><td>${label}</td>
    <td class="num">${num(value, d)}</td>
    <td class="num dim">${num(bound, d)} ${unit}</td>
    <td class="track" data-near="${frac > 0.8 ? 1 : 0}"><div class="bar"><i style="transform:scaleX(${frac})"></i></div></td></tr>`;
}

function renderEnv(s) {
  const L = s.limits || {}, T = s.telemetry || {}, c = s.command;
  $('#env').innerHTML =
    envRow('Horizontal speed', Math.hypot(c.vx || 0, c.vy || 0), L.max_horizontal_speed, 'm/s', 2) +
    envRow('Vertical speed', Math.abs(c.vz || 0), L.max_vertical_speed, 'm/s', 2) +
    envRow('Yaw rate', Math.abs(c.yaw_rate || 0), L.max_yaw_rate, '°/s', 0) +
    envRow(`Altitude, floor ${num(L.min_altitude, 1)} m`, T.altitude, L.max_altitude, 'm', 1) +
    envRow('Distance from home', T.radius, L.max_radius, 'm', 1);
}

function renderRungs(s) {
  document.querySelectorAll('#rungs .rung[data-rung]').forEach((r) => {
    if (r.dataset.rung === s.rung) r.setAttribute('aria-current', 'true');
    else r.removeAttribute('aria-current');
  });
  const name = RUNGS[s.rung] || s.rung;
  $('#reason').innerHTML = `<b>${esc(name)}.</b> ${esc(s.reason || '')}`;
}

function renderLog(s) {
  const rows = (s.events || []).map((e) => `<tr data-rung="${esc(e.rung)}">
      <td class="t measure">${utc(e.wall)}</td>
      <td class="dim">${esc(RUNGS[e.rung] || e.rung)}</td>
      <td class="r">${esc(e.text)}</td></tr>`).join('');
  $('#log').innerHTML = rows || '<tr><td class="t measure">waiting</td><td class="dim">none</td><td class="r">Nothing has happened yet.</td></tr>';
}

function renderLinks(s) {
  const m = s.links.mavlink, v = s.links.video, p = s.links.policy;

  $('#l-mavlink').dataset.state = m.ok ? 'up' : 'down';
  $('#v-mavlink').textContent = m.ok ? 'receiving' : 'no telemetry';

  const vState = !v.present ? 'off' : (v.age_ms === null ? 'down' : (v.ok ? 'up' : 'down'));
  $('#l-video').dataset.state = vState;
  $('#v-video').textContent = !v.present ? 'not in use'
    : v.age_ms === null ? 'no frames' : `${v.age_ms.toFixed(0)} ms`;

  $('#l-policy').dataset.state = !p.present ? 'off' : (p.ok ? 'up' : 'down');
  $('#v-policy').textContent = !p.present ? 'not in use'
    : `${p.calls} calls, ${p.failures} failed`;
}

function renderCamera(s) {
  const v = s.links.video, cam = $('#cam'), over = $('#camover');
  const state = !v.present ? 'off' : (v.age_ms === null ? 'waiting' : (v.ok ? 'live' : 'stale'));
  cam.dataset.state = state;
  over.hidden = state === 'live';
  over.textContent = state === 'off' ? 'No camera on this run.'
    : state === 'waiting' ? 'Waiting for the first frame.'
    : 'Signal lost. The frame is stale, so the policy is not being asked.';
  $('#c-age').textContent = v.age_ms === null || !v.present ? 'no frames' : `${v.age_ms.toFixed(0)} ms`;
  $('#c-frames').textContent = v.frames ?? 0;
  $('#c-recon').textContent = v.reconnects ?? 0;
}

function renderPanel(s) {
  if (!s || s.status === 'starting') return;
  $('#src').textContent = s.source || 'no source';
  renderLinks(s); renderAxes(s); renderRungs(s); renderEnv(s); renderLog(s); renderCamera(s);
  const latched = s.rung === 'estop';
  const stop = $('#estop');
  stop.dataset.latched = latched ? '1' : '0';
  stop.textContent = latched ? 'Clear the stop' : 'E-STOP';
  $('#f-hz').textContent = s.hz ? `${s.hz.toFixed(1)} Hz` : 'waiting';
  $('#f-elapsed').textContent = `${num(s.elapsed, 0)} s`;
}

// --- driver: the live panel polls the pilot process -------------------------
if (typeof HALLMARK_DEMO === 'undefined') {
  buildAxes();

  const ONLINE = 'Runs on this laptop only. The panel watches the chain of order; the one order it can give is the stop, on this button or the space bar.';
  const OFFLINE = 'The pilot process is not answering, so nothing on this page is live. The drone is not being commanded from here.';

  async function poll() {
    try {
      const r = await fetch('/api/state', { cache: 'no-store' });
      renderPanel(await r.json());
      $('#foot').dataset.state = 'up';
      $('#foot').querySelector('p').textContent = ONLINE;
    } catch (e) {
      // The pilot process going away is itself information worth showing.
      $('#foot').dataset.state = 'down';
      $('#foot').querySelector('p').textContent = OFFLINE;
    }
  }

  const pollFrame = () => { $('#frame').src = '/api/frame.jpg?t=' + Date.now(); };

  $('#estop').addEventListener('click', async () => {
    const latched = $('#estop').dataset.latched === '1';
    await fetch(latched ? '/api/clear_estop' : '/api/estop', { method: 'POST' });
    poll();
  });
  // Space is the keyboard stop: the mouse is the slow path when something is going wrong.
  addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) {
      e.preventDefault();
      fetch('/api/estop', { method: 'POST' }).then(poll);
    }
  });

  poll(); pollFrame();
  setInterval(poll, POLL_MS);
  setInterval(pollFrame, FRAME_MS);
}
