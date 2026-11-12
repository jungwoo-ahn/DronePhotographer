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

// A value that rounds to zero prints as 0, not -0.
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '—'
  : (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d);
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
  $('#v-policy').textContent = !p.present ? 'not in use'
    : (s.model && s.model.asking) ? 'thinking' : `${p.calls} asked`;
  const rec = s.recording;
  $('#l-rec').hidden = !rec;
  if (rec) {
    const t = Math.floor(rec.seconds || 0);
    $('#l-rec').dataset.state = rec.recording ? 'rec' : 'down';
    $('#v-rec').textContent = rec.recording
      ? `● ${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`
      : (rec.error ? 'stopped' : 'starting');
    $('#l-rec').title = rec.error || `${rec.dir}\\${rec.file || ''}`;
  }
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
    set('#e-wait', offer.expires_in_s === null
      ? 'It waits for you: the aircraft holds still, so the view it was worked out from stays true.'
      : `Withdrawn in ${num(offer.expires_in_s, 1)} s if you leave it.`);
  } else if (!move) {
    // No approval gate on this run: the policy is driving the aircraft directly.
    set('#e-kind', 'Flying'); set('#e-when', utc(s.wall));
    set('#e-big', num(s.hz, 1)); set('#e-bigu', 'Hz');
    set('#e-say', 'The policy is flying directly, without approval.');
    $('#e-cells').innerHTML = cells(s.command);
    show('#e-cells', true); show('#e-foot', false); show('#e-wait', false);
  } else {
    set('#e-kind', 'Next move'); set('#e-when', '—'); set('#e-big', '—'); set('#e-bigu', '');
    set('#e-say', s.model && s.model.asking
      ? 'The model is working out the next moves from the camera\'s view.'
      : 'Waiting for the policy to suggest a move.');
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

// --- the shot: what the model is asked for ------------------------------------
// Profile positions are in the renders' 1024x768 pixels, which is also the view sent to
// the model, so they draw straight over "what the model sees".
const FRAME_W = 1024, FRAME_H = 768;
const SIDES = ['front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left'];
let SHOTS = [];
const side = (bearing) => SIDES[Math.floor((((bearing + 22.5) % 360) + 360) % 360 / 45)];
const svgEl = (tag, attrs, text) => {
  const a = Object.entries(attrs).map(([k, v]) => `${k}="${v}"`).join(' ');
  return text === undefined ? `<${tag} ${a}/>` : `<${tag} ${a}>${text}</${tag}>`;
};

const shotImage = (s) => '/api/shot.jpg?id=' + encodeURIComponent(s.id);

// The choices, as the photos they lead to: one row per side of the subject.
function fillShots(shots) {
  SHOTS = shots || [];
  const groups = [];
  for (const s of SHOTS) {
    let g = groups.find((x) => x.name === s.group);
    if (!g) groups.push(g = { name: s.group, shots: [] });
    g.shots.push(s);
  }
  $('#pick-n').textContent = `(${SHOTS.length})`;
  $('#pick-groups').innerHTML = groups.map((g) => `<div class="pick__row">
      <span class="pick__side">${esc(g.name.replace(/^From the /, ''))}</span>
      <div class="pick__shots">${g.shots.map((s) => `<button class="pick__shot" type="button"
          data-id="${esc(s.id)}" title="${esc(s.name)}" aria-pressed="false">${s.image
            ? `<img src="${shotImage(s)}" alt="" loading="lazy" width="128" height="96">`
            : `<span class="pick__text">${esc(s.name)}</span>`}
          <span class="pick__occ num">${num(s.profile.occupancy, 0)}%</span></button>`).join('')}
      </div></div>`).join('');
}

function renderShot(s) {
  const wrap = $('#shotwrap');
  wrap.hidden = !s.model || !SHOTS.length;
  if (wrap.hidden) return;
  const shot = SHOTS.find((x) => x.id === s.shot);
  if (!shot) return;
  if (wrap.dataset.shown === shot.id) return;           // drawn already
  wrap.dataset.shown = shot.id;
  for (const b of document.querySelectorAll('.pick__shot'))
    b.setAttribute('aria-pressed', String(b.dataset.id === shot.id));

  const p = shot.profile;
  const img = $('#goal-img');
  img.hidden = !shot.image; $('#goal-none').hidden = !!shot.image;
  if (shot.image) { img.src = shotImage(shot); img.alt = `The goal photo: ${shot.name}`; }
  $('#shot-name').textContent = shot.name;
  const [words, nums] = shot.sentence.replace('Move the camera to achieve this shot: ', '').split(' (');
  $('#shot-say').textContent = words.charAt(0).toUpperCase() + words.slice(1);
  $('#shot-nums').textContent = nums ? 'The model reads: ' + nums.replace(/\)$/, '') : '';
  const cell = (k, v) => `<div class="cell"><span class="k">${k}</span><span class="v">${v}</span></div>`;
  $('#shot-cells').innerHTML = [
    cell('Subject fills', `${num(p.occupancy, 0)}% of the frame`),
    cell('In the frame', `${num(p.body_in_frame_ratio, 0)}% of it`),
    cell('From its', `${side(p.subject_bearing_deg)}, ${num(p.subject_bearing_deg, 0)}°`),
    cell('Camera', p.cam_to_obj_elevation_deg < 0
      ? `${num(-p.cam_to_obj_elevation_deg, 0)}° above` : `${num(p.cam_to_obj_elevation_deg, 0)}° below`),
  ].join('');

  const box = $('#target');
  const pct = (v, of) => `${(100 * v / of).toFixed(2)}%`;
  box.style.left = pct(p.object_center_x - p.bbox_x_offset, FRAME_W);
  box.style.top = pct(p.object_center_y - p.bbox_y_offset, FRAME_H);
  box.style.width = pct(2 * p.bbox_x_offset, FRAME_W);
  box.style.height = pct(2 * p.bbox_y_offset, FRAME_H);
  box.hidden = false; $('#target-note').hidden = false;
}

// --- the model, and where its moves lead -----------------------------------------
function renderModel(s) {
  const m = s.model, say = $('#model-say');
  if (!m) return;
  let state = 'idle', text = 'The model has not been asked yet.';
  if (m.asking) {
    state = 'thinking';
    text = `The model is working out the next moves: ${num(m.asking_for_s, 0)} s`
      + (m.last_pass_s ? `, the last took ${num(m.last_pass_s, 0)} s.` : '. The first one is slow.');
  } else if (m.plan.length) {
    const ago = m.answered_wall ? Math.max(0, Date.now() / 1000 - m.answered_wall) : null;
    text = `The model answered in ${num(m.last_pass_s, 1)} s, ${num(ago, 0)} s ago, with `
      + `${m.plan.length} moves. ${m.passes} forward passes so far.`;
  }
  if (m.error && !m.asking) { state = 'error'; text = `The last request failed: ${m.error}`; }
  say.dataset.state = state;
  $('#model-text').textContent = text;
}

function niceStep(span) {
  for (const st of [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10]) if (span / st <= 6) return st;
  return 20;
}

function renderPlan(s) {
  const m = s.model, wrap = $('#planwrap');
  wrap.hidden = !m;
  if (!m) return;
  const plan = m.plan || [];
  const key = JSON.stringify([plan.map((p) => [p.x, p.y]), m.offered, m.dropped]);
  if (wrap.dataset.key === key) return;
  wrap.dataset.key = key;

  const state = (n) => (n < m.offered ? 'done' : n === m.offered ? 'now' : m.dropped ? 'dropped' : 'next');
  $('#plan-steps').innerHTML = plan.map((p) => `<li data-state="${state(p.n)}">
      <span class="steps__n num">${p.n}</span><span class="steps__t">${esc(p.text)}</span>
      <span class="steps__v num">${num(p.tilt, 0)}°</span></li>`).join('')
    || '<li data-state="next"><span class="steps__t">No moves yet.</span></li>';

  // From above, the aircraft at the start facing up the page: x forward is up, y left is left.
  const pts = [{ n: 0, x: 0, y: 0, heading_deg: 0 }, ...plan].map((p) => ({ ...p, sx: -p.y, sy: -p.x }));
  const xs = pts.map((p) => p.sx), ys = pts.map((p) => p.sy);
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 0.3);
  const k = 170 / span, midx = (Math.max(...xs) + Math.min(...xs)) / 2, midy = (Math.max(...ys) + Math.min(...ys)) / 2;
  const X = (v) => 120 + (v - midx) * k, Y = (v) => 120 + (v - midy) * k;
  const step = niceStep(span), out = [];
  for (let g = Math.ceil((midx - 120 / k) / step) * step; g <= midx + 120 / k; g += step)
    out.push(svgEl('line', { class: 'plan__grid', x1: X(g), y1: 0, x2: X(g), y2: 240 }));
  for (let g = Math.ceil((midy - 120 / k) / step) * step; g <= midy + 120 / k; g += step)
    out.push(svgEl('line', { class: 'plan__grid', x1: 0, y1: Y(g), x2: 240, y2: Y(g) }));
  out.push(svgEl('polyline', { class: 'plan__path', points: pts.map((p) => `${X(p.sx)},${Y(p.sy)}`).join(' ') }));
  for (const p of pts) {
    const h = p.heading_deg * Math.PI / 180, x = X(p.sx), y = Y(p.sy);
    if (p.n === 0) {
      out.push(svgEl('path', { class: 'plan__drone', d: `M${x} ${y - 9} L${x + 7} ${y + 6} L${x - 7} ${y + 6} Z` }));
      continue;
    }
    out.push(svgEl('line', { class: 'plan__head', x1: x, y1: y, x2: x - 12 * Math.sin(h), y2: y - 12 * Math.cos(h) }));
    out.push(svgEl('circle', { class: `plan__dot plan__dot--${state(p.n)}`, cx: x, cy: y, r: 7 }));
    out.push(svgEl('text', { class: 'plan__n', x, y: y + 3.5, 'text-anchor': 'middle' }, p.n));
  }
  $('#plan-map').innerHTML = out.join('');

  const last = plan[plan.length - 1];
  const dir = (v, pos, neg) => `${num(Math.abs(v), 2)} m ${v >= 0 ? pos : neg}`;
  $('#plan-fine').textContent = last
    ? `Grid squares are ${step * 100} cm. After all ${plan.length} moves: ${dir(last.x, 'forward', 'back')}, `
      + `${dir(last.y, 'left', 'right')}, ${dir(last.z, 'up', 'down')}, turned `
      + `${num(Math.abs(last.heading_deg), 0)}° ${last.heading_deg >= 0 ? 'left' : 'right'}, `
      + `camera at ${num(last.tilt, 0)}°. Each move is ${num(m.move_s, 1)} s.`
    : '';
}

// --- photos: frames saved on the laptop --------------------------------------------
function renderPhotos(s) {
  const p = s.photos;
  $('#photo').hidden = !p;
  const last = p && p.last, box = $('#shotlast');
  box.hidden = !last;
  if (!last || box.dataset.name === last.name) return;
  box.dataset.name = last.name;
  $('#lastphoto').src = '/api/photo.jpg?name=' + encodeURIComponent(last.name);
  const size = last.size ? ` at ${last.size[0]}x${last.size[1]}` : '';
  $('#lastphoto-say').textContent = `${p.count} photo${p.count === 1 ? '' : 's'} taken. `
    + `The last at ${utc(last.wall)} UTC${size}.`;
  $('#lastphoto-where').textContent = `${last.dir}\\${last.files.photo}`;
}

function renderPanel(s) {
  if (!s) return;
  if (s.status === 'starting') {
    $('#band-say').textContent = (s.mode ? s.mode + ': ' : '') + (s.phase || 'Starting.');
    return;
  }
  renderLinks(s); renderCamera(s); renderBand(s); renderEntry(s); renderRecord(s);
  renderShot(s); renderModel(s); renderPlan(s); renderPhotos(s);
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

  const pollFrame = () => {
    $('#frame').src = '/api/frame.jpg?t=' + Date.now();
    $('#modelframe').src = '/api/model_frame.jpg?t=' + Date.now();
  };

  $('#run').addEventListener('click', () => post('/api/accept'));
  $('#skip').addEventListener('click', () => post('/api/skip'));
  const LABEL = $('#photo').innerHTML;
  const takePhoto = () => {
    const btn = $('#photo');
    const say = (t) => { btn.textContent = t; setTimeout(() => { btn.innerHTML = LABEL; }, 1200); };
    fetch('/api/photo', { method: 'POST' }).then((r) => r.json())
      .then((r) => { say(r.taken ? 'Saved' : 'No frame to save'); poll(); }, () => say('Not saved'));
  };
  $('#photo').addEventListener('click', takePhoto);
  addEventListener('keydown', (e) => {
    if (e.target === document.body && e.code === 'KeyP' && !$('#photo').hidden) {
      e.preventDefault(); takePhoto();
    }
  });
  // Open unless this browser closed it last time.
  try { if (localStorage.getItem('pick-open') === '0') $('#pick').open = false; } catch (e) { /* none */ }
  $('#pick').addEventListener('toggle', () => {
    try { localStorage.setItem('pick-open', $('#pick').open ? '1' : '0'); } catch (e) { /* none */ }
  });
  $('#pick-groups').addEventListener('click', (e) => {
    const b = e.target.closest('.pick__shot');
    if (!b) return;
    post('/api/shot?id=' + encodeURIComponent(b.dataset.id));
    b.blur();               // hand the keys back: Enter, Esc and the space bar work on the page
  });
  const loadShots = () => fetch('/api/shots', { cache: 'no-store' })
    .then((r) => r.json()).then(fillShots, () => setTimeout(loadShots, 2000));
  loadShots();
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
