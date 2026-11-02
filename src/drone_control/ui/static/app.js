// Polls /api/state. Deliberately dumb: all decisions live in the supervisor, this
// file only renders them. Nothing here may ever command the aircraft except E-STOP.
const $ = s => document.querySelector(s);
const POLL_MS = 150;        // ~7 Hz: well under the 20 Hz loop, plenty for an eye
const FRAME_MS = 250;

const AXES = [
  { key: 'vx', ours: 'vx', label: 'forward', sent: 'forward', flip: false, unit: 'm/s', max: 3 },
  { key: 'vy', ours: 'vy', label: 'left', sent: 'right', flip: true, unit: 'm/s', max: 3 },
  { key: 'vz', ours: 'vz', label: 'up', sent: 'down', flip: true, unit: 'm/s', max: 1 },
  { key: 'yaw_rate', ours: 'yaw_rate', label: 'yaw ccw', sent: 'yawspeed', flip: true, unit: '°/s', max: 60 },
];

const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : v.toFixed(d);

function pill(id, ok, present, text) {
  const el = $(id);
  el.className = 'pill ' + (!present ? 'off' : ok ? 'ok' : 'bad');
  el.querySelector('span').textContent = text;
}

function renderAxes(s) {
  const tb = $('#axes');
  if (!tb.children.length) {
    tb.innerHTML = AXES.map(a => `<tr data-a="${a.key}">
      <td>${a.label}${a.flip ? ' <span class="flip">⇄</span>' : ''}</td>
      <td class="num ours">–</td>
      <td><div class="bar"><i></i></div></td>
      <td class="num sent">–</td></tr>`).join('');
  }
  const clamped = (s.clamped || []).length > 0;
  AXES.forEach(a => {
    const row = tb.querySelector(`[data-a="${a.key}"]`);
    const ours = s.command[a.ours], sent = s.sent[a.sent];
    row.querySelector('.ours').textContent = fmt(ours, a.max > 10 ? 1 : 2);
    row.querySelector('.sent').textContent = fmt(sent, a.max > 10 ? 1 : 2);
    // Bar grows from the centre so sign is visible at a glance -- a flipped axis
    // shows up as a bar on the wrong side, which is the bug class this repo keeps hitting.
    const frac = Math.max(-1, Math.min(1, (ours || 0) / a.max));
    const bar = row.querySelector('.bar');
    bar.classList.toggle('clamped', clamped);
    const i = bar.querySelector('i');
    i.style.left = (frac >= 0 ? 50 : 50 + frac * 50) + '%';
    i.style.width = Math.abs(frac) * 50 + '%';
  });
}

function gauge(label, value, limit, unit, d = 1) {
  const frac = limit ? Math.min(1, Math.abs(value) / limit) : 0;
  const cls = frac >= 1 ? 'over' : frac > 0.8 ? 'near' : '';
  return `<div class="gauge ${cls}"><div class="lab"><span>${label}</span>
    <span>${fmt(value, d)} / ${fmt(limit, d)} ${unit}</span></div>
    <div class="track"><div class="fill" style="width:${frac * 100}%"></div></div></div>`;
}

function renderGauges(s) {
  const L = s.limits || {}, T = s.telemetry || {};
  const speed = Math.hypot(s.command.vx || 0, s.command.vy || 0);
  $('#gauges').innerHTML =
    gauge('horizontal speed', speed, L.max_horizontal_speed, 'm/s', 2) +
    gauge('altitude', T.altitude, L.max_altitude, 'm') +
    gauge('distance from home', T.radius, L.max_radius, 'm') +
    gauge('yaw rate', Math.abs(s.command.yaw_rate || 0), L.max_yaw_rate, '°/s', 0);
}

function renderLadder(s) {
  document.querySelectorAll('#ladder .rung[data-rung]').forEach(r => {
    r.classList.toggle('active', r.dataset.rung === s.rung);
  });
}

function renderLog(s) {
  const t0 = s.t;
  $('#log').innerHTML = (s.events || []).map(e =>
    `<li><span class="ts">-${(t0 - e.t).toFixed(1)}s</span>
     <span class="r-${e.rung}">${esc(e.text)}</span></li>`).join('')
    || '<li><span class="ts">–</span><span>no events yet</span></li>';
}
const esc = t => String(t).replace(/[<>&]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]));

function render(s) {
  if (!s || s.status === 'starting') { $('#reason').textContent = 'waiting for the control loop…'; return; }
  const v = s.links.video, p = s.links.policy, m = s.links.mavlink;
  pill('#l-mavlink', m.ok, true, m.ok ? 'telemetry ok' : 'no telemetry');
  pill('#l-video', v.ok, v.present,
    !v.present ? 'not used' : v.age_ms === null ? 'no frames' : `${v.age_ms.toFixed(0)} ms · ${v.frames} f`);
  pill('#l-policy', p.ok, p.present,
    !p.present ? 'not used' : `${p.calls} calls · ${p.failures} fail`);

  renderLadder(s); renderAxes(s); renderGauges(s); renderLog(s);
  $('#reason').textContent = s.reason || '';
  $('#src').textContent = s.source ? `· ${s.source}` : '';
  const cl = $('#clamped');
  cl.hidden = !(s.clamped || []).length;
  cl.textContent = (s.clamped || []).join(' · ');
  $('#rate').textContent = s.hz ? `control loop ${s.hz.toFixed(1)} Hz · elapsed ${fmt(s.elapsed, 0)}s` : '';
  $('#estop').classList.toggle('armed', s.rung === 'estop');
  $('#estop').textContent = s.rung === 'estop' ? 'E-STOPPED' : 'E-STOP';

  const stale = v.present && !v.ok;
  $('#nosig').style.display = (!v.present || stale || v.age_ms === null) ? 'grid' : 'none';
  $('#nosig').textContent = !v.present ? 'CAMERA NOT IN USE' : 'NO SIGNAL — frame stale, policy not asked';
  $('#camage').textContent = v.age_ms === null ? '' : `frame ${v.age_ms.toFixed(0)} ms old`;
}

async function poll() {
  try {
    const r = await fetch('/api/state', { cache: 'no-store' });
    render(await r.json());
    $('#foot').classList.remove('offline');
    $('#foot').textContent = 'localhost only · the panel observes the supervisor; its only command is the e-stop';
  } catch (e) {
    // The pilot process going away is itself information worth showing.
    $('#foot').classList.add('offline');
    $('#foot').textContent = 'pilot process not reachable — the drone is NOT being commanded from here';
  }
}

function pollFrame() {
  const img = $('#frame');
  img.src = '/api/frame.jpg?t=' + Date.now();
}

$('#estop').addEventListener('click', async () => {
  const stopped = $('#estop').classList.contains('armed');
  await fetch(stopped ? '/api/clear_estop' : '/api/estop', { method: 'POST' });
  poll();
});
// Space is the keyboard e-stop: the mouse is the slow path when something goes wrong.
addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); fetch('/api/estop', { method: 'POST' }).then(poll); }
});

poll(); pollFrame();
setInterval(poll, POLL_MS);
setInterval(pollFrame, FRAME_MS);
