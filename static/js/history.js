// static/js/history.js
// Session history, analytics and chart rendering.
//
// Sessions live in localStorage on the user's own device, so the Flask
// app stays stateless and nothing extra is needed to deploy this.

const STORE_KEY = 'shapeform.sessions.v1';
const MAX_SESSIONS = 300;

// ---- Reference data --------------------------------------------------------
export const EXERCISE_META = {
  squat: { label: 'Squat', icon: 'bi-person-standing' },
  lunge: { label: 'Lunge', icon: 'bi-person-walking' },
  pushup: { label: 'Push-up', icon: 'bi-person-arms-up' },
  deadlift: { label: 'Deadlift', icon: 'bi-arrow-bar-up' },
  bicep_curl: { label: 'Bicep curl', icon: 'bi-arrow-repeat' }
};

const MODE_LABEL = { webcam: 'Live', video: 'Video', image: 'Photo' };

// Model class names rewritten as something a person would say.
const FAULT_LABELS = {
  extreme_backward_lean: 'Leaning backward',
  extreme_forward_lean: 'Leaning forward',
  foots_too_close: 'Stance too narrow',
  foots_too_far: 'Stance too wide',
  hand_too_far_or_incorrect_position: 'Hand placement off',
  hips_too_high: 'Hips riding high',
  incorrect_leg_position: 'Leg alignment off',
  back_arch_posture: 'Back arching',
  hand_grip_width: 'Grip width off',
  leg_position_width: 'Stance width off',
  back_too_backward_lean: 'Leaning backward',
  back_too_forward_lean: 'Leaning forward',
  hand_position_too_close: 'Hands too close',
  hand_position_too_wide: 'Hands too wide',
  hand_above_near_head: 'Curling too high',
  one_hand_up_other_down: 'Arms out of sync',
  unknown: 'Unclear position'
};

export const faultLabel = (key) =>
  FAULT_LABELS[key] || String(key || '').replace(/_/g, ' ');

const exerciseLabel = (key) =>
  (EXERCISE_META[key] || {}).label || key || 'Session';

const exerciseIcon = (key) =>
  (EXERCISE_META[key] || {}).icon || 'bi-activity';

// ---- Store -----------------------------------------------------------------
export function loadSessions() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function persist(sessions) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(sessions.slice(0, MAX_SESSIONS)));
    return true;
  } catch {
    return false;
  }
}

export function saveSession(session) {
  const sessions = loadSessions();
  sessions.unshift(session);
  persist(sessions);
  return session;
}

export function deleteSession(id) {
  persist(loadSessions().filter(s => s.id !== id));
}

export function clearSessions() {
  try { localStorage.removeItem(STORE_KEY); } catch { /* ignore */ }
}

// ---- Formatting ------------------------------------------------------------
export function formatDuration(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

export function formatClock(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

const startOfDay = (d) => {
  const x = new Date(d);
  x.setHours(0, 0, 0, 0);
  return x;
};

const dayKey = (d) => startOfDay(d).toISOString().slice(0, 10);

function formatWhen(iso) {
  const then = new Date(iso);
  const days = Math.round((startOfDay(new Date()) - startOfDay(then)) / 86400000);
  const time = then.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  if (days === 0) return `Today, ${time}`;
  if (days === 1) return `Yesterday, ${time}`;
  if (days < 7) return `${then.toLocaleDateString([], { weekday: 'long' })}, ${time}`;
  return then.toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric' });
}

// ---- Analytics -------------------------------------------------------------
export function computeTotals(sessions) {
  const t = {
    sessions: sessions.length,
    reps: 0,
    good: 0,
    bad: 0,
    seconds: 0,
    accuracy: 0,
    faults: {},
    byExercise: {},
    bestSession: null
  };

  sessions.forEach(s => {
    t.reps += s.reps || 0;
    t.good += s.good || 0;
    t.bad += s.bad || 0;
    t.seconds += s.durationSec || 0;

    Object.entries(s.faults || {}).forEach(([k, n]) => {
      t.faults[k] = (t.faults[k] || 0) + n;
    });

    const ex = t.byExercise[s.exercise] || { reps: 0, good: 0, sessions: 0, best: 0 };
    ex.reps += s.reps || 0;
    ex.good += s.good || 0;
    ex.sessions += 1;
    ex.best = Math.max(ex.best, s.reps || 0);
    t.byExercise[s.exercise] = ex;

    if (!t.bestSession || (s.reps || 0) > (t.bestSession.reps || 0)) t.bestSession = s;
  });

  t.accuracy = t.reps ? Math.round((t.good / t.reps) * 100) : 0;
  return t;
}

export function personalBests(sessions) {
  const best = {};
  sessions.forEach(s => {
    if (!s.exercise) return;
    if (!best[s.exercise] || (s.reps || 0) > best[s.exercise]) best[s.exercise] = s.reps || 0;
  });
  return best;
}

export function currentStreak(sessions) {
  if (!sessions.length) return 0;
  const days = new Set(sessions.map(s => dayKey(s.startedAt)));
  const today = startOfDay(new Date());

  // A streak stays alive if you trained today or yesterday.
  let cursor = new Date(today);
  if (!days.has(dayKey(cursor))) {
    cursor.setDate(cursor.getDate() - 1);
    if (!days.has(dayKey(cursor))) return 0;
  }

  let streak = 0;
  while (days.has(dayKey(cursor))) {
    streak += 1;
    cursor.setDate(cursor.getDate() - 1);
  }
  return streak;
}

/**
 * Bucket reps into chart columns.
 * week  -> last 7 days
 * month -> last 6 weeks
 * year  -> last 12 months
 */
export function bucketReps(sessions, range) {
  const now = new Date();
  const buckets = [];

  if (range === 'year') {
    for (let i = 11; i >= 0; i--) {
      const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
      buckets.push({
        label: d.toLocaleDateString([], { month: 'short' }),
        start: d,
        end: new Date(d.getFullYear(), d.getMonth() + 1, 1),
        value: 0,
        isNow: i === 0
      });
    }
  } else if (range === 'month') {
    const thisWeekStart = startOfDay(now);
    thisWeekStart.setDate(thisWeekStart.getDate() - ((thisWeekStart.getDay() + 6) % 7));
    for (let i = 5; i >= 0; i--) {
      const start = new Date(thisWeekStart);
      start.setDate(start.getDate() - i * 7);
      const end = new Date(start);
      end.setDate(end.getDate() + 7);
      buckets.push({
        label: start.toLocaleDateString([], { day: 'numeric', month: 'short' }),
        start,
        end,
        value: 0,
        isNow: i === 0
      });
    }
  } else {
    for (let i = 6; i >= 0; i--) {
      const start = startOfDay(now);
      start.setDate(start.getDate() - i);
      const end = new Date(start);
      end.setDate(end.getDate() + 1);
      buckets.push({
        label: start.toLocaleDateString([], { weekday: 'short' }),
        start,
        end,
        value: 0,
        isNow: i === 0
      });
    }
  }

  sessions.forEach(s => {
    const when = new Date(s.startedAt);
    const bucket = buckets.find(b => when >= b.start && when < b.end);
    if (bucket) bucket.value += s.reps || 0;
  });

  return buckets;
}

// ---- Rendering helpers -----------------------------------------------------
const $ = (id) => document.getElementById(id);
const esc = (str) => String(str).replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function accuracyClass(pct) {
  if (pct >= 80) return '';
  if (pct >= 55) return ' is-mid';
  return ' is-low';
}

/** Lifetime counters in the app bar. */
export function renderRail(sessions) {
  const t = computeTotals(sessions);
  $('rail-sessions').textContent = t.sessions;
  $('rail-reps').textContent = t.reps;
  $('rail-streak').textContent = currentStreak(sessions);
  $('tab-history-count').textContent = t.sessions;
}

/** Personal-best flags on the exercise picker. */
export function renderPersonalBests(sessions) {
  const best = personalBests(sessions);
  document.querySelectorAll('[data-pb]').forEach(el => {
    const reps = best[el.dataset.pb];
    if (reps) {
      el.innerHTML = `<i class="bi bi-trophy-fill"></i>Best ${reps} reps`;
      el.classList.remove('hidden');
    } else {
      el.classList.add('hidden');
    }
  });
}

/** The History tab. */
export function renderHistory(sessions, filter = 'all') {
  const list = $('session-list');
  const empty = $('history-empty');
  const rows = filter === 'all' ? sessions : sessions.filter(s => s.exercise === filter);

  $('history-sub').textContent = sessions.length
    ? `${sessions.length} session${sessions.length === 1 ? '' : 's'} logged, newest first.`
    : 'Every set you have logged, newest first.';

  if (!rows.length) {
    list.innerHTML = '';
    empty.classList.remove('hidden');
    empty.querySelector('h3').textContent =
      sessions.length ? 'Nothing matches that filter' : 'No sessions yet';
    empty.querySelector('p').textContent = sessions.length
      ? 'Try another exercise, or pick All to see everything you have logged.'
      : 'Run a set from the Train tab. When you stop the camera, the session lands here with your reps, accuracy and the faults you hit.';
    return;
  }

  empty.classList.add('hidden');
  list.innerHTML = rows.map(s => {
    const acc = s.reps ? Math.round((s.good / s.reps) * 100) : 0;
    const topFault = Object.entries(s.faults || {}).sort((a, b) => b[1] - a[1])[0];
    return `
      <article class="session-card">
        <div class="session-icon"><i class="bi ${exerciseIcon(s.exercise)}"></i></div>
        <div class="session-main">
          <div class="session-title">
            ${esc(exerciseLabel(s.exercise))}
            <span class="session-mode">${esc(MODE_LABEL[s.mode] || s.mode || 'Live')}</span>
          </div>
          <div class="session-meta">
            <span><i class="bi bi-calendar3"></i>${esc(formatWhen(s.startedAt))}</span>
            <span><i class="bi bi-stopwatch"></i>${esc(formatDuration(s.durationSec))}</span>
            ${topFault
        ? `<span><i class="bi bi-exclamation-triangle"></i>Mostly ${esc(faultLabel(topFault[0]))}</span>`
        : '<span><i class="bi bi-check2-circle"></i>No faults flagged</span>'}
          </div>
        </div>
        <div class="session-figures">
          <div class="session-figure">
            <div class="session-figure-value">${s.reps || 0}</div>
            <div class="session-figure-label">Reps</div>
          </div>
          <div class="session-figure">
            <div class="session-figure-value">${s.good || 0}</div>
            <div class="session-figure-label">Clean</div>
          </div>
          <span class="accuracy-pill${accuracyClass(acc)}">${acc}%</span>
        </div>
      </article>`;
  }).join('');
}

/** The Progress tab. */
export function renderProgress(sessions, range = 'week') {
  const t = computeTotals(sessions);
  const panels = document.querySelectorAll('#view-progress > .panel-grid, #view-progress > .panel');
  const empty = $('progress-empty');

  // Headline numbers
  $('hs-time').innerHTML = `${Math.round(t.seconds / 60)}<small>min</small>`;
  $('hs-reps').textContent = t.reps;
  $('hs-accuracy').innerHTML = `${t.accuracy}<small>%</small>`;
  $('hs-sessions').textContent = t.sessions;

  const streak = currentStreak(sessions);
  if (!t.sessions) {
    $('progress-headline').textContent = 'Nothing logged yet';
    $('progress-subline').textContent = 'Finish a session and your numbers show up here.';
  } else if (streak > 1) {
    $('progress-headline').textContent = `${streak} days in a row`;
    $('progress-subline').textContent = 'Keep the streak going, consistency is what moves form.';
  } else {
    $('progress-headline').textContent = 'Great work so far';
    $('progress-subline').textContent = 'Your lifetime totals across every logged session.';
  }

  if (!t.sessions) {
    panels.forEach(p => p.classList.add('hidden'));
    empty.classList.remove('hidden');
    return;
  }
  panels.forEach(p => p.classList.remove('hidden'));
  empty.classList.add('hidden');

  renderVolumeChart(sessions, range);
  renderAccuracyRing(t);
  renderStreakStrip(sessions);
  renderFaults(t);
  renderBreakdown(t);
}

function renderVolumeChart(sessions, range) {
  const buckets = bucketReps(sessions, range);
  const peak = Math.max(1, ...buckets.map(b => b.value));

  $('chart-sub').textContent = {
    week: 'Reps completed per day',
    month: 'Reps completed per week',
    year: 'Reps completed per month'
  }[range];

  $('volume-chart').innerHTML = buckets.map(b => {
    const pct = Math.round((b.value / peak) * 100);
    return `
      <div class="chart-col${b.isNow ? ' is-today' : ''}">
        <div class="chart-bar-wrap">
          <div class="chart-bar${b.value ? '' : ' is-empty'}" style="height:${b.value ? Math.max(pct, 4) : 4}%">
            <span class="chart-bar-value">${b.value}</span>
          </div>
        </div>
        <div class="chart-label">${esc(b.label)}</div>
      </div>`;
  }).join('');

  const top = buckets.reduce((a, b) => (b.value > a.value ? b : a), buckets[0]);
  $('chart-peak').textContent = top && top.value
    ? `Peak ${top.value} reps (${top.label})`
    : 'No reps in this range yet';
}

function renderAccuracyRing(t) {
  const circumference = 2 * Math.PI * 52;
  const pct = Math.max(0, Math.min(100, t.accuracy));
  const ring = $('accuracy-ring');
  ring.setAttribute('stroke-dasharray', circumference.toFixed(1));
  ring.setAttribute('stroke-dashoffset', (circumference * (1 - pct / 100)).toFixed(1));
  $('ring-value').textContent = `${pct}%`;
  $('ring-note').innerHTML = t.reps
    ? `<strong>${t.good}</strong> clean of <strong>${t.reps}</strong> reps logged.`
    : 'No reps logged yet.';
}

function renderStreakStrip(sessions) {
  const days = new Set(sessions.map(s => dayKey(s.startedAt)));
  const today = startOfDay(new Date());
  let html = '';

  for (let i = 13; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const active = days.has(dayKey(d));
    html += `
      <div class="streak-day${active ? ' is-active' : ''}${i === 0 ? ' is-today' : ''}">
        <div class="streak-dot">
          <i class="bi ${active ? 'bi-check-lg' : 'bi-dash'}"></i>
        </div>
        <div class="streak-label">${d.toLocaleDateString([], { weekday: 'narrow' })}</div>
      </div>`;
  }
  $('streak-strip').innerHTML = html;
}

function renderFaults(t) {
  const entries = Object.entries(t.faults).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const list = $('fault-list');
  const empty = $('fault-empty');

  if (!entries.length) {
    list.innerHTML = '';
    empty.classList.remove('hidden');
    return;
  }
  empty.classList.add('hidden');

  const peak = entries[0][1];
  list.innerHTML = entries.map(([key, n], i) => `
    <div class="fault-row${i === 0 ? ' is-top' : ''}">
      <span class="fault-name">${esc(faultLabel(key))}</span>
      <span class="fault-count">${n} rep${n === 1 ? '' : 's'}</span>
      <div class="fault-track">
        <div class="fault-fill" style="width:${Math.round((n / peak) * 100)}%"></div>
      </div>
    </div>`).join('');
}

function renderBreakdown(t) {
  const entries = Object.entries(t.byExercise).sort((a, b) => b[1].reps - a[1].reps);
  const peak = Math.max(1, ...entries.map(([, v]) => v.reps));

  $('exercise-breakdown').innerHTML = entries.map(([key, v]) => {
    const acc = v.reps ? Math.round((v.good / v.reps) * 100) : 0;
    return `
      <div class="breakdown-card">
        <div class="breakdown-head">
          <div class="breakdown-icon"><i class="bi ${exerciseIcon(key)}"></i></div>
          <div class="breakdown-name">${esc(exerciseLabel(key))}</div>
        </div>
        <div class="breakdown-reps">${v.reps}</div>
        <div class="breakdown-meta">${acc}% clean &middot; best ${v.best} &middot; ${v.sessions} session${v.sessions === 1 ? '' : 's'}</div>
        <div class="breakdown-track">
          <div class="breakdown-fill" style="width:${Math.round((v.reps / peak) * 100)}%"></div>
        </div>
      </div>`;
  }).join('');
}
