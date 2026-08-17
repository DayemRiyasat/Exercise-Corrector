// static/js/dashboard.js
// Dashboard page: rep volume chart, accuracy ring, streak strip, fault
// breakdown, per-exercise splits and the session history list.
//
// All numbers come from /api/stats and /api/sessions, so the aggregation
// lives in Python and this file only draws.

import { fetchSessions, fetchStats, clearSessions } from './api.js';

const $ = (id) => document.getElementById(id);
const toast = (msg, icon) => window.shapeform && window.shapeform.toast(msg, icon);

const esc = (str) => String(str).replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const MODE_LABEL = { webcam: 'Live', video: 'Video', image: 'Photo' };

let state = {
  sessions: [],
  stats: null,
  filter: 'all',
  range: 'week'
};

// ---- Formatting ------------------------------------------------------------
function formatDuration(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

const startOfDay = (d) => {
  const x = new Date(d);
  x.setHours(0, 0, 0, 0);
  return x;
};

const dayKey = (d) => {
  const x = startOfDay(d);
  return `${x.getFullYear()}-${x.getMonth() + 1}-${x.getDate()}`;
};

function formatWhen(iso) {
  const then = new Date(iso);
  if (isNaN(then)) return 'Unknown date';
  const days = Math.round((startOfDay(new Date()) - startOfDay(then)) / 86400000);
  const time = then.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  if (days === 0) return `Today, ${time}`;
  if (days === 1) return `Yesterday, ${time}`;
  if (days < 7) return `${then.toLocaleDateString([], { weekday: 'long' })}, ${time}`;
  return then.toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric' });
}

const exerciseName = (slug) =>
  ((state.stats && state.stats.labels && state.stats.labels[slug]) || {}).name || slug;

const exerciseIcon = (slug) =>
  ((state.stats && state.stats.labels && state.stats.labels[slug]) || {}).icon || 'bi-activity';

const faultLabel = (key) =>
  ((state.stats && state.stats.faultLabels) || {})[key] ||
  String(key || '').replace(/_/g, ' ');

function accuracyClass(pct) {
  if (pct >= 80) return '';
  if (pct >= 55) return ' is-mid';
  return ' is-low';
}

// ---- Bucketing -------------------------------------------------------------
/**
 * week  -> last 7 days
 * month -> last 6 weeks
 * year  -> last 12 months
 */
function bucketReps(sessions, range) {
  const now = new Date();
  const buckets = [];

  if (range === 'year') {
    for (let i = 11; i >= 0; i--) {
      const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
      buckets.push({
        label: d.toLocaleDateString([], { month: 'short' }),
        start: d,
        end: new Date(d.getFullYear(), d.getMonth() + 1, 1),
        value: 0, isNow: i === 0
      });
    }
  } else if (range === 'month') {
    const weekStart = startOfDay(now);
    weekStart.setDate(weekStart.getDate() - ((weekStart.getDay() + 6) % 7));
    for (let i = 5; i >= 0; i--) {
      const start = new Date(weekStart);
      start.setDate(start.getDate() - i * 7);
      const end = new Date(start);
      end.setDate(end.getDate() + 7);
      buckets.push({
        label: start.toLocaleDateString([], { day: 'numeric', month: 'short' }),
        start, end, value: 0, isNow: i === 0
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
        start, end, value: 0, isNow: i === 0
      });
    }
  }

  sessions.forEach(s => {
    const when = new Date(s.startedAt);
    if (isNaN(when)) return;
    const bucket = buckets.find(b => when >= b.start && when < b.end);
    if (bucket) bucket.value += s.reps || 0;
  });

  return buckets;
}

// ---- Renderers -------------------------------------------------------------
function renderHeadline() {
  const s = state.stats;
  $('hs-time').innerHTML = `${Math.round(s.seconds / 60)}<small>min</small>`;
  $('hs-reps').textContent = s.reps;
  $('hs-accuracy').innerHTML = `${s.accuracy}<small>%</small>`;
  $('hs-sessions').textContent = s.sessions;
  $('hs-streak').innerHTML = `${s.streak}<small>d</small>`;

  if (!s.sessions) {
    $('progress-headline').textContent = 'Nothing logged yet';
    $('progress-subline').textContent = 'Finish a set in the analyser and your numbers show up here.';
  } else if (s.streak > 1) {
    $('progress-headline').textContent = `${s.streak} days in a row`;
    $('progress-subline').textContent = 'Keep the streak going, consistency is what moves form.';
  } else {
    $('progress-headline').textContent = 'Your training';
    $('progress-subline').textContent = 'Every set you have logged, and what it adds up to.';
  }
}

function renderVolumeChart() {
  const buckets = bucketReps(state.sessions, state.range);
  const peak = Math.max(1, ...buckets.map(b => b.value));

  $('chart-sub').textContent = {
    week: 'Reps completed per day',
    month: 'Reps completed per week',
    year: 'Reps completed per month'
  }[state.range];

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

function renderAccuracyRing() {
  const circumference = 2 * Math.PI * 52;
  const pct = Math.max(0, Math.min(100, state.stats.accuracy));
  const ring = $('accuracy-ring');
  ring.setAttribute('stroke-dasharray', circumference.toFixed(1));
  ring.setAttribute('stroke-dashoffset', (circumference * (1 - pct / 100)).toFixed(1));
  $('ring-value').textContent = `${pct}%`;
  $('ring-note').innerHTML = state.stats.reps
    ? `<strong>${state.stats.good}</strong> clean of <strong>${state.stats.reps}</strong> reps logged.`
    : 'No reps logged yet.';
}

function renderStreakStrip() {
  const days = new Set(state.sessions.map(s => dayKey(s.startedAt)));
  const today = startOfDay(new Date());
  let html = '';

  for (let i = 13; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const active = days.has(dayKey(d));
    html += `
      <div class="streak-day${active ? ' is-active' : ''}${i === 0 ? ' is-today' : ''}">
        <div class="streak-dot"><i class="bi ${active ? 'bi-check-lg' : 'bi-dash'}"></i></div>
        <div class="streak-label">${d.toLocaleDateString([], { weekday: 'narrow' })}</div>
      </div>`;
  }
  $('streak-strip').innerHTML = html;
}

function renderFaults() {
  const entries = Object.entries(state.stats.faults || {})
    .sort((a, b) => b[1] - a[1]).slice(0, 6);
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

function renderBreakdown() {
  const entries = Object.entries(state.stats.byExercise || {})
    .sort((a, b) => b[1].reps - a[1].reps);
  const peak = Math.max(1, ...entries.map(([, v]) => v.reps));

  $('exercise-breakdown').innerHTML = entries.map(([key, v]) => {
    const acc = v.reps ? Math.round((v.good / v.reps) * 100) : 0;
    return `
      <a class="breakdown-card" href="/exercises/${encodeURIComponent(key)}">
        <div class="breakdown-head">
          <div class="breakdown-icon"><i class="bi ${exerciseIcon(key)}"></i></div>
          <div class="breakdown-name">${esc(exerciseName(key))}</div>
        </div>
        <div class="breakdown-reps">${v.reps}</div>
        <div class="breakdown-meta">${acc}% clean &middot; best ${v.best} &middot; ${v.sessions} session${v.sessions === 1 ? '' : 's'}</div>
        <div class="breakdown-track">
          <div class="breakdown-fill" style="width:${Math.round((v.reps / peak) * 100)}%"></div>
        </div>
      </a>`;
  }).join('');
}

function renderHistory() {
  const list = $('session-list');
  const empty = $('history-empty');
  const rows = state.filter === 'all'
    ? state.sessions
    : state.sessions.filter(s => s.exercise === state.filter);

  const total = state.sessions.length;
  $('history-sub').textContent = total
    ? `${total} session${total === 1 ? '' : 's'} logged, newest first.`
    : 'Every set you have logged, newest first.';

  if (!rows.length) {
    list.innerHTML = '';
    empty.classList.remove('hidden');
    empty.querySelector('h3').textContent =
      total ? 'Nothing matches that filter' : 'No sessions yet';
    empty.querySelector('p').textContent = total
      ? 'Try another exercise, or pick All to see everything you have logged.'
      : 'Run a set from the analyser. When you stop the camera, the session lands here with your reps, accuracy and the faults you hit.';
    return;
  }

  empty.classList.add('hidden');
  list.innerHTML = rows.map(s => {
    const acc = s.accuracy ?? (s.reps ? Math.round((s.good / s.reps) * 100) : 0);
    const topFault = Object.entries(s.faults || {}).sort((a, b) => b[1] - a[1])[0];
    return `
      <article class="session-card">
        <div class="session-icon"><i class="bi ${exerciseIcon(s.exercise)}"></i></div>
        <div class="session-main">
          <div class="session-title">
            ${esc(exerciseName(s.exercise))}
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

function renderAll() {
  const panels = document.querySelectorAll('[data-dash-panel]');
  const empty = $('progress-empty');

  renderHeadline();

  if (!state.stats.sessions) {
    panels.forEach(p => p.classList.add('hidden'));
    empty.classList.remove('hidden');
    return;
  }

  panels.forEach(p => p.classList.remove('hidden'));
  empty.classList.add('hidden');

  renderVolumeChart();
  renderAccuracyRing();
  renderStreakStrip();
  renderFaults();
  renderBreakdown();
  renderHistory();
}

// ---- Data ------------------------------------------------------------------
async function load() {
  const [statsRes, sessionsRes] = await Promise.all([fetchStats(), fetchSessions()]);

  $('dash-loading').classList.add('hidden');

  if (!statsRes.success || !sessionsRes.success) {
    toast('Could not load your sessions', 'bi-exclamation-triangle-fill');
    state.stats = state.stats || {
      sessions: 0, reps: 0, good: 0, bad: 0, seconds: 0,
      accuracy: 0, streak: 0, faults: {}, byExercise: {}, labels: {}, faultLabels: {}
    };
    state.sessions = [];
    renderAll();
    return;
  }

  state.stats = statsRes.stats;
  state.sessions = sessionsRes.sessions;
  renderAll();
}

// ---- Controls --------------------------------------------------------------
$('history-filters').addEventListener('click', (e) => {
  const chip = e.target.closest('.filter-chip');
  if (!chip) return;
  state.filter = chip.dataset.filter;
  document.querySelectorAll('.filter-chip')
    .forEach(c => c.classList.toggle('active', c === chip));
  renderHistory();
});

$('range-toggle').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-range]');
  if (!btn) return;
  state.range = btn.dataset.range;
  document.querySelectorAll('#range-toggle button')
    .forEach(b => b.classList.toggle('active', b === btn));
  renderVolumeChart();
});

$('clear-history').addEventListener('click', async () => {
  const count = state.sessions.length;
  if (!count) { toast('No sessions to clear', 'bi-info-circle-fill'); return; }
  if (!confirm(`Delete all ${count} saved session${count === 1 ? '' : 's'}? This cannot be undone.`)) return;

  const res = await clearSessions();
  if (!res.success) {
    toast(res.error || 'Could not clear history', 'bi-exclamation-triangle-fill');
    return;
  }
  await load();
  toast('History cleared', 'bi-trash3-fill');
});

// ---- Boot ------------------------------------------------------------------
load();
