// static/js/api.js
// Thin wrapper over the session and stats endpoints.
//
// Every call resolves to an object rather than throwing, so a page can
// degrade quietly if the network drops mid-set instead of losing the
// user's work to an unhandled rejection.

async function request(url, options) {
  try {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...options
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return { success: false, error: data.error || `Request failed (${res.status})` };
    }
    return data;
  } catch (err) {
    return { success: false, error: err.message || 'Network error' };
  }
}

export function fetchSessions(exercise) {
  const query = exercise && exercise !== 'all'
    ? `?exercise=${encodeURIComponent(exercise)}`
    : '';
  return request(`/api/sessions${query}`);
}

export function postSession(session) {
  return request('/api/sessions', {
    method: 'POST',
    body: JSON.stringify(session)
  });
}

export function deleteSession(id) {
  return request(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

export function clearSessions() {
  return request('/api/sessions', { method: 'DELETE' });
}

export function fetchStats() {
  return request('/api/stats');
}
