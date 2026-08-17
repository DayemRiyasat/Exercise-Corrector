// static/js/site.js
// Site-wide behaviour that every page shares: navigation, the toast, the
// FAQ accordion and scroll reveals. Loaded as a plain script, so it can
// expose a small global that the page modules reuse.

(function () {
  'use strict';

  // ---- Toast --------------------------------------------------------------
  let toastTimer = null;

  function toast(message, icon) {
    const el = document.getElementById('toast');
    const text = document.getElementById('toast-text');
    if (!el || !text) return;

    text.textContent = message;
    const glyph = el.querySelector('i');
    if (glyph) glyph.className = 'bi ' + (icon || 'bi-check-circle-fill');

    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { el.classList.remove('show'); }, 3600);
  }

  window.shapeform = { toast: toast };

  // ---- Navigation ---------------------------------------------------------
  const nav = document.getElementById('site-nav');
  const toggle = document.getElementById('nav-toggle');
  const menu = document.getElementById('nav-menu');

  if (toggle && menu) {
    toggle.addEventListener('click', function () {
      const open = menu.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(open));
      toggle.innerHTML = open
        ? '<i class="bi bi-x-lg"></i>'
        : '<i class="bi bi-list"></i>';
    });

    // Close the menu after following a link on mobile.
    menu.addEventListener('click', function (e) {
      if (e.target.closest('a') && menu.classList.contains('open')) {
        menu.classList.remove('open');
        toggle.setAttribute('aria-expanded', 'false');
        toggle.innerHTML = '<i class="bi bi-list"></i>';
      }
    });
  }

  if (nav) {
    const onScroll = function () {
      nav.classList.toggle('scrolled', window.scrollY > 10);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  // ---- FAQ: one open at a time -------------------------------------------
  const faqItems = Array.prototype.slice.call(document.querySelectorAll('.faq-item'));
  faqItems.forEach(function (item) {
    item.addEventListener('toggle', function () {
      if (!item.open) return;
      faqItems.forEach(function (other) {
        if (other !== item) other.open = false;
      });
    });
  });

  // ---- Scroll reveal ------------------------------------------------------
  const reveals = document.querySelectorAll('[data-reveal]');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (!reveals.length) return;

  if (reduceMotion || !('IntersectionObserver' in window)) {
    reveals.forEach(function (el) { el.classList.add('revealed'); });
    return;
  }

  const observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('revealed');
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

  reveals.forEach(function (el) { observer.observe(el); });
})();
