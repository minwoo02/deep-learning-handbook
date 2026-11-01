/* Desktop navigation control. Mobile keeps the theme's native menu button. */
(() => {
  const storageKey = 'handbook.navigationCollapsed';
  let collapsed = false;
  try {
    collapsed = localStorage.getItem(storageKey) === 'true';
  } catch (_) {
    // The control still works if browser storage is unavailable.
  }

  function numberNavigationItems(sidebar) {
    // Each subcategory owns its sequence; titles in Markdown and nav stay unchanged.
    sidebar.querySelectorAll('.md-nav[data-md-level]').forEach(nav => {
      if (Number(nav.dataset.mdLevel) < 2) return;
      const list = nav.querySelector(':scope > .md-nav__list');
      if (!list) return;
      let number = 1;
      list.querySelectorAll(':scope > .md-nav__item').forEach(item => {
        const titles = item.querySelectorAll(':scope > .md-nav__link > .md-ellipsis');
        if (!titles.length) return;
        titles.forEach(title => title.querySelector('.handbook-nav-number')?.remove());
        const value = titles[0].textContent.trim() === '개요' ? 0 : number++;
        titles.forEach(title => {
          const prefix = document.createElement('span');
          prefix.className = 'handbook-nav-number';
          prefix.textContent = `${value}. `;
          title.prepend(prefix);
        });
      });
    });
  }

  function initializeNavigationControl() {
    const header = document.querySelector('.md-header__inner');
    const sidebar = document.querySelector('.md-sidebar--primary');
    if (!header || !sidebar) return;
    sidebar.id = 'handbook-primary-navigation';
    numberNavigationItems(sidebar);
    const hasPageNavigation = !!sidebar.querySelector('.md-nav__item--active.md-nav__item--nested');
    document.body.classList.toggle('handbook-navigation-unavailable', !hasPageNavigation);

    let button = document.getElementById('handbook-navigation-toggle');
    if (!button) {
      button = document.createElement('button');
      button.id = 'handbook-navigation-toggle';
      button.type = 'button';
      button.className = 'handbook-navigation-toggle';
      button.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/><path data-handbook-chevron/></svg>';
      header.prepend(button);

      button.addEventListener('click', () => {
        collapsed = !collapsed;
        try {
          localStorage.setItem(storageKey, String(collapsed));
        } catch (_) {
          // Persistence is optional; toggling is not.
        }
        updateControl();
      });
    }

    function updateControl() {
      const label = collapsed ? '목차 펼치기' : '목차 접기';
      document.body.classList.toggle('handbook-navigation-collapsed', collapsed);
      button.setAttribute('aria-controls', sidebar.id);
      button.setAttribute('aria-expanded', String(!collapsed));
      button.setAttribute('aria-label', label);
      button.title = label;
      button.hidden = !hasPageNavigation;
      button.querySelector('[data-handbook-chevron]').setAttribute('d', collapsed ? 'm13 9 3 3-3 3' : 'm16 9-3 3 3 3');
      // Let the theme recalculate sticky sidebars after the content width changes.
      window.dispatchEvent(new Event('resize'));
    }

    updateControl();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeNavigationControl, { once: true });
  } else {
    initializeNavigationControl();
  }
  // Keep the control compatible with the theme's optional instant navigation.
  if (typeof document$ !== 'undefined') document$.subscribe(initializeNavigationControl);
})();
