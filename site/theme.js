/* Progressive enhancement: the complete research narrative works without JS. */
(() => {
  const root = document.documentElement;
  const button = document.querySelector('#theme-toggle');
  const modes = ['auto', 'light', 'dark'];
  let mode = 'auto';
  try { mode = localStorage.getItem('geoai-theme') || 'auto'; } catch (_) { /* storage optional */ }
  if (!modes.includes(mode)) mode = 'auto';
  const apply = () => {
    if (mode === 'auto') root.removeAttribute('data-theme');
    else root.dataset.theme = mode;
    button.textContent = `Theme: ${mode}`;
    button.setAttribute('aria-label', `Colour theme: ${mode}. Activate to switch theme.`);
  };
  apply();
  button.hidden = false;
  button.addEventListener('click', () => {
    mode = modes[(modes.indexOf(mode) + 1) % modes.length];
    apply();
    try { localStorage.setItem('geoai-theme', mode); } catch (_) { /* storage optional */ }
  });
})();
