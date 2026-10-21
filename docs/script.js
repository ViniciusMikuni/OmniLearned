// Decorative point cloud; not a visualization of measured or simulated events.
const svg = document.querySelector('#particle-art');
if (svg) {
  let seed = 31;
  const random = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
  let shapes = '<defs><radialGradient id="glow"><stop stop-color="#d4ef9d" stop-opacity=".45"/><stop offset="1" stop-color="#c6a0ff" stop-opacity="0"/></radialGradient></defs><ellipse cx="370" cy="320" rx="285" ry="260" fill="url(#glow)"/>';
  const points = Array.from({length: 185}, () => {
    const t = random(), angle = random() * Math.PI * 2, r = Math.sqrt(random()) * (30 + t * 185);
    return {x: 130 + t * 460 + Math.cos(angle) * r * .32, y: 330 + Math.sin(angle) * r, r: .8 + random() * 2.4};
  });
  points.forEach((p, i) => {
    if (i % 2 === 0) { const q = points[(i + 15) % points.length]; if (Math.hypot(p.x - q.x, p.y - q.y) < 100) shapes += `<path d="M${p.x},${p.y}L${q.x},${q.y}" stroke="#c6a0ff" stroke-width=".6" opacity=".25"/>`; }
    shapes += `<circle cx="${p.x}" cy="${p.y}" r="${p.r}" fill="${i % 5 ? '#c6a0ff' : '#d4ef9d'}" opacity="${.3 + random() * .65}"/>`;
  });
  svg.innerHTML = shapes;
}
const dialog = document.querySelector('#figure-dialog');
if (dialog && typeof dialog.showModal === 'function') {
  document.querySelectorAll('.zoom').forEach(link => link.addEventListener('click', event => {
    event.preventDefault();
    const original = link.querySelector('img');
    dialog.querySelector('img').src = link.href;
    dialog.querySelector('img').alt = original.alt;
    dialog.querySelector('p').textContent = original.alt;
    dialog.showModal();
  }));
  dialog.querySelector('button').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {if (event.target === dialog) {const b = dialog.getBoundingClientRect();if(event.clientX < b.left || event.clientX > b.right || event.clientY < b.top || event.clientY > b.bottom) dialog.close();}});
}
if ('IntersectionObserver' in window && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const observer = new IntersectionObserver(entries => entries.forEach(entry => {if(entry.isIntersecting) {entry.target.classList.add('reveal');observer.unobserve(entry.target);}}), {threshold: .08});
  document.querySelectorAll('.study-heading,.section-heading').forEach(el => observer.observe(el));
}

// Play only while visible, respect reduced motion, and preserve user pauses.
const video = document.querySelector('.molecule-video');
if (video && 'IntersectionObserver' in window && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
  let started = false;
  const videoObserver = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting && !started) {
      started = true;
      video.play().catch(() => {});
      videoObserver.disconnect();
    }
  }, {threshold: .3});
  videoObserver.observe(video);
}
