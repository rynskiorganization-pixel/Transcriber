// Offline support: cache the app shell and the transformers.js library.
// Whisper model files are cached separately by transformers.js itself.
const CACHE = 'mt-v2';
const SHELL = ['./', 'index.html', 'app.js', 'worker.js', 'capture-worklet.js', 'manifest.webmanifest',
  'icon.svg', 'icon-192.png', 'icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  const sameOrigin = url.origin === self.location.origin;
  const lib = url.hostname === 'cdn.jsdelivr.net';
  if (!sameOrigin && !lib) return; // models, API calls: straight to network
  // Network first for the app (so updates arrive), cache fallback when offline.
  e.respondWith(
    fetch(req).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
      return res;
    }).catch(() => caches.match(req).then((r) => r || caches.match('index.html')))
  );
});
