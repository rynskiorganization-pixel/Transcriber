// Meeting Transcriber — installable web app.
// Records, transcribes locally with Whisper (worker.js), labels speakers,
// summarises via the Claude API (opt-in), exports, and keeps history in IndexedDB.

const $ = (s) => document.querySelector(s);
const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
  (navigator.maxTouchPoints > 1 && /Mac/.test(navigator.platform));
const SEG_SEC = isMobile ? 30 : 20;          // live chunk length
const DIAR_MAX_SEC = isMobile ? 8 * 60 : 15 * 60; // browser speaker-model limit

// ---------- settings ----------
const DEFAULTS = {
  model: 'onnx-community/whisper-small',
  lang: '',
  live: true,
  autoSpk: true,
  apiKey: '',
  claudeModel: 'claude-sonnet-5-5',
};
const settings = Object.assign({}, DEFAULTS, safeJSON(lsGet('mt.settings')));
function lsGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch {} }
function safeJSON(s) { try { return JSON.parse(s) || {}; } catch { return {}; } }
function saveSettings() { lsSet('mt.settings', JSON.stringify(settings)); }

// ---------- worker plumbing ----------
const worker = new Worker('worker.js', { type: 'module' });
const jobs = new Map();
let jobSeq = 0;
let chain = Promise.resolve();
let queued = 0;
let speakersResolve = null;
const downloads = {};

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'segments') { jobs.get(m.id)?.resolve(m.segments); jobs.delete(m.id); }
  else if (m.type === 'error') {
    if (m.id != null && jobs.has(m.id)) { jobs.get(m.id).reject(new Error(m.message)); jobs.delete(m.id); }
    else if (m.where === 'diarize' && speakersResolve) { speakersResolve.reject(new Error(m.message)); speakersResolve = null; }
    else setStatus('⚠ ' + m.message);
  }
  else if (m.type === 'speakers') { speakersResolve?.resolve(m.turns); speakersResolve = null; }
  else if (m.type === 'status') setStatus(m.text);
  else if (m.type === 'ready') {
    $('#dlBar').classList.add('hidden');
    $('#deviceInfo').textContent = m.device === 'webgpu' ? 'Running on the GPU (WebGPU).' : 'Running on the CPU (WebAssembly).';
    setStatus(recording ? 'Recording…' : 'Model ready');
  }
  else if (m.type === 'download') {
    downloads[m.file] = m;
    let l = 0, t = 0;
    for (const d of Object.values(downloads)) { l += d.loaded; t += d.total; }
    $('#dlBar').classList.remove('hidden');
    $('#dlBar i').style.width = (100 * l / t).toFixed(1) + '%';
    setStatus(`Downloading model (first time only): ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB`);
  }
};

function transcribeJob(audio, offset) {
  const id = ++jobSeq;
  return new Promise((resolve, reject) => {
    jobs.set(id, { resolve, reject });
    worker.postMessage({ type: 'transcribe', id, audio, offset, language: settings.lang,
      model: settings.model, prefer: 'auto' }, [audio.buffer]);
  });
}

function enqueue(sess, audio16, offset) {
  if (rms(audio16) < 0.002) return chain; // silence: skip, avoids hallucinations
  queued++; updateQueue();
  chain = chain
    .then(() => transcribeJob(audio16, offset))
    .then((segs) => { sess.segments.push(...segs); sess.segments.sort((a, b) => a.start - b.start); if (sess === current) renderTranscript(); })
    .catch((err) => setStatus('⚠ ' + err.message))
    .finally(() => { queued--; updateQueue(); });
  return chain;
}

function updateQueue() {
  $('#queueInfo').textContent = queued ? `transcribing… ${queued} chunk${queued > 1 ? 's' : ''} queued` : '';
}

// ---------- audio helpers ----------
function rms(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / Math.max(1, a.length)); }

function concat(list, len) {
  const out = new Float32Array(len ?? list.reduce((n, a) => n + a.length, 0));
  let o = 0; for (const a of list) { out.set(a, o); o += a.length; }
  return out;
}

// Find the quietest 100 ms in the last `win` seconds — cut there so words aren't split.
function findCut(buf, sr, win = 4) {
  const frame = Math.round(sr * 0.1);
  const from = Math.max(0, buf.length - Math.round(win * sr));
  let best = buf.length, bestE = Infinity;
  for (let s = from; s + frame <= buf.length; s += frame) {
    let e = 0; for (let i = s; i < s + frame; i++) e += buf[i] * buf[i];
    if (e < bestE) { bestE = e; best = s + (frame >> 1); }
  }
  return best;
}

async function to16k(x, sr) {
  if (sr === 16000) return x.slice();
  const len = Math.ceil(x.length * 16000 / sr);
  const oc = new OfflineAudioContext(1, len, 16000);
  const b = oc.createBuffer(1, x.length, sr);
  b.copyToChannel(x, 0);
  const src = oc.createBufferSource(); src.buffer = b; src.connect(oc.destination); src.start();
  const r = await oc.startRendering();
  return r.getChannelData(0).slice();
}

async function decodeTo16k(blob) {
  const ab = await blob.arrayBuffer();
  const oc = new OfflineAudioContext(1, 1, 16000);
  const buf = await oc.decodeAudioData(ab); // resampled to 16 kHz by the browser
  if (buf.numberOfChannels === 1) return buf.getChannelData(0).slice();
  const out = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c); for (let i = 0; i < d.length; i++) out[i] += d[i] / buf.numberOfChannels;
  }
  return out;
}

// ---------- session model ----------
let current = null;
function newSession(title, source) {
  return { id: 'S' + Date.now(), title, created: Date.now(), duration: 0, source,
    segments: [], speakerNames: {}, summary: '', audioBlob: null, mime: '', env: [], pcm16: [] };
}

// ---------- recording ----------
let recording = false, ctx, micStream, tabStream, node, recorder, recChunks, pending, pendingLen,
  processedNative, sr, cutIdx, deferred, timerInt, wakeLock, startT;

async function startRecording() {
  const source = document.querySelector('input[name=source]:checked').value;
  const sess = newSession('Meeting ' + new Date().toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }), source);
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    if (source === 'tab') {
      setStatus('Pick the meeting tab and tick “Also share tab audio”.');
      tabStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      if (!tabStream.getAudioTracks().length) {
        tabStream.getTracks().forEach((t) => t.stop()); tabStream = null;
        throw new Error('No tab audio shared. Try again and tick “Also share tab audio”.');
      }
      tabStream.getVideoTracks()[0]?.addEventListener('ended', () => recording && stopRecording());
    }
  } catch (err) {
    micStream?.getTracks().forEach((t) => t.stop());
    setStatus('⚠ ' + (err.name === 'NotAllowedError' ? 'Permission denied.' : err.message));
    return;
  }

  current = sess;
  renderAll();
  worker.postMessage({ type: 'load', model: settings.model, prefer: 'auto' });

  ctx = new AudioContext();
  await ctx.resume();
  sr = ctx.sampleRate;
  await ctx.audioWorklet.addModule('capture-worklet.js');
  const merger = ctx.createChannelMerger(2);
  const micSrc = ctx.createMediaStreamSource(micStream);
  micSrc.connect(merger, 0, 0);
  let recStream = micStream;
  if (tabStream) {
    const tabSrc = ctx.createMediaStreamSource(new MediaStream(tabStream.getAudioTracks()));
    tabSrc.connect(merger, 0, 1);
    const dest = ctx.createMediaStreamDestination();
    micSrc.connect(dest); tabSrc.connect(dest);
    recStream = dest.stream;
  }
  node = new AudioWorkletNode(ctx, 'capture', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
    channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'discrete' });
  merger.connect(node);
  const mute = ctx.createGain(); mute.gain.value = 0;
  node.connect(mute).connect(ctx.destination);

  pending = []; pendingLen = 0; processedNative = 0; cutIdx = 0; deferred = [];
  node.port.onmessage = (e) => onPCM(sess, e.data);

  const mime = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/webm', 'audio/ogg;codecs=opus']
    .find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || '';
  recChunks = [];
  recorder = new MediaRecorder(recStream, mime ? { mimeType: mime } : undefined);
  recorder.ondataavailable = (e) => e.data.size && recChunks.push(e.data);
  recorder.start(1000);
  sess.mime = recorder.mimeType || mime;

  recording = true;
  startT = Date.now();
  $('#recBtn').classList.add('on');
  $('#recBtn').setAttribute('aria-label', 'Stop recording');
  $('#importBtn').disabled = true;
  setButtons(false);
  setStatus(settings.live ? 'Recording… (text appears every ~' + SEG_SEC + ' s)' : 'Recording… (transcription runs after you stop)');
  timerInt = setInterval(() => { $('#timer').textContent = fmt((Date.now() - startT) / 1000); }, 250);
  acquireWakeLock();
}

function onPCM(sess, { pcm, env }) {
  for (let i = 0; i < env.length; i++) sess.env.push(env[i]);
  if (!pcm.length) return;
  pending.push(pcm); pendingLen += pcm.length;
  $('#level').style.width = Math.min(100, rms(pcm) * 600) + '%';
  if (pendingLen >= SEG_SEC * sr) cutChunk(sess, false);
}

async function cutChunk(sess, final) {
  const buf = concat(pending, pendingLen);
  const cut = final ? buf.length : findCut(buf, sr);
  const head = buf.subarray(0, cut);
  const rest = buf.slice(cut);
  pending = [rest]; pendingLen = rest.length;
  const offset = processedNative / sr;
  processedNative += cut;
  const idx = cutIdx++;
  const a16 = await to16k(head, sr);
  sess.pcm16[idx] = a16;
  if (settings.live) await enqueue(sess, a16.slice(), offset).catch(() => {});
  else deferred.push({ idx, offset });
}

async function stopRecording() {
  if (!recording) return;
  recording = false;
  clearInterval(timerInt);
  releaseWakeLock();
  $('#recBtn').classList.remove('on');
  $('#recBtn').setAttribute('aria-label', 'Start recording');
  $('#recBtn').disabled = true;
  $('#level').style.width = '0';
  const sess = current;

  // drain the worklet, then stop everything
  node.port.postMessage('stop');
  await new Promise((r) => setTimeout(r, 150));
  const stopped = new Promise((r) => { recorder.onstop = r; });
  recorder.stop();
  await stopped;
  micStream.getTracks().forEach((t) => t.stop());
  tabStream?.getTracks().forEach((t) => t.stop());
  tabStream = null;
  await ctx.close();

  sess.audioBlob = new Blob(recChunks, { type: sess.mime || 'audio/webm' });
  sess.duration = processedNative / sr + pendingLen / sr;
  if (pendingLen > sr * 0.3) await cutChunk(sess, true);
  else processedNative += pendingLen;

  if (!settings.live) for (const d of deferred) enqueue(sess, sess.pcm16[d.idx].slice(), d.offset);
  setStatus('Finishing transcription…');
  await chain;
  await finishSession(sess);
  $('#recBtn').disabled = false;
  $('#importBtn').disabled = false;
}

async function finishSession(sess) {
  if (sess.source === 'tab') labelByChannel(sess);
  else if (settings.autoSpk && sess.segments.length) await identifySpeakers(sess, true);
  sess.pcm16 = sess.pcm16.filter(Boolean);
  await dbPut(sess);
  setStatus(sess.segments.length ? 'Done — saved on this device.' : 'Done — no speech detected.');
  setButtons(true);
  renderAll();
  renderHistory();
}

// ---------- speakers ----------
function labelByChannel(sess) {
  const env = sess.env; // interleaved [mic, tab] per 100 ms
  const frames = env.length / 2;
  for (const s of sess.segments) {
    let m = 0, t = 0;
    for (let f = Math.floor(s.start * 10); f < Math.min(frames, Math.ceil(s.end * 10)); f++) { m += env[2 * f]; t += env[2 * f + 1]; }
    s.speaker = m >= t ? 'Me' : 'Them';
  }
}

async function identifySpeakers(sess, auto = false) {
  let audio = sess.pcm16 && sess.pcm16.filter(Boolean).length ? concat(sess.pcm16.filter(Boolean)) : null;
  if (!audio) {
    if (!sess.audioBlob) return;
    setStatus('Decoding audio…');
    audio = await decodeTo16k(sess.audioBlob);
  }
  const dur = audio.length / 16000;
  if (dur > DIAR_MAX_SEC) {
    setStatus(`Speaker labels in the browser work up to ${DIAR_MAX_SEC / 60} min. Use the desktop app for longer meetings.`);
    return;
  }
  try {
    const turns = await new Promise((resolve, reject) => {
      speakersResolve = { resolve, reject };
      worker.postMessage({ type: 'diarize', audio }, [audio.buffer]);
    });
    const order = [];
    for (const t of turns) if (!order.includes(t.speaker)) order.push(t.speaker);
    const name = (lbl) => 'Speaker ' + (order.indexOf(lbl) + 1);
    for (const s of sess.segments) {
      const score = {};
      for (const t of turns) {
        const ov = Math.min(s.end, t.end) - Math.max(s.start, t.start);
        if (ov > 0) score[t.speaker] = (score[t.speaker] || 0) + ov;
      }
      const best = Object.entries(score).sort((a, b) => b[1] - a[1])[0];
      if (best) s.speaker = name(best[0]);
    }
    if (!auto) { await dbPut(sess); setStatus('Speakers labelled. Tap a name to rename it.'); }
    if (sess === current) renderTranscript();
  } catch (err) {
    setStatus('⚠ Speaker labels failed: ' + err.message);
  }
}

// ---------- import a file ----------
async function importFile(file) {
  const sess = newSession(file.name.replace(/\.[^.]+$/, ''), 'file');
  sess.audioBlob = file; sess.mime = file.type;
  current = sess;
  renderAll();
  setButtons(false);
  $('#recBtn').disabled = true; $('#importBtn').disabled = true;
  try {
    setStatus('Decoding audio…');
    const pcm = await decodeTo16k(file);
    sess.duration = pcm.length / 16000;
    sess.pcm16 = [pcm];
    worker.postMessage({ type: 'load', model: settings.model, prefer: 'auto' });
    const piece = 120 * 16000;
    let pos = 0;
    while (pos < pcm.length) {
      let end = Math.min(pcm.length, pos + piece);
      if (end < pcm.length) end = pos + findCut(pcm.subarray(pos, end), 16000, 6);
      enqueue(sess, pcm.slice(pos, end), pos / 16000);
      pos = end;
    }
    setStatus('Transcribing…');
    await chain;
    await finishSession(sess);
  } catch (err) {
    setStatus('⚠ Could not read that file: ' + err.message);
  }
  $('#recBtn').disabled = false; $('#importBtn').disabled = false;
}

// ---------- summary ----------
const SUMMARY_SYSTEM = `You summarise meeting transcripts produced by automatic speech recognition, so expect some recognition errors and infer the intended words where obvious.
Write in the main language of the meeting (Polish or English).
Output Markdown with these sections:
## Summary — 3 to 6 sentences.
## Decisions — bullet list.
## Action items — bullets formatted "**Owner** — task — due date (if mentioned)". Use "Unassigned" if the owner is unclear.
## Open questions — bullet list.
Use the speaker names as given. Do not invent facts; write "None" for an empty section.`;

async function summarize() {
  const sess = current;
  if (!settings.apiKey) { setStatus('Add your Claude API key in Settings first.'); openSettings(); return; }
  $('#summaryBtn').disabled = true;
  setStatus('Writing summary with Claude…');
  try {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': settings.apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({
        model: settings.claudeModel,
        max_tokens: 2500,
        system: SUMMARY_SYSTEM,
        messages: [{ role: 'user', content: 'Meeting: ' + sess.title + '\n\nTranscript:\n\n' + toText(sess, false) }],
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error?.message || res.statusText);
    sess.summary = data.content.filter((c) => c.type === 'text').map((c) => c.text).join('\n');
    await dbPut(sess);
    renderSummary();
    setStatus('Summary ready.');
  } catch (err) {
    setStatus('⚠ Summary failed: ' + err.message);
  }
  $('#summaryBtn').disabled = false;
}

// ---------- exports ----------
function fmt(sec, withHours = false) {
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const p = (n) => String(n).padStart(2, '0');
  return (h || withHours) ? `${p(h)}:${p(m)}:${p(s)}` : `${p(m)}:${p(s)}`;
}
function spkName(sess, s) { return s.speaker ? (sess.speakerNames[s.speaker] || s.speaker) : ''; }

// Merge consecutive lines from the same speaker into paragraphs.
function paragraphs(sess) {
  const out = [];
  for (const s of sess.segments) {
    const name = spkName(sess, s);
    const last = out[out.length - 1];
    if (last && last.name === name && s.start - last.end < 3 && last.text.length < 900) {
      last.text += ' ' + s.text; last.end = s.end;
    } else out.push({ name, start: s.start, end: s.end, text: s.text });
  }
  return out;
}

function toText(sess, header = true) {
  const lines = paragraphs(sess).map((p) => `[${fmt(p.start)}] ${p.name ? p.name + ': ' : ''}${p.text}`);
  const head = header ? `${sess.title}\n${new Date(sess.created).toLocaleString()} · ${fmt(sess.duration)}\n\n` : '';
  return head + lines.join('\n');
}

function toMarkdown(sess) {
  let md = `# ${sess.title}\n\n_${new Date(sess.created).toLocaleString()} · ${fmt(sess.duration)}_\n\n`;
  if (sess.summary) md += sess.summary.trim() + '\n\n';
  md += '## Transcript\n\n';
  for (const p of paragraphs(sess)) md += `**[${fmt(p.start)}]${p.name ? ' ' + p.name + ':' : ''}** ${p.text}\n\n`;
  return md;
}

function toSRT(sess) {
  const t = (x) => fmt(x, true) + ',' + String(Math.floor((x % 1) * 1000)).padStart(3, '0');
  return sess.segments.map((s, i) => `${i + 1}\n${t(s.start)} --> ${t(s.end)}\n${spkName(sess, s) ? spkName(sess, s) + ': ' : ''}${s.text}\n`).join('\n');
}

async function exportAs(kind) {
  const sess = current;
  const base = sess.title.replace(/[\\/:*?"<>|]+/g, '-').trim() || 'meeting';
  let blob, name;
  if (kind === 'md') { blob = new Blob([toMarkdown(sess)], { type: 'text/markdown' }); name = base + '.md'; }
  else if (kind === 'txt') { blob = new Blob([toText(sess)], { type: 'text/plain' }); name = base + '.txt'; }
  else if (kind === 'srt') { blob = new Blob([toSRT(sess)], { type: 'application/x-subrip' }); name = base + '.srt'; }
  else if (kind === 'json') {
    const { audioBlob, pcm16, env, ...rest } = sess;
    blob = new Blob([JSON.stringify(rest, null, 2)], { type: 'application/json' }); name = base + '.json';
  } else if (kind === 'audio') {
    if (!sess.audioBlob) return setStatus('No audio saved for this session.');
    blob = sess.audioBlob;
    const ext = /mp4|m4a|aac/.test(sess.mime) ? 'm4a' : /ogg/.test(sess.mime) ? 'ogg' : /mpeg|mp3/.test(sess.mime) ? 'mp3' : /wav/.test(sess.mime) ? 'wav' : 'webm';
    name = base + '.' + ext;
  } else if (kind === 'copy') {
    try { await navigator.clipboard.writeText(toMarkdown(sess)); setStatus('Copied.'); } catch { setStatus('Copy failed.'); }
    return;
  }
  const file = new File([blob], name, { type: blob.type || 'application/octet-stream' });
  if (isMobile && navigator.canShare?.({ files: [file] })) {
    try { await navigator.share({ files: [file], title: name }); return; } catch (e) { if (e.name === 'AbortError') return; }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(file); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

// ---------- rendering ----------
function setStatus(t) { $('#status').textContent = t; }
function setButtons(on) {
  const has = on && current && current.segments.length;
  $('#speakersBtn').disabled = !has || current.source === 'tab';
  $('#summaryBtn').disabled = !has;
  $('#exportBtn').disabled = !(on && current);
}
function esc(s) { return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

function speakerColor(sess, label) {
  if (label === 'Me') return 'var(--sp1)';
  if (label === 'Them') return 'var(--sp2)';
  const order = [...new Set(sess.segments.map((s) => s.speaker).filter(Boolean))];
  return `var(--sp${(order.indexOf(label) % 5) + 1})`;
}

function renderTranscript() {
  const el = $('#transcript');
  const sess = current;
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
  if (!sess || !sess.segments.length) {
    el.innerHTML = `<div class="empty">${recording ? 'Listening… first text arrives after ~' + SEG_SEC + ' s.' : 'Press record. Text appears here as you speak.'}</div>`;
    return;
  }
  el.innerHTML = sess.segments.map((s) => {
    const n = spkName(sess, s);
    const spk = n ? `<span class="spk" data-l="${esc(s.speaker)}" style="color:${speakerColor(sess, s.speaker)}">${esc(n)}</span>` : '';
    return `<div class="line"><div class="ts">${fmt(s.start)}</div><div>${spk}${esc(s.text)}</div></div>`;
  }).join('');
  if (nearBottom || recording) el.scrollTop = el.scrollHeight;
}

function renderSummary() {
  const card = $('#summaryCard');
  if (!current || !current.summary) { card.classList.add('hidden'); return; }
  card.classList.remove('hidden');
  const lines = current.summary.split('\n');
  let html = '', inList = false;
  const inline = (t) => esc(t).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  for (const raw of lines) {
    const l = raw.trim();
    const li = /^[-*•]\s+(.*)/.exec(l);
    if (li) { if (!inList) { html += '<ul>'; inList = true; } html += `<li>${inline(li[1])}</li>`; continue; }
    if (inList) { html += '</ul>'; inList = false; }
    if (!l) continue;
    const h = /^#{1,4}\s+(.*)/.exec(l);
    html += h ? `<h3>${inline(h[1])}</h3>` : `<p>${inline(l)}</p>`;
  }
  if (inList) html += '</ul>';
  $('#summary').innerHTML = html;
}

function renderAll() {
  $('#sessTitle').textContent = current ? current.title : 'Transcript';
  renderTranscript();
  renderSummary();
}

async function renderHistory() {
  const list = await dbAll();
  const el = $('#history');
  if (!list.length) { el.innerHTML = '<div class="empty">No recordings yet.</div>'; return; }
  list.sort((a, b) => b.created - a.created);
  el.innerHTML = list.map((s) => `<div class="hist"><div><b>${esc(s.title)}</b><br><span class="small">${new Date(s.created).toLocaleDateString()} · ${fmt(s.duration)} · ${s.segments.length} lines${s.summary ? ' · summary' : ''}</span></div>
    <div class="row" style="flex-wrap:nowrap"><button class="btn" data-open="${s.id}">Open</button><button class="btn" data-del="${s.id}" aria-label="Delete">✕</button></div></div>`).join('');
}

// ---------- IndexedDB ----------
let dbP;
function db() {
  dbP ??= new Promise((res, rej) => {
    const r = indexedDB.open('meeting-transcriber', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('sessions', { keyPath: 'id' });
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
  return dbP;
}
async function tx(mode, fn) {
  const d = await db();
  return new Promise((res, rej) => {
    const t = d.transaction('sessions', mode); const st = t.objectStore('sessions');
    const out = fn(st);
    t.oncomplete = () => res(out?.result); t.onerror = () => rej(t.error);
  });
}
async function dbPut(sess) {
  const { pcm16, ...rest } = sess; // raw PCM is too big to keep; audioBlob is kept
  try { await tx('readwrite', (st) => st.put(rest)); } catch (e) { setStatus('⚠ Could not save: ' + e.message); }
}
async function dbAll() { try { return (await tx('readonly', (st) => st.getAll())) || []; } catch { return []; } }
async function dbGet(id) { return tx('readonly', (st) => st.get(id)); }
async function dbDel(id) { return tx('readwrite', (st) => st.delete(id)); }

// ---------- wake lock ----------
async function acquireWakeLock() { try { wakeLock = await navigator.wakeLock?.request('screen'); } catch {} }
function releaseWakeLock() { try { wakeLock?.release(); } catch {} wakeLock = null; }
document.addEventListener('visibilitychange', () => {
  if (recording && document.visibilityState === 'visible') acquireWakeLock();
});
window.addEventListener('beforeunload', (e) => { if (recording || queued) { e.preventDefault(); e.returnValue = ''; } });

// ---------- settings dialog ----------
function openSettings() {
  $('#model').value = settings.model; $('#lang').value = settings.lang;
  $('#live').checked = settings.live; $('#autoSpk').checked = settings.autoSpk;
  $('#apiKey').value = settings.apiKey; $('#claudeModel').value = settings.claudeModel;
  $('#settings').showModal();
}
$('#settings').addEventListener('close', () => {
  settings.model = $('#model').value; settings.lang = $('#lang').value;
  settings.live = $('#live').checked; settings.autoSpk = $('#autoSpk').checked;
  settings.apiKey = $('#apiKey').value.trim(); settings.claudeModel = $('#claudeModel').value.trim() || DEFAULTS.claudeModel;
  saveSettings();
});

// ---------- wiring ----------
$('#settingsBtn').onclick = openSettings;
$('#recBtn').onclick = () => (recording ? stopRecording() : startRecording());
$('#importBtn').onclick = () => $('#fileIn').click();
$('#fileIn').onchange = (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) importFile(f); };
$('#speakersBtn').onclick = async () => { $('#speakersBtn').disabled = true; await identifySpeakers(current); $('#speakersBtn').disabled = false; };
$('#summaryBtn').onclick = summarize;
$('#exportBtn').onclick = (e) => { e.stopPropagation(); $('#exportMenu').classList.toggle('hidden'); };
document.addEventListener('click', () => $('#exportMenu').classList.add('hidden'));
$('#exportMenu').onclick = (e) => { const x = e.target.dataset.x; if (x) exportAs(x); };
$('#sessTitle').onclick = async () => {
  if (!current || recording) return;
  const t = prompt('Rename this meeting', current.title);
  if (t && t.trim()) { current.title = t.trim(); await dbPut(current); renderAll(); renderHistory(); }
};
$('#transcript').onclick = async (e) => {
  const l = e.target.closest('.spk')?.dataset.l;
  if (!l || recording) return;
  const n = prompt(`Name for ${current.speakerNames[l] || l}`, current.speakerNames[l] || l);
  if (n && n.trim()) { current.speakerNames[l] = n.trim(); await dbPut(current); renderTranscript(); }
};
$('#history').onclick = async (e) => {
  const open = e.target.dataset.open, del = e.target.dataset.del;
  if (recording) return;
  if (open) { current = await dbGet(open); current.pcm16 = []; renderAll(); setButtons(true); window.scrollTo({ top: 0, behavior: 'smooth' }); }
  if (del && confirm('Delete this recording and transcript?')) {
    await dbDel(del); if (current?.id === del) { current = null; renderAll(); setButtons(false); } renderHistory();
  }
};

if (!navigator.mediaDevices?.getDisplayMedia || isMobile) $('#tabOpt').classList.add('hidden');
if (!navigator.mediaDevices?.getUserMedia) { setStatus('This browser cannot record audio.'); $('#recBtn').disabled = true; }
if (navigator.storage?.persist) navigator.storage.persist().catch(() => {});
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
renderHistory();
