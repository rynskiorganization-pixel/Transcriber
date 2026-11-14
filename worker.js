// Runs Whisper (speech-to-text) and pyannote (speaker segmentation) locally
// in the browser via transformers.js. Audio never leaves the device.
import {
  pipeline,
  AutoProcessor,
  AutoModelForAudioFrameClassification,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 1; // GitHub Pages can't enable threads; avoids extra memory

let asr = null;
let asrKey = '';
let device = 'wasm';

// Phrases Whisper is known to invent on silence/noise (esp. in Polish).
const HALLUCINATIONS = [
  /amara\.org/i,
  /^napisy (stworzone|wykonane|przygotowane)/i,
  /^(dziękuję( bardzo)?( za uwagę)?|thank you( very much)?|thanks for watching)[.!]?$/i,
  /^(\.|…|-)+$/,
];

async function hasWebGPU() {
  try { return !!(self.navigator.gpu && (await self.navigator.gpu.requestAdapter())); }
  catch { return false; }
}

async function loadASR(model, prefer) {
  const key = model + '|' + prefer;
  if (asr && asrKey === key) return asr;
  const gpu = prefer !== 'wasm' && (await hasWebGPU());
  if (model.includes('large') && !gpu) {
    throw new Error('The large model needs WebGPU (desktop Chrome/Edge). Pick "small" or "base".');
  }
  device = gpu ? 'webgpu' : 'wasm';
  const dtype = gpu
    ? { encoder_model: model.includes('large') ? 'fp16' : 'fp32', decoder_model_merged: 'q4' }
    : 'q8';
  postMessage({ type: 'status', text: `Loading ${model.split('/')[1]} (${device})…` });
  asr = await pipeline('automatic-speech-recognition', model, {
    device,
    dtype,
    progress_callback: (p) => {
      if (p.status === 'progress' && p.total) {
        postMessage({ type: 'download', file: p.file, loaded: p.loaded, total: p.total });
      }
    },
  });
  asrKey = key;
  // Warm up so the first real chunk isn't slow (skipped on CPU to save memory).
  if (gpu) await asr(new Float32Array(16000), { return_timestamps: true });
  postMessage({ type: 'ready', device });
  return asr;
}

function clean(text) {
  const t = (text || '').trim();
  if (!t) return '';
  return HALLUCINATIONS.some((r) => r.test(t)) ? '' : t;
}

async function transcribe({ id, audio, offset, language, model, prefer }) {
  const p = await loadASR(model, prefer);
  const dur = audio.length / 16000;
  const out = await p(audio, {
    language: language || undefined, // undefined = auto-detect per 30 s window
    task: 'transcribe',
    return_timestamps: true,
    chunk_length_s: 30,
    stride_length_s: 5,
  });
  const chunks = out.chunks && out.chunks.length ? out.chunks : [{ timestamp: [0, dur], text: out.text }];
  const segments = [];
  for (const c of chunks) {
    const text = clean(c.text);
    if (!text) continue;
    const s = c.timestamp?.[0] ?? 0;
    const e = c.timestamp?.[1] ?? dur;
    segments.push({ start: offset + s, end: offset + Math.min(e, dur), text });
  }
  postMessage({ type: 'segments', id, segments });
}

let diarModel = null;
let diarProc = null;
async function diarize({ audio }) {
  const id = 'onnx-community/pyannote-segmentation-3.0';
  if (!diarModel) {
    postMessage({ type: 'status', text: 'Loading speaker model…' });
    diarProc = await AutoProcessor.from_pretrained(id);
    diarModel = await AutoModelForAudioFrameClassification.from_pretrained(id, { device: 'wasm', dtype: 'fp32' });
  }
  postMessage({ type: 'status', text: 'Identifying speakers…' });
  const inputs = await diarProc(audio);
  const { logits } = await diarModel(inputs);
  const raw = diarProc.post_process_speaker_diarization(logits, audio.length)[0];
  const labels = diarModel.config.id2label;
  const turns = [];
  for (const r of raw) {
    const label = String(labels[r.id] ?? r.id);
    if (/NO_SPEAKER/i.test(label)) continue;
    // Overlapping speech comes as "SPEAKER_1+SPEAKER_2": attribute to the first.
    turns.push({ speaker: label.split('+')[0], start: r.start, end: r.end });
  }
  postMessage({ type: 'speakers', turns });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') await loadASR(m.model, m.prefer);
    else if (m.type === 'transcribe') await transcribe(m);
    else if (m.type === 'diarize') await diarize(m);
  } catch (err) {
    postMessage({ type: 'error', id: m.id, where: m.type, message: String(err?.message || err) });
  }
};
