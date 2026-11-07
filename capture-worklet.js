// Collects raw PCM from the audio graph and posts it to the page in ~0.5 s blocks.
// Input channel 0 = microphone, channel 1 = tab/call audio (silent in mic-only mode).
// Also posts a 100 ms loudness envelope per channel, used to label Me vs Them.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(Math.round(sampleRate * 0.5));
    this.n = 0;
    this.envFrames = Math.round(sampleRate * 0.1);
    this.ea = 0; this.eb = 0; this.ec = 0;
    this.env = [];
    this.running = true;
    this.port.onmessage = (e) => { if (e.data === 'stop') { this.flush(); this.running = false; } };
  }
  flush() {
    if (this.n === 0 && this.env.length === 0) return;
    this.port.postMessage({ pcm: this.block.slice(0, this.n), env: Float32Array.from(this.env) });
    this.n = 0;
    this.env = [];
  }
  process(inputs) {
    if (!this.running) return false;
    const inp = inputs[0];
    if (!inp || inp.length === 0) return true;
    const a = inp[0];
    const b = inp.length > 1 ? inp[1] : null;
    for (let i = 0; i < a.length; i++) {
      const x = a[i];
      const y = b ? b[i] : 0;
      let m = x + y;
      if (m > 1) m = 1; else if (m < -1) m = -1;
      this.block[this.n++] = m;
      this.ea += x * x; this.eb += y * y; this.ec++;
      if (this.ec >= this.envFrames) {
        this.env.push(Math.sqrt(this.ea / this.ec), Math.sqrt(this.eb / this.ec));
        this.ea = 0; this.eb = 0; this.ec = 0;
      }
      if (this.n === this.block.length) this.flush();
    }
    return true;
  }
}
registerProcessor('capture', Capture);
