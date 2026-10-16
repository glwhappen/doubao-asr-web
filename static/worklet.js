// 麦克风采集线程：把任意采样率转成 16kHz 单声道 PCM16，攒成 20ms（640 字节）一块发给主线程
class PCM16Worklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const target = (options && options.processorOptions && options.processorOptions.targetRate) || 16000;
    this.targetRate = target;
    this.ratio = sampleRate / target;    // 例如 48000/16000 = 3
    this.acc = 0;
    this.count = 0;
    this.frac = 0;
    this.blockSamples = (target * 20) / 1000;   // 320 样点 = 20ms
    this.chunk = new Int16Array(this.blockSamples * 10);
    this.offset = 0;
    this.tick = 0;
    this.peak = 0;
  }

  emit() {
    if (this.offset === 0) return;
    const slice = this.chunk.slice(0, this.offset);
    this.offset = 0;
    this.port.postMessage({ pcm: slice.buffer }, [slice.buffer]);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];

    for (let i = 0; i < ch.length; i++) {
      const v = ch[i];
      this.acc += v;
      this.count++;
      const a = v < 0 ? -v : v;
      if (a > this.peak) this.peak = a;
      this.frac += 1;
      if (this.frac >= this.ratio) {
        let s = this.acc / this.count;
        this.acc = 0;
        this.count = 0;
        this.frac -= this.ratio;
        if (s > 1) s = 1; else if (s < -1) s = -1;
        this.chunk[this.offset++] = s < 0 ? s * 0x8000 : s * 0x7fff;
        if (this.offset === this.chunk.length) this.emit();
      }
    }

    // 电平表：降频上报，避免消息过多
    if (++this.tick >= 8) {
      this.port.postMessage({ peak: this.peak });
      this.tick = 0;
      this.peak = 0;
    }
    return true;
  }
}

registerProcessor('pcm16', PCM16Worklet);
