import { getLogger } from "../utils/logger.js";

const logger = getLogger("audio.processor");

export class AudioProcessor {
  public constructor(
    public readonly discordSampleRate = 48_000,
    public readonly geminiInputSampleRate = 16_000,
    public readonly geminiOutputSampleRate = 24_000,
  ) {
    logger.debug(
      {
        discordSampleRate,
        geminiInputSampleRate,
        geminiOutputSampleRate,
      },
      "AudioProcessor initialized",
    );
  }

  public discordToGemini(pcm48kStereo: Buffer): Buffer {
    if (pcm48kStereo.length === 0) {
      return Buffer.alloc(0);
    }

    const stereoSamples = new Int16Array(
      pcm48kStereo.buffer,
      pcm48kStereo.byteOffset,
      Math.floor(pcm48kStereo.length / 2),
    );

    const mono48kLength = Math.floor(stereoSamples.length / 2);
    const mono48k = new Float32Array(mono48kLength);

    for (let i = 0; i < mono48kLength; i += 1) {
      const left = stereoSamples[i * 2] ?? 0;
      const right = stereoSamples[i * 2 + 1] ?? 0;
      mono48k[i] = (left + right) / 2;
    }

    const mono16kLength = Math.floor(mono48k.length / 3);
    const mono16k = new Int16Array(mono16kLength);

    for (let i = 0; i < mono16kLength; i += 1) {
      const base = i * 3;
      const s0 = mono48k[base] ?? 0;
      const s1 = mono48k[base + 1] ?? s0;
      const s2 = mono48k[base + 2] ?? s1;
      const averaged = (s0 + s1 + s2) / 3;
      mono16k[i] = clampI16(averaged);
    }

    return Buffer.from(mono16k.buffer, mono16k.byteOffset, mono16k.byteLength);
  }

  public geminiToDiscord(pcm24kMono: Buffer): Buffer {
    if (pcm24kMono.length === 0) {
      return Buffer.alloc(0);
    }

    const mono24k = new Int16Array(
      pcm24kMono.buffer,
      pcm24kMono.byteOffset,
      Math.floor(pcm24kMono.length / 2),
    );

    if (mono24k.length === 0) {
      return Buffer.alloc(0);
    }

    const mono48k = new Int16Array(mono24k.length * 2);

    for (let i = 0; i < mono24k.length; i += 1) {
      const current = mono24k[i] ?? 0;
      const next = mono24k[i + 1] ?? current;
      mono48k[i * 2] = current;
      mono48k[i * 2 + 1] = clampI16((current + next) / 2);
    }

    const stereo48k = new Int16Array(mono48k.length * 2);
    for (let i = 0; i < mono48k.length; i += 1) {
      const sample = mono48k[i] ?? 0;
      const outIndex = i * 2;
      stereo48k[outIndex] = sample;
      stereo48k[outIndex + 1] = sample;
    }

    return Buffer.from(stereo48k.buffer, stereo48k.byteOffset, stereo48k.byteLength);
  }
}

function clampI16(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  if (value > 32_767) {
    return 32_767;
  }
  if (value < -32_768) {
    return -32_768;
  }
  return Math.trunc(value);
}
