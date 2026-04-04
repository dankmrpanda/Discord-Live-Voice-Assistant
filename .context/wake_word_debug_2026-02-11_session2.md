# Wake Word Detection Debug — 2026-02-11 (Session 2: Audio Clipping & AGC)

## Symptom
User saying "Hey Jarvis" clearly into microphone, but wake word scores remain near 0.000000 across 1200+ audio chunks. Even `predict_clip()` on saved WAV fails (max 0.004463).

## Root Cause: Discord Bot Receives Raw Audio Without AGC

Discord clients apply **receive-side Automatic Gain Control (AGC)** when playing audio for human users. Bots receive the **raw Opus-decoded PCM** without any gain normalization. The user's microphone input was extremely loud:

- **RMS: 7908** (24% of full scale) — normal speech is typically 5-10%
- **Peak: ±32767** — constantly hitting the int16 ceiling (clipping)
- **Effective frequency: ~1946 Hz** — not natural speech, dominated by clipping artifacts
- **Large jumps (>10000): 7708** in 10s — severe waveform discontinuities

This means the wake word model receives heavily clipped, distorted audio that sounds nothing like natural speech, even though other Discord users in the same call hear perfectly normal audio (their client AGC normalizes it).

## Evidence

### From `session_20260211_220407.log`:
```
AUDIO FORMAT DIAGNOSTIC:
   First 20 int16 values: [0, 0, -1, -1, -3, -3, -11, -11, -26, -26, -51, -51, -91, -91, -150, -150, -230, -230, -337, -337]
   Avg diff between even/odd (L/R if stereo): 0.0  → MONO data in stereo container

CONVERSION DIAGNOSTIC:
   After pcm_to_numpy: min=-1.0000, max=1.0000  → full-scale audio
   After resample (48k→16k): min=-0.9994, max=1.0004
   Final PCM: min=-32747, max=32767  → clipping

Audio stats (throughout session):
   chunk #4:  rms=20209, peak=32767, rms_pct=61.7%
   chunk #7:  rms=18268, peak=32767, rms_pct=55.8%
   chunk #8:  rms=21364, peak=32767, rms_pct=65.2%
   chunk #401: rms=23623, peak=32767, rms_pct=72.1%

predict_clip max scores: hey_jarvis_v0.1=0.004463  → model cannot detect in clipped audio
```

### From WAV analysis (`analyze_wav.py`):
```
Shape: (160000,), Min: -32767, Max: 32767, RMS: 7908.21
Large jumps (>10000): 7708, (>20000): 4083, (>30000): 2419

RMS per 100ms (note massive gaps of silence from batched delivery):
   0-100ms:   13.6%
   400-500ms: 22.1%
   500-600ms: 63.0%  ← extremely loud
   700-2400ms: 0.0%  ← bulk silence packets eating capture window

Top frequency peaks: 1558 Hz dominant → clipping harmonics, not speech formants
```

### Key architectural insight:
- Discord sends the **same Opus stream** to all recipients
- Human users' Discord clients apply AGC, noise suppression, echo cancellation during **playback**
- Bots decode the Opus but have **no client-side processing** → raw, potentially clipped audio
- There is **no Discord API** to get the post-processed audio

## Hypotheses Eliminated
- ❌ Audio pipeline corruption — conversion stages all produce correct values
- ❌ Stereo/mono mismatch — confirmed mono data in stereo container (L/R diff = 0.0)
- ❌ Code regression — git diff shows audio pipeline unchanged between working/current commits
- ❌ openwakeword version drift — only v0.6.0 exists
- ❌ Streaming vs clip prediction bug — `predict_clip()` also fails
- ❌ Microphone sample rate — irrelevant; Discord client resamples to 48kHz Opus before transmission

## Fix Applied: AGC (Automatic Gain Control)

### `src/wake_word/detector.py`
Added `_apply_agc()` method that normalizes volume before wake word detection:
- Per-user exponential smoothing of RMS estimate
- Target RMS: 2000 (~6% full scale, typical quiet speech)
- Gain clamped to [0.05, 10.0] range
- Applied after debug WAV save but before `model.predict()`
- Logs gain factor on first chunk per user

### Diagnostic additions:
1. **Raw 48kHz stereo WAV** (`debug_raw_48k_user_<id>.wav`) — what Discord delivers before conversion
2. **AGC-normalized WAV** (`debug_audio_agc_user_<id>.wav`) — normalized version with `predict_clip()` test
3. **Extended debug capture** from 10s → 30s to ensure wake word utterance is captured
4. Raw Discord audio passed through callback chain: `capture.py` → `voice_handler.py` → `detector.py`

### Modified files:
- `src/wake_word/detector.py` — AGC logic, raw audio dump, AGC WAV dump, extended capture
- `src/audio/capture.py` — passes raw Discord audio (`pcm_data`) through callback
- `src/bot/voice_handler.py` — receives raw audio in callback, forwards to detector

## What the next test run will tell us

The logs will show:
```
🔊 AGC for user <id>: input_rms=XXXX, target_rms=2000, gain=0.XXX
   Input was X.Xx louder than target
📼 Raw Discord audio saved: logs/debug_raw_48k_user_<id>.wav
📼 AGC-normalized WAV saved: logs/debug_audio_agc_user_<id>.wav
   🔬 AGC predict_clip max scores: hey_jarvis_v0.1=X.XXXXXX
   ✅ AGC predict_clip DETECTS wake word! Volume was the issue.
         — OR —
   ❌ AGC predict_clip also fails. Issue is NOT just volume.
```

If AGC predict_clip detects: volume was the sole issue, AGC fix is sufficient.
If AGC predict_clip also fails: there's an additional audio quality problem beyond volume.

## Key Files & Architecture (updated)
```
Discord 48kHz stereo PCM (raw, no AGC)
  → sink.py: WakeWordSink.write(data, user_id)
    → capture.py: process_discord_audio_per_user()  [resamples to 16kHz mono]
      → voice_handler.py: _on_audio_chunk_received(audio, user, raw_discord_data)
        → detector.py: _dump_raw_discord_audio()  [saves 48kHz stereo debug WAV]
        → detector.py: process_audio_for_user()
          → _dump_audio_to_wav()  [saves 16kHz mono debug WAV + AGC WAV + predict_clip tests]
          → _apply_agc()  [normalizes volume per-user]
          → model.predict(agc_normalized_int16)  → scores dict → threshold check
```
