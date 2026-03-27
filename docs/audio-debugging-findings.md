# Audio Debugging Findings

Date: 2026-03-26

This document records the investigation into the long-standing "malformed audio" issue
where audio from Discord voice channel users was not processed correctly, resulting in
garbled input to the Gemini Live API.

---

## Audio Pipeline Overview

```
Discord Voice Channel
  (Opus-encoded RTP packets)
        |
        v
py-cord PacketRouter (Opus decode -> 48kHz stereo 16-bit PCM)
        |
        v
WakeWordSink.write(data, user)          [src/audio/sink.py]
  - Extracts PCM bytes from VoiceData object
  - Extracts integer user ID from Member object
        |
        v
AudioCapture.process_discord_audio_per_user()  [src/audio/capture.py]
  - Per-user buffering with deque
  - VAD (voice activity detection)
  - Streaming ring buffer for Gemini
        |
        v
AudioProcessor.discord_to_gemini()      [src/audio/processor.py]
  - PCM frame alignment validation
  - Explicit stereo-to-mono deinterleave
  - Stateful anti-alias filter (per-user)
  - Decimation 48kHz -> 16kHz
        |
        v
Gemini Live API (16kHz mono 16-bit PCM)
```

### Audio Formats at Each Stage

| Stage | Sample Rate | Channels | Bit Depth | Bytes/20ms frame |
|-------|-------------|----------|-----------|------------------|
| Discord (decoded) | 48,000 Hz | 2 (stereo) | 16-bit | 3,840 |
| After mono conversion | 48,000 Hz | 1 (mono) | 16-bit | 1,920 |
| After resampling (Gemini in) | 16,000 Hz | 1 (mono) | 16-bit | ~640 |
| Gemini output | 24,000 Hz | 1 (mono) | 16-bit | varies |
| Discord playback | 48,000 Hz | 2 (stereo) | 16-bit | 3,840 |

---

## Root Causes Found

### 1. py-cord VoiceData/bytes API Mismatch (CRITICAL)

**Location:** `src/audio/sink.py` - `WakeWordSink.write()`

**Problem:** py-cord's master branch `PacketRouter._do_run()` calls:
```python
self.sink.write(data, data.source)
```
where:
- `data` is a `VoiceData` object (has `.pcm` attribute with actual PCM bytes)
- `data.source` is a `Member` object (has `.id` attribute with integer user ID)

The original code treated `data` as raw `bytes` and `user` as `int`. When `data`
(a VoiceData object) was passed to `np.frombuffer()`, it produced completely wrong
audio data or crashed.

**Fix:** Extract `.pcm` from VoiceData and `.id` from Member with type-checking
fallbacks for both old and new py-cord API versions.

**Diagnostic:** The first-chunk log line now prints `data_type=` and `user_type=`
to confirm which API version is active.

---

### 2. Stateless Chunk-by-Chunk Resampling (HIGH)

**Location:** `src/audio/processor.py` - `resample()` / `resample_stateful()`

**Problem:** `scipy.signal.resample_poly()` was called independently on each ~20ms
chunk with no filter state carried between calls. The internal FIR anti-aliasing
filter's state was reset at every chunk boundary, causing:

- Audible clicks/pops at every chunk boundary (~50 per second)
- Phase discontinuities accumulating over time
- Loss of low-energy signal near chunk edges

For 48kHz->16kHz downsampling with a typical 60-tap FIR filter, approximately 20
samples at each boundary were corrupted. With 50 chunks/second, that's 1,000
corrupted boundaries per second of audio.

**Fix:** Replaced with a two-stage stateful approach:
1. Pre-designed 8th-order Butterworth anti-aliasing filter (cutoff at target Nyquist)
2. `scipy.signal.sosfilt()` with per-user `zi` state preserved between calls
3. Simple decimation (take every Nth sample) after filtering

Per-user state ensures multi-user audio streams don't contaminate each other.
State is cleaned up when users leave the channel via `reset_user_state()`.

**Why sosfilt + decimate instead of resample_poly:**
`resample_poly` is a single-call function that designs and applies a filter
internally. It has no API for carrying state between calls. `sosfilt` accepts
a `zi` parameter that preserves filter memory across calls, making it suitable
for streaming/chunked processing.

---

### 3. Fragile Stereo-to-Mono Detection (MEDIUM)

**Location:** `src/audio/processor.py` - `stereo_to_mono()` / `discord_to_gemini()`

**Problem:** The old `stereo_to_mono()` method used `len(audio) % 2 == 0` to guess
if a 1D array was interleaved stereo. Since virtually all audio arrays have even
length, this worked most of the time. But:

- Non-standard frames from packet loss or PLC could have odd length, bypassing
  mono conversion entirely and doubling the sample count
- The `is_stereo=True` flag was already passed through the pipeline but ignored
  in favor of the length-based guess

**Fix:** `discord_to_gemini()` now uses the explicit `is_stereo` flag and does
direct reshape + mean:
```python
if is_stereo:
    audio = audio.reshape(-1, 2).mean(axis=1).astype(np.float32)
```

Added PCM frame alignment validation before conversion - if `len(pcm_data)` is
not a multiple of `bytes_per_frame` (4 for stereo, 2 for mono), trailing bytes
are trimmed with a warning log.

---

## Additional Improvements

### Ring Buffer Sizing
- Increased streaming ring buffer from 100 to 250 frames (2s -> 5s headroom)
- Added capacity warning at 80% full to detect consumer lag

### Per-User State Cleanup
- `AudioCapture.cleanup_user()` now calls `processor.reset_user_state(user_id)`
  to free resampler filter state when users leave the channel

### Diagnostic Logging
- First audio chunk logs `data_type`, `pcm_size`, `user_type`, `user_id`
- PCM frame alignment warnings when trimming is needed
- Buffer capacity warnings when streaming buffer is nearly full

---

## How to Diagnose Future Audio Issues

### Quick Checks

1. **Check first-chunk log:** Look for `"First audio chunk:"` in logs. Confirm
   `data_type` and `user_type` match expectations.

2. **Check conversion ratio:** A 3840-byte Discord frame should produce ~640
   bytes of Gemini audio (6:1 ratio for stereo 48kHz -> mono 16kHz).

3. **Check for frame alignment warnings:** Search logs for `"PCM data not frame-aligned"`.

4. **Check buffer capacity:** Search logs for `"Streaming buffer near capacity"`.

### Audio Dump for Offline Analysis

Enable `log_audio: true` in `config.yaml`, then dump audio at key pipeline stages:

```python
# In capture.py, after discord_to_gemini():
with open(f"debug_user_{user_id}_gemini.pcm", "ab") as f:
    f.write(gemini_pcm)
```

Play back with:
```bash
ffplay -f s16le -ar 16000 -ac 1 debug_user_123_gemini.pcm
```

### Common Symptoms and Causes

| Symptom | Likely Cause |
|---------|-------------|
| Garbled/random noise | VoiceData object passed as bytes (type mismatch) |
| Regular clicks/pops every ~20ms | Stateless resampling (filter state not preserved) |
| Audio pitched up or sped up | Stereo treated as mono (double samples) |
| Audio pitched down or slowed | Mono treated as stereo (half samples) |
| Missing audio segments | Ring buffer overflow (consumer too slow) |
| Silent/very quiet audio | Wrong normalization or clipping |
| Echo or doubled audio | Per-user isolation broken (same audio in multiple buffers) |

---

## py-cord Version & Docker Issues

### 4. py-cord master Breaking Changes — `discord.sinks` Missing (CRITICAL)

**Symptom:**
```
AttributeError: module 'discord' has no attribute 'sinks'
```
followed by:
```
AttributeError: 'DiscordBot' object has no attribute 'slash_command'. Did you mean: 'add_command'?
```

**Problem:** The bot was installing py-cord from `@master` branch via git:
```
git+https://github.com/Pycord-Development/pycord.git@master#egg=py-cord[voice]
```
py-cord's master branch evolved past the stable API — the `sinks` module and
`slash_command` method were restructured or removed. Tracking `@master` means
any upstream commit can break the bot without warning.

**Fix:** Pinned to the latest stable release `py-cord[voice]==2.7.1` which
includes the voice fix from PR #2812 (the reason master was originally used).

**Location:** `requirements.txt`

---

### 5. discord.py vs py-cord Package Conflict (CRITICAL)

**Symptom:** Same as above — `discord.sinks` missing, `slash_command` missing —
even when py-cord is listed in requirements.

**Problem:** Both `discord.py` and `py-cord` install as the `discord` Python
package. If any transitive dependency pulls in `discord.py`, it silently
overwrites py-cord. The bot then runs with `discord.py` which has no `sinks`
module and uses `add_command` instead of `slash_command`.

**Root cause:** pip has no protection against two packages installing into the
same namespace. Installation order determines which one wins.

**Fix (Dockerfile):**
```dockerfile
# Install py-cord FIRST, then other deps, then remove any discord.py that snuck in
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "py-cord[voice]==2.7.1" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y discord.py 2>/dev/null; true
```

**How to diagnose:** Run inside the container:
```bash
docker exec <container> python -c "import discord; print(discord.__version__); print(hasattr(discord, 'sinks'))"
```
- If `sinks` is `False` → `discord.py` is installed, not py-cord
- If version doesn't match `2.7.1` → wrong package or version

**How to check which package owns the discord namespace:**
```bash
docker exec <container> pip show py-cord
docker exec <container> pip show discord.py
```
Only one of these should be installed. If both appear, uninstall `discord.py`.

---

### 6. Docker Image Not Rebuilt After Code Changes

**Symptom:** Source code changes (in `src/`) take effect immediately due to
volume mount, but dependency changes (in `requirements.txt` or `Dockerfile`)
have no effect.

**Problem:** `docker-compose.yml` mounts `../src:/app/src:ro` which overrides
the image's `/app/src` at runtime. This means Python source changes are live.
However, installed packages (`pip install`) are baked into the Docker image at
build time. Changes to `requirements.txt` or `Dockerfile` require a rebuild.

**Fix:** Always rebuild after changing dependencies:
```bash
docker compose build --no-cache
docker compose up -d
```

**When rebuild is needed:**
- Changed `requirements.txt` (added/removed/changed a dependency)
- Changed `Dockerfile` (modified install steps, base image, etc.)
- Changed system-level dependencies (apt packages)

**When rebuild is NOT needed:**
- Changed any `.py` file under `src/` (volume-mounted live)
- Changed `config.yaml` (volume-mounted live)

---

## py-cord Voice API Notes

- **py-cord 2.7.1** (pinned stable): `sink.write(data, user)` receives raw `bytes` and `int` user ID
- **py-cord master** (unstable): may pass `VoiceData` objects and `Member` objects — our code handles both
- Issue #2833 was about voice CONNECTION stability, not audio format
- PR #2812 fixed voice disconnection issues, included in v2.7.0+
- The `@Filters.container` decorator only filters by user list, doesn't transform data
- Discord always decodes Opus to 48kHz stereo 16-bit PCM (hardcoded in py-cord)
- **Never track `@master`** for production — pin to a tagged release

### Common Symptoms and Causes (Updated)

| Symptom | Likely Cause |
|---------|-------------|
| `module 'discord' has no attribute 'sinks'` | discord.py installed instead of py-cord, or py-cord master broke API |
| `has no attribute 'slash_command'` | Same as above — wrong discord package |
| Garbled/random noise | VoiceData object passed as bytes (type mismatch) |
| Regular clicks/pops every ~20ms | Stateless resampling (filter state not preserved) |
| Audio pitched up or sped up | Stereo treated as mono (double samples) |
| Audio pitched down or slowed | Mono treated as stereo (half samples) |
| Missing audio segments | Ring buffer overflow (consumer too slow) |
| Silent/very quiet audio | Wrong normalization or clipping |
| Echo or doubled audio | Per-user isolation broken (same audio in multiple buffers) |
| Code changes have no effect in Docker | Image not rebuilt after dependency changes |
