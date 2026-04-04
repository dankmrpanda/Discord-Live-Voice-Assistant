# Wake Word Detection Debug — 2026-02-11

## Symptom
User said "Hey Jarvis" but wake word never triggered. Log showed `capture_users=0, detector_users=0` and `WakeWordSink cleanup - processed 0 total audio chunks`.

## Root Causes Found

### 1. `MIN_SAMPLES` too high (BUG — FIXED)
- **File:** `src/wake_word/detector.py` line ~323
- **Problem:** Uncommitted change raised `MIN_SAMPLES` from 16 to 400. After resampling Discord 48kHz stereo → 16kHz mono, chunks can be as small as 320 samples. The 400 threshold silently dropped valid audio, preventing inference.
- **Fix:** Reverted to `MIN_SAMPLES = 16`. OpenWakeWord buffers internally in 1280-sample windows, so small chunks are fine.
- **Lesson:** Don't raise minimum thresholds without checking actual chunk sizes from py-cord, which vary by version.

### 2. First join had 0 audio chunks (NOT A BUG)
- **Problem:** First session lasted only ~6s in LISTENING state before user typed `/ask`. No one was speaking (members=1, user was typing). Discord only sends audio when someone speaks.
- **Evidence:** Second join in same log worked perfectly — 833 chunks, `capture_users=1`.
- **Lesson:** 0 chunks ≠ broken pipeline. Check if anyone was actually speaking.

### 3. Wake word scores always 0.0000 despite loud audio (INCONCLUSIVE)
- In second session, 833 chunks with 50%+ RMS all scored 0.0000 for `hey_jarvis_v0.1`.
- Old working session: same user scored 0.883 for the same model.
- Most likely user didn't say "Hey Jarvis" during the second session (they may have tested in a session not captured in logs).
- Also: `requirements.txt` changed from `git+...pycord@master` to `py-cord[voice]==2.7.0rc1`, changing chunk sizes from 3840 → 15360 bytes. This interacts with the MIN_SAMPLES issue.

## What Worked (Unchanged & Correct)
- `sink.py` — `WakeWordSink.write()` with `@Filters.container` is correct. Passes all users when `filtered_users=[]`.
- `capture.py` — `process_discord_audio_per_user()` correctly resamples 48kHz stereo → 16kHz mono.
- `processor.py` — Polyphase resampling (down by 3) is correct for 48k→16k.
- `detector.py` — OpenWakeWord `Model(wakeword_models=["hey_jarvis_v0.1"], inference_framework="onnx")` and `model.predict(audio_np)` API usage is correct per openwakeword v0.6.0 docs.
- py-cord `start_recording()` → `recv_audio()` → `unpack_audio()` → `_process_audio_packet()` → `sink.write()` pipeline is functional.
- Per-user model instances prevent audio mixing between users.
- Thread pool executor for CPU-bound inference avoids blocking the event loop.

## What Didn't Work / Red Herrings
- Comparing `git diff 501775b HEAD` showed sink/capture/detector unchanged — but there were **uncommitted local changes** not visible in that diff. Always check `git status` and `git diff HEAD` (working tree).
- `self._voice_client.paused` in py-cord's `unpack_audio()` can silently drop all audio if True — was not the issue here but is a footgun to watch for.
- The old session had 3 users speaking; the new had only 1 user who wasn't always speaking. User count affects whether audio flows.

## Diagnostic Improvements Added
- `sink.py`: Logs immediately on first audio chunk per user (not just every 500th).
- `voice_handler.py`: Heartbeat now includes `sink_chunks` count; warns after 20s of 0 audio.
- `voice_handler.py`: Logs `recording=` and `paused=` state after `start_recording()`.
- `main.py`: Logs py-cord and openwakeword versions at startup.

## Key Files & Architecture
```
Discord 48kHz stereo PCM
  → sink.py: WakeWordSink.write(data, user_id)  [called by py-cord's recv_audio thread]
    → capture.py: process_discord_audio_per_user()  [resamples to 16kHz mono]
      → voice_handler.py: _on_audio_chunk_received()  [checks state == LISTENING]
        → detector.py: process_audio_for_user()  [per-user model, thread pool inference]
          → model.predict(np.int16 array)  → scores dict → threshold check
```
