# Voice Receive Corruption Debug (2026-04-03)

## Symptoms observed

- Repeated:
  - `discord.ext.voice_recv.gateway WS payload has extra keys: {'seq': ...}`
  - `discord.ext.voice_recv.reader Received unexpected rtcp packet: type=200`
  - `discord.ext.voice_recv.reader Received packet for unknown ssrc ...`
- Fatal listener crash:
  - `discord.opus.OpusError: corrupted stream`
  - `Voice receive listener stopped with error: corrupted stream`
- Bot then receives zero chunks and warns that no audio was received.

## Root cause

This matches ongoing `discord-ext-voice-recv` compatibility issues after Discord DAVE/E2EE voice changes (early 2026).  
With `discord.py==2.7.1` + `discord-ext-voice-recv==0.5.2a179`, receive-side decode can attempt Opus decoding before/without correct DAVE handling, producing garbled audio or `corrupted stream`.

## Local patch implemented

### 1) DAVE-aware manual Opus decode in sink

- File: `src/audio/sink.py`
- `WakeWordSink.wants_opus()` changed to `True`.
- Sink now:
  1. Reads `VoiceData.opus`
  2. Decrypts via active Discord DAVE session (`dave_session.decrypt(..., davey.MediaType.audio, ...)`) when available
  3. Decodes with per-user `discord.opus.Decoder`
  4. Drops bad/corrupt frames instead of crashing listener threads

### 2) Auto-restart listener after unexpected receive failure

- File: `src/bot/voice_handler.py`
- On `_on_listening_stopped(error)`:
  - records last error
  - schedules async listener restart if still connected and not intentionally leaving
- Adds guarded restart task/lock and cancellation during normal `leave_channel()`.

### 3) Tests added

- File: `tests/test_sink_adapter.py`
- Added coverage for:
  - Opus path routes decoded PCM to capture
  - Corrupt/decode-fail Opus frame is dropped safely

## Validation

- `python -m pytest tests/test_sink_adapter.py -q` -> `4 passed`
- `python -m pytest tests/test_voice_pipeline.py -q` -> `5 passed`
- `python -m pytest -q` -> `40 passed`

## Upstream references

- Issue #49 (camera/unknown SSRC/corrupted stream):  
  https://github.com/imayhaveborkedit/discord-ext-voice-recv/issues/49
- Issue #50 (RTCP log spam):  
  https://github.com/imayhaveborkedit/discord-ext-voice-recv/issues/50
- Issue #53 (garbled/corrupted receive on newer discord.py):  
  https://github.com/imayhaveborkedit/discord-ext-voice-recv/issues/53
- PR #54 (DAVE receive-side handling proposal, still open):  
  https://github.com/imayhaveborkedit/discord-ext-voice-recv/pull/54
- Discord DAVE protocol docs:  
  https://discord.com/developers/docs/topics/voice-connections#end-to-end-encryption-dave-protocol

