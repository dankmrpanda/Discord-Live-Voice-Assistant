# Voice Connection Debugging Findings

Date: 2026-03-27

This document records the investigation into the bot failing to connect to Discord voice
channels after deploying to a VPS running Docker.

---

## Symptom Timeline

### Phase 1 — Initial Failure (py-cord 2.7.1)

The bot would repeatedly join and leave the voice channel over ~28 seconds, then fail:

```
Voice state update: jarvis - before=None, after=d
Voice state update: jarvis - before=d, after=None   (2s later)
Voice state update: jarvis - before=None, after=d   (reconnect attempt)
... (repeats 4-5 times)
Voice connection verification timed out after 10.0s
Failed to join voice channel: Voice connection verification failed
```

Root cause was masked by py-cord's internal `reconnect=True` loop burning 28+ seconds
before surfacing the real error.

### Phase 2 — Real Error Exposed (after retry refactor)

After switching to `reconnect=False` and adding explicit retry logic, the real error
became visible immediately:

```
Voice connection attempt 1 failed: Shard ID None WebSocket closed with 4017
Voice connection attempt 2 failed: Shard ID None WebSocket closed with 4017
Voice connection attempt 3 failed: Shard ID None WebSocket closed with 4017
```

**Close code 4017** = Discord's voice gateway rejecting the client because it does not
support the DAVE (end-to-end encryption) protocol.

---

## Root Cause: Discord DAVE Protocol Enforcement

On **March 2, 2026**, Discord globally enforced the DAVE (Dave's Audio/Video E2EE)
protocol for all non-stage voice calls. This replaced the old encryption modes:

| Old (removed)                    | New (required)                        |
|----------------------------------|---------------------------------------|
| `xsalsa20_poly1305`              | `aead_xchacha20_poly1305_rtpsize`     |
| `xsalsa20_poly1305_suffix`       | `aead_aes256_gcm_rtpsize`             |
| `xsalsa20_poly1305_lite`         |                                       |

**py-cord 2.7.1** only supported the old modes → close code 4017 on every connection
attempt.

References:
- Issue: https://github.com/Pycord-Development/pycord/issues/3135
- Fix PR: https://github.com/Pycord-Development/pycord/pull/3143
- Discord DAVE protocol spec: https://github.com/discord/dave-protocol
- Discord enforcement announcement: https://support.discord.com/hc/en-us/articles/38749827197591

---

## Fix: Upgrade to py-cord 2.8.0rc1

PR [#3143](https://github.com/Pycord-Development/pycord/pull/3143) ("Rewrite Voice
Internals & DAVE Support (send)") was merged on March 14, 2026. It completely rewrote
py-cord's voice subsystem:

- New `discord/voice/` package replacing the monolithic `discord/voice_client.py`
- Voice gateway upgraded to v8 (`wss://{endpoint}/?v=8`)
- DAVE protocol negotiation via the `davey` package (Rust/OpenMLS-based)
- New `VoiceConnectionState` state machine managing the connection lifecycle
- `aead_xchacha20_poly1305_rtpsize` as the supported encryption mode

**py-cord 2.8.0rc1** (released March 21, 2026) is the first release containing this PR.
It adds DAVE protocol support via the `davey` package (a Rust-based Python wheel
implementing the OpenMLS stack).

### Dependency changes (`requirements.txt`)

```diff
- py-cord[voice]==2.7.1
- PyNaCl>=1.5.0
+ py-cord[voice]==2.8.0rc1
+ PyNaCl>=1.6.0,<1.7
```

`davey>=0.1.4` is pulled automatically by the `[voice]` extra.

### py-cord 2.8 Sink API breaking changes

The voice receive internals were rewritten. Custom `Sink` subclasses need three new
additions or `start_recording()` will crash with `AttributeError`:

```python
class MyCustomSink(Sink):
    # Required by SinkEventRouter in discord/voice/receive/router.py
    __sink_listeners__: list = []

    def walk_children(self):
        return iter([])

    def is_opus(self) -> bool:
        return False  # False = want decoded PCM; True = want raw Opus
```

### `start_recording()` API changes

The third positional `channel` argument was deprecated in 2.7 and removed in 2.8:

```diff
- self._voice_client.start_recording(sink, callback, channel)
+ self._voice_client.start_recording(sink, callback)
```

The `callback` signature also changed — it now receives a single optional `exception`
parameter instead of `(sink, channel, *args)`:

```diff
- async def _on_recording_finished(self, sink, channel, *args):
-     sink.cleanup()
+ def _on_recording_finished(self, exception: Exception = None):
+     if exception:
+         logger.warning(f"Recording stopped with error: {exception}")
+     self._sink.cleanup()
```

---

## Known Limitation: Voice Receive Not Yet Working

As of py-cord 2.8.0rc1, **voice receive (recording/sinks) is not yet functional** with
DAVE-encrypted channels. PR #3143 only implemented the send side. The bot will connect
and can play audio (Gemini responses), but `WakeWordSink.write()` will never be called,
meaning:

- Wake word detection does not work
- The bot cannot hear users speak
- `/ask` (text prompts) is the only way to interact

py-cord emits this warning at startup to confirm the limitation:
```
RuntimeWarning: Voice reception is currently broken due to Discord's DAVE (End-to-End
Encryption) protocol. Follow development progress at
https://github.com/Pycord-Development/pycord/issues/3139
```

**Current feature status:**

| Feature | Status |
|---|---|
| Voice channel connection | ✅ Working |
| Audio playback (Gemini responses) | ✅ Working |
| `/ask` text prompts | ✅ Working |
| Wake word detection | ❌ Broken — DAVE receive pending |
| Voice capture / `WakeWordSink.write()` | ❌ Broken — DAVE receive pending |

**Tracking:** https://github.com/Pycord-Development/pycord/issues/3139

**What to do when DAVE receive ships:** Rebuild the Docker image after a new py-cord
release that closes issue #3139. No code changes should be needed — `WakeWordSink` is
already updated for the new 2.8 API and the `write()` method already handles `VoiceData`
objects correctly:

- `data` is a `VoiceData` object — use `data.pcm` for raw PCM bytes
- `user` is a `Member`/`User` object — use `user.id` for the integer user ID

---

## Other Code Changes Made

### Connection retry logic (`voice_handler.py`)

Replaced single `channel.connect(timeout=60, reconnect=True)` with explicit retry loop:

- 3 attempts with `reconnect=True, timeout=30`
- `_cleanup_partial_voice_client()` helper force-disconnects any partial connection
  between attempts, ensuring each attempt starts from a clean state
- `_wait_for_voice_ready()` now uses py-cord 2.8's `wait_until_connected()` instead of
  polling `is_connected()` with a manual timer

**Note:** `wait_until_connected()` is a **synchronous** method (not async) despite
internally waiting on a threading event. It returns `bool` directly. Our wrapper handles
both sync and async returns for forward compatibility:

```python
result = self._voice_client.wait_until_connected(timeout=timeout)
if asyncio.iscoroutine(result) or asyncio.isfuture(result):
    ready = await result
else:
    ready = result
```

### Network investigation

Docker uses `network_mode: "host"` so there is no NAT. The VPS nftables rules have
`policy accept` on the output chain and `ct state established,related accept` on input,
so return UDP traffic from Discord media servers flows freely. Firewall was ruled out as
a cause.
