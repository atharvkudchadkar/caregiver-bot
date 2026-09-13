"""Background voice command listener for the caregiver robot.

Wraps the microphone capture + faster-whisper transcription verified in
voice_test.py into a background thread that listens continuously, cuts
utterances out with simple energy-based voice activity detection, transcribes
each one offline, and maps the recognized text onto one of seven commands:

  "go to <room>"        navigate to a learned room (same as run_world.py --goto)
  "explore"             autonomously explore visible free space (same as --explore)
  "come home" / "go home"   return to the startup pose (same as --goal 0 0)
  "stop"                 immediately stop and drop to manual control
  "reset"                 reset to the spawn pose (same as pressing 'r' in the viewer)
  "grab/attach the wheelchair"    line up on the wheelchair and clamp its handles
  "let go/detach the wheelchair"  release the clamp (same as pressing 'x')

This module never touches simulator/world state itself; run_world.py polls
VoiceListener.poll() each control tick and translates a VoiceCommand into the
same calls the keyboard/CLI paths already use.

Standalone test (prints what it hears, no simulator involved):
  python voice_control.py --device 1 --rooms kitchen,bedroom,bathroom,living_room
"""
import argparse
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


@dataclass
class VoiceCommand:
    kind: str                    # "goto" | "explore" | "home" | "stop" | "reset" | "attach" | "detach"
    room: Optional[str] = None
    text: str = ""                # what was actually heard, for logging


def parse_command(text, rooms):
    """Map free-form recognized text onto one of the seven supported commands.

    `rooms` is the set of room names the world/config actually knows about, so
    "go to the kitchen" only resolves if "kitchen" is a real room.
    """
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return None
    if re.search(r"\bstop\b", t):
        return VoiceCommand("stop", text=text)
    if re.search(r"\breset\b", t):
        return VoiceCommand("reset", text=text)
    if re.search(r"\bexplore\b", t):
        return VoiceCommand("explore", text=text)
    if re.search(r"\b(come|go|head) (back )?home\b", t) or "return home" in t:
        return VoiceCommand("home", text=text)
    if re.search(r"\b(attach|grab|hook up|pick up|connect|get)\b.*\bwheelchair\b", t):
        return VoiceCommand("attach", text=text)
    if re.search(r"\b(detach|release|let go|drop|unhook|disconnect)\b.*\bwheelchair\b", t):
        return VoiceCommand("detach", text=text)
    for room in rooms:
        spoken = room.replace("_", " ")
        if re.search(rf"\bgo to (the )?{re.escape(spoken)}\b", t) or \
           re.search(rf"\b{re.escape(spoken)}\b", t):
            return VoiceCommand("goto", room=room, text=text)
    return None


class VoiceListener:
    """Continuous mic listener; call start(), poll() each tick, stop() at exit.

    Voice activity detection (deciding when you start/stop talking) is the
    fragile part of "always listening" speech recognition - much more so than
    the push-to-talk flow in voice_test.py, where you told it exactly when to
    start and stop recording. To make it robust this:
      - calibrates the energy threshold from a second of actual room noise at
        startup instead of using one fixed number for every mic/room,
      - requires a few consecutive loud frames before it starts recording, so
        a single click/pop/breath doesn't cut a false utterance,
      - keeps a short pre-roll buffer so the first syllable isn't clipped when
        speech is detected a beat late,
      - passes the captured clip through faster-whisper's built-in Silero VAD
        filter (vad_filter=True) to strip any residual leading/trailing
        silence before transcription, which is a common source of Whisper
        hallucinating extra words on padding silence.
    """

    def __init__(self, rooms, device=None, model_size="small", language="en",
                 energy_threshold=None, calibration_multiplier=3.0,
                 onset_frames=2, silence_hang_ms=700, max_utterance_s=8,
                 preroll_ms=300, debug=False):
        self.rooms = list(rooms)
        self.device = device
        self.model_size = model_size
        self.language = language
        self.energy_threshold = energy_threshold  # None = auto-calibrate at start()
        self.calibration_multiplier = calibration_multiplier
        self.onset_frames = onset_frames
        self.silence_hang_ms = silence_hang_ms
        self.max_utterance_s = max_utterance_s
        self.preroll_ms = preroll_ms
        self.debug = debug
        self.model = None
        self._audio_q = queue.Queue()
        self._commands = queue.Queue()
        self._stop_event = threading.Event()
        self._stream = None
        self._worker = None

    def start(self):
        from faster_whisper import WhisperModel
        print(f"[voice] loading faster-whisper model {self.model_size!r} (CPU, int8)...")
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                       device=self.device, blocksize=FRAME_SAMPLES,
                                       callback=self._on_audio)
        self._stream.start()
        if self.energy_threshold is None:
            self.energy_threshold = self._calibrate()
        print(f"[voice] noise gate set to {self.energy_threshold:.4f} "
              f"(pass --voice-threshold to override)")
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        print("[voice] listening. Say: \"go to <room>\", \"explore\", \"come home\", "
              "\"stop\", \"reset\", \"grab the wheelchair\", or \"let go of the wheelchair\".")

    def stop(self):
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if self._worker is not None:
            self._worker.join(timeout=2)

    def poll(self):
        """Return the next recognized VoiceCommand, or None if nothing new."""
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    # -- internals ------------------------------------------------------
    def _calibrate(self, seconds=1.0):
        """Sample ambient noise for `seconds` and set the gate above it."""
        print(f"[voice] calibrating noise floor ({seconds:.0f}s, stay quiet)...")
        levels = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                frame = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            levels.append(float(np.sqrt(np.mean(np.square(frame)))))
        floor = float(np.median(levels)) if levels else 0.003
        return max(0.006, floor * self.calibration_multiplier)

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            print(f"[voice] stream warning: {status}")
        self._audio_q.put(indata[:, 0].copy())

    def _run(self):
        preroll_frames = max(1, int(self.preroll_ms / FRAME_MS))
        preroll = deque(maxlen=preroll_frames)
        buffer = []
        recording = False
        onset_run = 0
        silence_frames = 0
        silence_needed = max(1, int(self.silence_hang_ms / FRAME_MS))
        max_frames = int(self.max_utterance_s * 1000 / FRAME_MS)
        last_debug = 0.0
        while not self._stop_event.is_set():
            try:
                frame = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            level = float(np.sqrt(np.mean(np.square(frame))))
            if self.debug and time.monotonic() - last_debug > 0.3:
                last_debug = time.monotonic()
                bar = "#" * min(40, int(level / max(self.energy_threshold, 1e-6) * 10))
                print(f"\r[voice] level={level:.4f} gate={self.energy_threshold:.4f} {bar:<40}",
                      end="", flush=True)
            loud = level > self.energy_threshold
            if not recording:
                preroll.append(frame)
                onset_run = onset_run + 1 if loud else 0
                if onset_run >= self.onset_frames:
                    recording = True
                    buffer = list(preroll)
                    silence_frames = 0
                continue
            buffer.append(frame)
            silence_frames = 0 if loud else silence_frames + 1
            if silence_frames >= silence_needed or len(buffer) >= max_frames:
                audio = np.concatenate(buffer)
                buffer, recording, onset_run, silence_frames = [], False, 0, 0
                preroll.clear()
                self._transcribe_and_queue(audio)

    def _transcribe_and_queue(self, audio):
        if len(audio) / SAMPLE_RATE < 0.3:
            return
        segments, _info = self.model.transcribe(
            audio, language=self.language, beam_size=5, vad_filter=True,
            condition_on_previous_text=False)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if self.debug:
            print()  # move off the live-level line
        if not text:
            if self.debug:
                print("[voice] (silence after VAD filtering, ignored)")
            return
        cmd = parse_command(text, self.rooms)
        print(f"[voice] heard {text!r} -> {cmd.kind if cmd else 'unrecognized, ignored'}")
        if cmd:
            self._commands.put(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="input device index or name substring (default: system default mic)")
    ap.add_argument("--model", default="small", help="faster-whisper model size (default: small)")
    ap.add_argument("--rooms", default="kitchen,bedroom,bathroom,living_room",
                     help="comma-separated known room names")
    ap.add_argument("--threshold", type=float, help="manual noise gate (default: auto-calibrated)")
    ap.add_argument("--debug", action="store_true", help="print a live mic level meter")
    args = ap.parse_args()
    device = args.device
    try:
        device = int(device)
    except (TypeError, ValueError):
        pass
    listener = VoiceListener(args.rooms.split(","), device=device, model_size=args.model,
                             energy_threshold=args.threshold, debug=args.debug)
    listener.start()
    print("Listening for commands. Ctrl+C to stop.")
    try:
        while True:
            cmd = listener.poll()
            if cmd:
                print(f"  -> command: {cmd}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
