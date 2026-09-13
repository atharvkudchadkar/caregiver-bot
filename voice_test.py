"""Standalone microphone speech-recognition verification tool.

NOT part of the robot pipeline. This exists purely to check, on this PC,
whether we can (a) capture audio from a chosen input device and (b)
transcribe speech from it accurately enough, before wiring any of this into
the bot. Nothing here touches build_world.py, run_world.py, or navigation.

Uses faster-whisper (a local CTranslate2 build of OpenAI's Whisper) so
everything runs offline: no audio leaves the machine.

Usage:
  python voice_test.py --list-devices
  python voice_test.py --device 1
  python voice_test.py --device "Realtek" --model small
  python voice_test.py --device 1 --expect "go to the kitchen" --trials 5

Each trial: press Enter to start recording, speak, press Enter to stop.
With --expect, accuracy is scored as 1 - word error rate against the phrase
you actually said (compare that to what you typed for --expect).
"""
import argparse
import sys

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


def list_devices():
    print(f"{'idx':>4}  {'in-ch':>5}  {'rate':>7}  name")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"{i:4d}  {d['max_input_channels']:5d}  {d['default_samplerate']:7.0f}  {d['name']}")


def resolve_device(spec):
    """Accept a device index, a name substring, or None for the system default."""
    if spec is None:
        return None
    try:
        return int(spec)
    except ValueError:
        pass
    matches = [i for i, d in enumerate(sd.query_devices())
               if spec.lower() in d["name"].lower() and d["max_input_channels"] > 0]
    if not matches:
        raise SystemExit(f"no input device matching {spec!r}; run --list-devices to see options")
    if len(matches) > 1:
        names = ", ".join(f"{i}:{sd.query_devices(i)['name']}" for i in matches)
        raise SystemExit(f"ambiguous device {spec!r}, matches: {names}; use the index instead")
    return matches[0]


def device_name(device):
    idx = device if device is not None else sd.default.device[0]
    return sd.query_devices(idx)["name"]


def record_until_enter(device):
    """Record mono float32 audio at 16kHz from `device` between two Enter presses."""
    frames = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"  [stream warning: {status}]", file=sys.stderr)
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                             device=device, callback=callback)
    input("  Press Enter to START recording...")
    stream.start()
    print("  Recording... press Enter to STOP.")
    input()
    stream.stop()
    stream.close()
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames, axis=0)[:, 0]


def word_error_rate(reference, hypothesis):
    """Standard word error rate: Levenshtein edit distance over words / reference length."""
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref)][len(hyp)] / max(len(ref), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true", help="list audio input devices and exit")
    ap.add_argument("--device", help="input device index or name substring (default: system default mic)")
    ap.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large-v3"],
                     help="faster-whisper model size, bigger = more accurate but slower (default: base)")
    ap.add_argument("--language", default=None, help="force a language code, e.g. en (default: auto-detect)")
    ap.add_argument("--expect", help="phrase you intend to say, to score transcription accuracy against")
    ap.add_argument("--trials", type=int, default=1, help="number of recordings to run (default: 1)")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    device = resolve_device(args.device)
    print(f"Using input device: {device_name(device)}")
    print(f"Loading faster-whisper model {args.model!r} (CPU, int8)... first run downloads the model.")
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    scores = []
    for trial in range(1, args.trials + 1):
        print(f"\n--- trial {trial}/{args.trials} ---")
        audio = record_until_enter(device)
        duration = len(audio) / SAMPLE_RATE
        if duration < 0.2:
            print("  (nothing captured, skipping)")
            continue
        segments, info = model.transcribe(audio, language=args.language, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        print(f"  captured {duration:.1f}s audio, detected language={info.language} "
              f"(p={info.language_probability:.2f})")
        print(f"  transcript: {text!r}")
        if args.expect:
            wer = word_error_rate(args.expect, text)
            accuracy = max(0.0, 1 - wer) * 100
            print(f"  expected:   {args.expect!r}")
            print(f"  word error rate: {wer:.2f}  (~{accuracy:.0f}% word accuracy)")
            scores.append(accuracy)

    if scores:
        print(f"\nAverage word accuracy over {len(scores)} trial(s): {sum(scores) / len(scores):.0f}%")


if __name__ == "__main__":
    main()
