"""
Add echo/reverb to a video's audio using ffmpeg.

Usage examples (PowerShell):
python backend/scripts/add_reverb_to_video.py -i test_clips\input.mp4 -o backend/tmp\out_reverb.mp4

Use an impulse response (IR) WAV for realistic reverb:
python backend/scripts/add_reverb_to_video.py -i test_clips\input.mp4 -o backend/tmp\out_ir.mp4 --ir ir_samples\room_ir.wav

Or use the builtin `aecho` filter (fast, parametric):
python backend/scripts/add_reverb_to_video.py -i test_clips\input.mp4 -o backend/tmp\out_echo.mp4 --delays 1000 1800 --decays 0.3 0.2 --ingain 0.8 --outgain 0.9

Requirements: `ffmpeg` available on PATH. Script creates a temporary folder under system temp.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def run(cmd):
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc


def check_ffmpeg():
    from shutil import which

    if which("ffmpeg") is None:
        print("ffmpeg not found in PATH. Install ffmpeg and ensure it's on PATH.", file=sys.stderr)
        sys.exit(1)


def build_parser():
    p = argparse.ArgumentParser(description="Add reverb/echo to a video's audio using ffmpeg")
    p.add_argument("-i", "--input", required=True, help="Input video file (mp4/mkv/...)" )
    p.add_argument("-o", "--output", required=True, help="Output video file")
    p.add_argument("--ir", help="Path to impulse response WAV. If provided uses convolution (afir).")
    p.add_argument("--sr", type=int, default=16000, help="Target sample rate for audio processing (default 16000)")
    p.add_argument("--ingain", type=float, default=0.8, help="Input gain for aecho (default 0.8)")
    p.add_argument("--outgain", type=float, default=0.9, help="Output gain for aecho (default 0.9)")
    p.add_argument("--delays", type=int, nargs='+', default=[1000, 1800], help="Delays in ms for aecho (space separated)")
    p.add_argument("--decays", type=float, nargs='+', default=[0.3, 0.2], help="Decay factors for aecho (space separated)")
    p.add_argument("--keep-temp", action='store_true', help="Keep temporary files for inspection")
    return p


def main():
    args = build_parser().parse_args()
    check_ffmpeg()

    if not os.path.exists(args.input):
        print("Input file not found:", args.input, file=sys.stderr)
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="reverb_")
    try:
        in_wav = os.path.join(tmpdir, "in.wav")
        out_wav = os.path.join(tmpdir, "out.wav")

        # extract audio (mono, target sr)
        cmd = [
            "ffmpeg", "-y", "-i", args.input,
            "-vn", "-ac", "1", "-ar", str(args.sr), "-acodec", "pcm_s16le", in_wav
        ]
        run(cmd)

        # If IR provided, convolve using afir
        if args.ir:
            if not os.path.exists(args.ir):
                print("IR file not found:", args.ir, file=sys.stderr)
                sys.exit(1)
            ir_conv = os.path.join(tmpdir, "ir_conv.wav")
            # convert IR to mono/sr
            run(["ffmpeg", "-y", "-i", args.ir, "-ac", "1", "-ar", str(args.sr), ir_conv])
            # convolve
            run(["ffmpeg", "-y", "-i", in_wav, "-i", ir_conv, "-filter_complex", "[0:a][1:a]afir", out_wav])
        else:
            # construct aecho parameters
            delays = "|".join(str(d) for d in args.delays)
            decays = "|".join(str(d) for d in args.decays)
            afilt = f"aecho={args.ingain}:{args.outgain}:{delays}:{decays}"
            run(["ffmpeg", "-y", "-i", in_wav, "-af", afilt, out_wav])

        # merge processed audio back into video (copy video stream)
        run([
            "ffmpeg", "-y", "-i", args.input, "-i", out_wav,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", args.output
        ])

        print("Wrote:", args.output)
        if args.keep_temp:
            print("Temporary files kept at:", tmpdir)
        else:
            shutil.rmtree(tmpdir)

    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("Error:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
