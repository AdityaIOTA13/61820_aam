#!/usr/bin/env python3
"""Multiply animated GIF playback speed by changing per-frame duration (no re-encode)."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_gif", type=Path)
    p.add_argument("output_gif", type=Path, nargs="?", default=None)
    p.add_argument(
        "--factor",
        type=float,
        default=3.0,
        help="Speed multiplier (3 = play 3x faster → duration / 3)",
    )
    args = p.parse_args()
    inp = args.input_gif.expanduser().resolve()
    if args.output_gif is None:
        out = inp
    else:
        out = args.output_gif.expanduser().resolve()
    fac = float(args.factor)
    if fac <= 0:
        sys.exit("--factor must be > 0")

    reader = imageio.get_reader(str(inp))
    meta = reader.get_meta_data()
    old_dur = float(meta.get("duration", 20))
    new_dur = max(1, int(round(old_dur / fac)))

    tmp = Path(tempfile.mkstemp(suffix=".gif", prefix="respeed_")[1])
    writer = imageio.get_writer(str(tmp), mode="I", duration=new_dur, loop=meta.get("loop", 0))
    try:
        for im in reader:
            writer.append_data(im)
    finally:
        writer.close()
        reader.close()

    try:
        if out == inp:
            os.replace(str(tmp), str(inp))
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(tmp), str(out))
    finally:
        tmp.unlink(missing_ok=True)
    print(f"Wrote {out}  duration {old_dur:g}ms -> {new_dur}ms  ({fac:g}x faster)", file=sys.stderr)


if __name__ == "__main__":
    main()
