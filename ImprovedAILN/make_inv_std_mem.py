#!/usr/bin/env python3
# inv_std (1/std) LUT .mem generator for Verilog $readmemh
# Each line is a hex word: round( (1<<Q_INV) / max(addr,1) ), addr=0..255
# Usage:
#   python3 make_inv_std_mem.py --q 14 --outfile inv_std_q14.mem
# Options:
#   --q <int>           : fractional bits (Q_INV), default 14
#   --outfile <path>    : output filename, default inv_std_q<q>.mem
#   --size <int>        : table size (default 256 for 8-bit std)
#   --map_zero_to_one   : map addr 0 to denom=1 (avoid div-by-zero), default ON
#   --no-clamp          : do not clamp to (1<<Q_INV)-1 (rarely needed)
#   --start <int> --end <int> : optional address range (inclusive), default 0..size-1

import argparse
from math import isfinite

def main():
    ap = argparse.ArgumentParser(description="Generate 1/std LUT .mem for $readmemh")
    ap.add_argument("--q", type=int, default=14, help="fractional bits (Q_INV)")
    ap.add_argument("--outfile", type=str, default=None, help="output .mem filename")
    ap.add_argument("--size", type=int, default=256, help="table size (default 256)")
    ap.add_argument("--map_zero_to_one", action="store_true", default=True,
                    help="map addr 0 to denom=1 to avoid div-by-zero")
    ap.add_argument("--no-map_zero_to_one", dest="map_zero_to_one", action="store_false")
    ap.add_argument("--no-clamp", action="store_true", help="disable clamp to max value")
    ap.add_argument("--start", type=int, default=0, help="start address (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="end address (inclusive)")
    args = ap.parse_args()

    Q = args.q
    N = args.size
    if args.end is None:
        args.end = N - 1

    if args.outfile is None:
        args.outfile = f"inv_std_q{Q}.mem"

    max_val = (1 << Q) - 1
    hex_digits = (Q + 3) // 4

    with open(args.outfile, "w") as f:
        for k in range(args.start, args.end + 1):
            denom = 1 if (k == 0 and args.map_zero_to_one) else max(k, 1)
            val = int(round((1 << Q) / denom))
            if not args.no_clamp and val > max_val:
                val = max_val
            f.write(f"{val:0{hex_digits}X}\n")

    print(f"Wrote {args.outfile} (Q={Q}, entries={args.end-args.start+1})")

if __name__ == "__main__":
    main()