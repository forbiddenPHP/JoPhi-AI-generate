#!/usr/bin/env python3
"""Calculate valid video dimensions (multiples of 64) for aspect ratios.

Usage:
  python ratios.py                    # Show all ratios and quality tiers
  python ratios.py 1920 1080          # Find nearest valid dimensions for 1920x1080
  python ratios.py 16:9 hd            # Lookup ratio + quality tier
"""

import sys
from fractions import Fraction

ALIGN = 64  # Both dimensions must be multiples of this
STAGE1_ALIGN = 32  # Stage 1 operates at half resolution (must align after /2)

RATIOS = {
    "16:9": Fraction(16, 9),
    "9:16": Fraction(9, 16),
    "21:9": Fraction(21, 9),  # Cinemascope (2.39:1 ≈ 21:9)
    "9:21": Fraction(9, 21),
    "3:2": Fraction(3, 2),
    "2:3": Fraction(2, 3),
    "4:3": Fraction(4, 3),
    "3:4": Fraction(3, 4),
    "4:5": Fraction(4, 5),
    "5:4": Fraction(5, 4),
    "1:1": Fraction(1, 1),
    "1:2": Fraction(1, 2),
    "2:1": Fraction(2, 1),
}

QUALITY_TIERS = {
    "240p": 448,
    "360p": 576,
    "480p": 640,
    "720p": 1280,
    "1080p": 1920,
    "1440p": 2560,
    "2160p": 3840,
    "4k": 4096,
}

MAX_DIM = 4096  # Don't go beyond this


def valid_dims_for_ratio(ratio: Fraction, max_dim: int = MAX_DIM) -> list[tuple[int, int]]:
    """Find all valid (width, height) pairs for a ratio, both multiples of ALIGN."""
    results = []
    for w in range(ALIGN, max_dim + 1, ALIGN):
        h = w / float(ratio)
        h_rounded = round(h / ALIGN) * ALIGN
        if h_rounded < ALIGN or h_rounded > max_dim:
            continue
        actual_ratio = w / h_rounded
        deviation = abs(actual_ratio - float(ratio)) / float(ratio) * 100
        if deviation < 5.0:  # Max 5% deviation
            results.append((w, h_rounded, deviation))
    # Deduplicate
    seen = set()
    unique = []
    for w, h, dev in results:
        if (w, h) not in seen:
            seen.add((w, h))
            unique.append((w, h, dev))
    return unique


def nearest_valid(target_w: int, target_h: int) -> list[tuple[int, int, float, str]]:
    """Find nearest valid dimensions for arbitrary target dimensions."""
    target_ratio = target_w / target_h
    results = []
    # Search around target dimensions
    for w in range(ALIGN, MAX_DIM + 1, ALIGN):
        for h in range(ALIGN, MAX_DIM + 1, ALIGN):
            # Only consider dimensions roughly in the ballpark
            if abs(w - target_w) > 256 or abs(h - target_h) > 256:
                continue
            actual_ratio = w / h
            ratio_dev = abs(actual_ratio - target_ratio) / target_ratio * 100
            size_dev = abs(w * h - target_w * target_h) / (target_w * target_h) * 100
            # Find matching named ratio
            ratio_name = ""
            for name, r in RATIOS.items():
                if abs(actual_ratio - float(r)) / float(r) < 0.01:
                    ratio_name = name
                    break
            results.append((w, h, ratio_dev, size_dev, ratio_name))
    results.sort(key=lambda x: (x[2] + x[3]))  # Sort by combined deviation
    return results[:10]


def print_all_ratios():
    """Print all ratios with their valid dimensions per quality tier."""
    print(f"{'Ratio':<8} {'Quality':<6} {'Width':>6} {'Height':>6} {'Pixels':>12} {'Deviation':>10}")
    print("-" * 56)
    for name, ratio in RATIOS.items():
        dims = valid_dims_for_ratio(ratio)
        for tier_name, tier_max in QUALITY_TIERS.items():
            # Find the largest dimensions where max(w,h) <= tier_max
            best = None
            for w, h, dev in dims:
                if max(w, h) <= tier_max:
                    best = (w, h, dev)
            if best:
                w, h, dev = best
                dev_str = "exact" if dev == 0 else f"{dev:.2f}%"
                print(f"{name:<8} {tier_name:<6} {w:>6} {h:>6} {w * h:>12,} {dev_str:>10}")
        print()


def print_nearest(target_w: int, target_h: int):
    """Print nearest valid dimensions for a target."""
    print(f"Target: {target_w}×{target_h} (ratio {target_w/target_h:.4f})")
    print()
    results = nearest_valid(target_w, target_h)
    print(f"{'Width':>6} {'Height':>6} {'Ratio Dev':>10} {'Size Dev':>10} {'Named Ratio':<10}")
    print("-" * 50)
    for w, h, ratio_dev, size_dev, ratio_name in results:
        print(f"{w:>6} {h:>6} {ratio_dev:>9.2f}% {size_dev:>9.2f}% {ratio_name:<10}")


def print_ratio_quality(ratio_name: str, quality: str):
    """Print dimensions for a specific ratio + quality combo."""
    if ratio_name not in RATIOS:
        print(f"Unknown ratio: {ratio_name}")
        print(f"Available: {', '.join(RATIOS.keys())}")
        sys.exit(1)
    if quality not in QUALITY_TIERS:
        print(f"Unknown quality: {quality}")
        print(f"Available: {', '.join(QUALITY_TIERS.keys())}")
        sys.exit(1)
    ratio = RATIOS[ratio_name]
    tier_max = QUALITY_TIERS[quality]
    dims = valid_dims_for_ratio(ratio, tier_max)
    if not dims:
        print(f"No valid dimensions for {ratio_name} at {quality}")
        sys.exit(1)
    w, h, dev = dims[-1]  # Largest that fits
    dev_str = "exact" if dev == 0 else f"({dev:.2f}% deviation)"
    print(f"{ratio_name} {quality}: {w}×{h} {dev_str}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_all_ratios()
    elif len(sys.argv) == 3:
        arg1, arg2 = sys.argv[1], sys.argv[2]
        if ":" in arg1:
            print_ratio_quality(arg1, arg2)
        else:
            print_nearest(int(arg1), int(arg2))
    else:
        print(__doc__)
