"""Clone-kompatible Dimensionen für generate.py.
Constraints:
  - width  % 128 == 0  (VAE: width//2 muss /32 teilbar und gerade sein)
  - height % 128 == 0  (VAE: height//2 muss /32 teilbar und gerade sein)
  - ratio deviation minimal (best-effort)
"""

_VIDEO_RATIOS = {
    "16:9": (16, 9), "9:16": (9, 16),
    "21:9": (21, 9), "9:21": (9, 21),
    "3:2": (3, 2), "2:3": (2, 3),
    "4:3": (4, 3), "3:4": (3, 4),
    "4:5": (4, 5), "5:4": (5, 4),
    "1:1": (1, 1), "1:2": (1, 2), "2:1": (2, 1),
}

_VIDEO_QUALITY_TARGET = {
    "240p": 240, "360p": 360, "480p": 480, "720p": 720,
    "1080p": 1080, "1440p": 1440, "2160p": 2160, "4k": 4096,
}

ALIGN = 64  # two-stage pipeline requires width % 64 == 0 and height % 64 == 0


def _nearest_aligned(val):
    """Round to nearest multiple of ALIGN."""
    return max(ALIGN, round(val / ALIGN) * ALIGN)


def resolve_clone(ratio_str, quality_str):
    """Bestes /128-kompatibles (w, h) für gegebene ratio+quality.
    Der GRÖSSERE Wert liegt möglichst nah am Quality-Target.
    Der kleinere Wert wird aus dem Ratio berechnet und auf /128 gerundet.
    """
    rw, rh = _VIDEO_RATIOS[ratio_str]
    ratio = rw / rh
    target = _VIDEO_QUALITY_TARGET[quality_str]

    # Der größere Wert soll ~target sein
    target_aligned = _nearest_aligned(target)

    if ratio >= 1.0:
        # width >= height
        w = target_aligned
        h = _nearest_aligned(w / ratio)
    else:
        # height > width
        h = target_aligned
        w = _nearest_aligned(h * ratio)

    # Prüfe ob es mit target+ALIGN oder target-ALIGN ein besseres Ratio-Match gibt
    best = (w, h)
    best_dev = abs(w / h - ratio) / ratio

    for offset in [-ALIGN, ALIGN]:
        if ratio >= 1.0:
            w2 = target_aligned + offset
            if w2 < ALIGN:
                continue
            h2 = _nearest_aligned(w2 / ratio)
        else:
            h2 = target_aligned + offset
            if h2 < ALIGN:
                continue
            w2 = _nearest_aligned(h2 * ratio)
        dev2 = abs(w2 / h2 - ratio) / ratio
        if dev2 < best_dev:
            best_dev = dev2
            best = (w2, h2)

    return best, best_dev


if __name__ == "__main__":
    print(f"{'params':<20} {'width':>6} {'height':>6} {'width/2':>8} {'height/2':>9} {'w/2 %32':>8} {'h/2 %32':>8} {'w/2/32 even':>12} {'h/2/32 even':>12} {'Abweichung':>11}")
    print("-" * 105)
    for quality in _VIDEO_QUALITY_TARGET:
        for ratio in _VIDEO_RATIOS:
            result, dev = resolve_clone(ratio, quality)
            if result:
                w, h = result
                params = f"{quality}, {ratio}"
                w2, h2 = w // 2, h // 2
                w2_mod32 = w2 % 32
                h2_mod32 = h2 % 32
                w2_div32_even = "✓" if (w2 // 32) % 2 == 0 else "✗"
                h2_div32_even = "✓" if (h2 // 32) % 2 == 0 else "✗"
                print(f"{params:<20} {w:>6} {h:>6} {w2:>8} {h2:>9} {w2_mod32:>8} {h2_mod32:>8} {w2_div32_even:>12} {h2_div32_even:>12} {dev:>10.1%}")
