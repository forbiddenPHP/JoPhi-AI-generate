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

_VIDEO_QUALITY = {
    "240p": 448, "360p": 576, "480p": 640, "720p": 1280,
    "1080p": 1920, "1440p": 2560, "2160p": 3840, "4k": 4096,
}


def resolve_clone(ratio_str, quality_str):
    """Bestes /128-kompatibles (w, h) für gegebene ratio+quality.
    Nimmt das Paar mit minimaler Ratio-Abweichung; bei Gleichstand das größere.
    """
    rw, rh = _VIDEO_RATIOS[ratio_str]
    ratio = rw / rh
    max_dim = _VIDEO_QUALITY[quality_str]

    best = None
    best_dev = float("inf")
    for w in range(128, max_dim + 1, 128):
        for h in range(128, max_dim + 1, 128):
            dev = abs(w / h - ratio) / ratio
            area = w * h
            if dev < best_dev or (dev == best_dev and area > best[0] * best[1]):
                best_dev = dev
                best = (w, h)
    return best, best_dev


if __name__ == "__main__":
    print(f"{'params':<20} {'width':>6} {'height':>6} {'width/2':>8} {'height/2':>9} {'w/2 %32':>8} {'h/2 %32':>8} {'w/2/32 even':>12} {'h/2/32 even':>12} {'Abweichung':>11}")
    print("-" * 105)
    for quality in _VIDEO_QUALITY:
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
