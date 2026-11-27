# Scores a saved frame for the posterised ("oil painting") look.
#
#   python band-scan.py frame.png [x0 y0 x1 y1]
#
# Pick a region that should be a smooth gradient: a plain wall, sky, skin. The score is the share of
# horizontal neighbour steps larger than 6 grey levels inside that region.
#   about 0.001   smooth
#   0.03 to 0.08  posterised
#   in between    look at it
# The default region suits a 1920x1080 capture with a plain wall on the left side.
import sys

import numpy as np
from PIL import Image

DEFAULT_BOX = (20, 150, 330, 700)


def score(path, box=DEFAULT_BOX):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    x0, y0, x1, y1 = box
    a = a[y0:y1, x0:x1]
    return float((np.abs(np.diff(a, axis=1)) > 6).mean())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: band-scan.py frame.png [x0 y0 x1 y1]")
        sys.exit(2)
    box = tuple(int(v) for v in sys.argv[2:6]) if len(sys.argv) >= 6 else DEFAULT_BOX
    s = score(sys.argv[1], box)
    print(f"{s:.3f}", "posterised" if s > 0.03 else ("check by eye" if s > 0.008 else "smooth"))
