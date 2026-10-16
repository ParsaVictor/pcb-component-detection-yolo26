"""Shared helpers for reading YOLO label files that may contain polygons."""


def parse_line(line):
    """Return (cls, cx, cy, w, h) or None.

    Roboflow YOLO26 exports may write segmentation polygons
    ("cls x1 y1 x2 y2 ...") instead of boxes ("cls cx cy w h"); for those
    the tight bounding box of the polygon is returned.
    """
    p = line.split()
    if len(p) < 5:
        return None
    cls = int(float(p[0]))
    v = [float(x) for x in p[1:]]
    if len(v) == 4:
        return cls, v[0], v[1], v[2], v[3]
    xs, ys = v[0::2], v[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
