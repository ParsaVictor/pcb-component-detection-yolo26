"""Tiled inference for small components on high-resolution boards.

A populated board downscaled to 640 px leaves an 0402 resistor a handful of
pixels wide — below what the detector can resolve. Tiling runs inference on
overlapping crops at native resolution and maps the boxes back, which is the
standard remedy when object size is small relative to the frame.

Only worth using when the source images are genuinely high resolution;
on already-small images it costs time and gains nothing.
"""

import numpy as np


def _iou_matrix(box, boxes):
    """IoU of one box against an array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area + areas - intersection
    return np.where(union > 0, intersection / union, 0.0)


def non_max_suppression(boxes, iou_threshold=0.45):
    """Class-aware NMS over ``(x1, y1, x2, y2, score, class)`` rows.

    Tiles overlap by design, so the same component is detected more than once
    near a seam; without this the counts are inflated.
    """
    if not len(boxes):
        return []

    boxes = np.asarray(boxes, dtype=np.float64)
    keep = []

    for cls in np.unique(boxes[:, 5]):
        subset = boxes[boxes[:, 5] == cls]
        order = np.argsort(-subset[:, 4])
        subset = subset[order]

        while len(subset):
            best = subset[0]
            keep.append(best)
            if len(subset) == 1:
                break
            ious = _iou_matrix(best[:4], subset[1:, :4])
            subset = subset[1:][ious < iou_threshold]

    return [tuple(b) for b in keep]


def tile_inference(model, image_path, tile_size=640, overlap=0.2, conf=0.25, iou=0.45):
    """Run the detector over overlapping tiles and return merged boxes.

    Returns a list of ``(x1, y1, x2, y2, score, class)`` in the coordinates of
    the original image, after NMS across tile boundaries.
    """
    # Imported here rather than at module scope so that non_max_suppression -
    # which is pure NumPy - stays usable without OpenCV installed.
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    height, width = image.shape[:2]
    step = max(1, int(tile_size * (1 - overlap)))

    # Ranges always include the final tile, so the right and bottom edges of
    # the board are covered even when the size is not a multiple of the step.
    y_positions = list(range(0, max(1, height - tile_size + 1), step))
    x_positions = list(range(0, max(1, width - tile_size + 1), step))
    if y_positions[-1] + tile_size < height:
        y_positions.append(max(0, height - tile_size))
    if x_positions[-1] + tile_size < width:
        x_positions.append(max(0, width - tile_size))

    collected = []
    for y in y_positions:
        for x in x_positions:
            tile = image[y:y + tile_size, x:x + tile_size]
            if tile.size == 0:
                continue

            # Pass the array directly - no temporary files on disk.
            results = model.predict(
                source=tile, imgsz=tile_size, conf=conf, iou=iou, verbose=False
            )
            if not results:
                continue

            for box in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, cls = box[:6]
                collected.append((x1 + x, y1 + y, x2 + x, y2 + y, score, int(cls)))

    return non_max_suppression(collected, iou_threshold=iou)
