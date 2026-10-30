# Dataset Doctor

Catch broken YOLO labels before you train on them.

YOLO labels are plain text, so a broken dataset usually trains without a single error: coordinates above 1.0
get clipped or dropped, OBB corners in the wrong order form a "butterfly", a `class 7` hides in a 5-class
dataset, and the same image sits in both train and val. You find out when the metrics look wrong.
Dataset Doctor finds these problems up front.

```
$ dataset-doctor scan ./dataset --format yolo-obb
Dataset Doctor v0.1.0 — yolo-obb — 4,812 images / 31,077 labels

 ERROR    self_intersecting_obb         41
 ERROR    label_without_image           12
 ERROR    class_id_out_of_range          3   (max=4, found 7)
 WARNING  coord_out_of_range           118   (max 1.032)
 WARNING  duplicate_label               27
 WARNING  split_leak                     9   (train ↔ val)
 INFO     image_without_label           64
```

## Install

```bash
pip install -e .            # from a checkout; Python 3.10+
```

## Usage

```bash
dataset-doctor scan ./dataset --format yolo-obb
dataset-doctor scan ./dataset --format yolo-hbb --data data.yaml --report html --out report/
dataset-doctor scan ./dataset --format yolo-seg --fail-on error     # CI: exit 1 if any ERROR
dataset-doctor fix  ./dataset --format yolo-obb --out ./dataset_fixed --clip-coords --reorder-obb --dedup
```

- `--format` (required): `yolo-hbb` (`cls cx cy w h`), `yolo-obb` (`cls x1 y1 ... x4 y4`) or
  `yolo-seg` (`cls x1 y1 ... xn yn`), all normalized to 0–1.
- `--data`: a `data.yaml` with `train`/`val`/`test` image folders (relative to the dataset) and `names` or `nc`.
  Defaults to `DATASET/data.yaml` if it exists. Without one, splits are the folders under `images/` and class
  ids are only checked for being non-negative integers.
- `--report json|html` writes `report.json` or `index.html` into `--out` (default `report/`). The HTML report
  has dataset statistics, a filter per check, and every problem sample with its labels drawn on the image
  (problem labels in red).
- `--fail-on error|warning` exits with status 1 if any issue at or above that severity was found.

### Dataset layout

The Ultralytics layout: `images/<split>/...` with the matching `labels/<split>/....txt` next to it (the last
`images` folder in an image path is swapped for `labels`). An empty label file marks a background image.

### fix

`fix` copies the dataset to `--out` (which must not exist) and repairs the copy; the original is never
modified. Every change is listed in `fix_log.json` with the line before and after.

| Flag | Repairs |
|---|---|
| `--clip-coords` | `coord_out_of_range`: clips coordinates into 0–1 (HBB: clips the box edges). Clipping a rotated box that crosses the image border bends it, so it may then show up as `non_rectangular_obb`. |
| `--reorder-obb` | `self_intersecting_obb`, `inconsistent_winding`: puts the corners in convex order with the dataset's majority winding, keeping the first corner first. |
| `--dedup` | `duplicate_label` (keeps the first line) and `duplicate_image` within a split (keeps the first file). |

Anything that needs a judgement call — `split_leak`, `label_conflict`, `class_id_out_of_range`, … — is only
reported.

## Checks

| Code | Severity | Meaning |
|---|---|---|
| `image_without_label` | INFO | Image has no label file (may be a background image) |
| `label_without_image` | ERROR | Label file has no image |
| `corrupt_image` | ERROR | Image can't be decoded, or a JPEG is truncated |
| `empty_file` | ERROR / INFO | 0-byte image (ERROR); empty label file, i.e. background (INFO) |
| `duplicate_image` | WARNING | Same file contents twice within a split |
| `split_leak` | WARNING | Same file contents in more than one split |
| `wrong_column_count` | ERROR | Wrong number of values for the format |
| `parse_error` | ERROR | Not a number, NaN or inf |
| `class_id_out_of_range` | ERROR | Class id not in `data.yaml`, negative, or not an integer |
| `coord_out_of_range` | WARNING / ERROR | Coordinate outside 0–1: up to 0.05 over is a WARNING, more is an ERROR |
| `non_positive_size` | ERROR | HBB width or height ≤ 0 |
| `tiny_box` | WARNING | Shortest side under 2 px |
| `self_intersecting_obb` | ERROR | OBB corners out of order (butterfly) |
| `non_rectangular_obb` | WARNING | An OBB corner is more than 10° off 90° |
| `degenerate_obb` | ERROR | OBB area under 1 px² |
| `inconsistent_winding` | WARNING | OBB wound the opposite way to the rest of the dataset |
| `invalid_polygon` | ERROR | Segment with < 3 points, zero area, or edges that cross each other. Edges that only touch or run back over each other are fine: that is how Ultralytics joins the parts of a multi-part COCO mask |
| `duplicate_label` | WARNING | Same class, IoU ≥ 0.95 with an earlier label in the image |
| `label_conflict` | WARNING | Different class, IoU ≥ 0.95 with an earlier label in the image |

Geometry is checked in pixels (normalized coordinates × image size): on a 1920×1080 image a rotated rectangle
looks like a parallelogram in normalized coordinates.

## Example datasets

```bash
python scripts/download_examples.py examples/
dataset-doctor scan examples/dota8 --format yolo-obb --report html
```

Downloads Ultralytics' `coco8`, `coco8-seg` and `dota8`. They are not committed: DOTA is licensed for
academic use only.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests build deliberately broken datasets with `tests/fixtures.make_dataset`.
