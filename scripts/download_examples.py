"""Download Ultralytics' tiny example datasets (coco8, coco8-seg, dota8) for trying out Dataset Doctor.

They are not committed to the repo: DOTA images carry an academic-use-only license.

    python scripts/download_examples.py examples/
    dataset-doctor scan examples/dota8 --format yolo-obb
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
DATASETS = {  # name: (format, number of classes)
    "coco8": ("yolo-hbb", 80),
    "coco8-seg": ("yolo-seg", 80),
    "dota8": ("yolo-obb", 15),
}


def main(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, (fmt, nc) in DATASETS.items():
        print(f"downloading {name} ...")
        with urllib.request.urlopen(f"{BASE_URL}/{name}.zip") as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extractall(dest)
        (dest / name / "data.yaml").write_text(f"train: images/train\nval: images/val\nnc: {nc}\n")
        print(f"  dataset-doctor scan {dest / name} --format {fmt}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
