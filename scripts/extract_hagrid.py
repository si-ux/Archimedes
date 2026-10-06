"""Turn HaGRID images into labelled hand landmarks (no recording needed).

HaGRID (https://github.com/hukenovs/hagrid) is a large public dataset of hand
gesture photos with per-image ``user_id``s, which lets us test on people the
model never saw. Download the classes listed in HAGRID_TO_OURS plus the
annotation JSONs (the small "hagrid-sample" subset is enough to start), then:

  python scripts/extract_hagrid.py --images path/to/hagrid/images \
      --annotations path/to/hagrid/annotations --per-class 1500 --out data/hagrid_lm.npz

For each image we run MediaPipe, keep the detected hand whose box overlaps the
annotated gesture box best, and store its 21 landmarks. "no_gesture" boxes in
the same images become the "none" class. Check HaGRID's licence before
redistributing anything derived from it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from archimedes_gestures.tracker import HandTracker  # noqa: E402

# HaGRID class -> our vocabulary
HAGRID_TO_OURS = {
    "palm": "open_palm",
    "stop": "open_palm",
    "fist": "fist",
    "one": "point",
    "peace": "peace",
    "like": "thumbs_up",
    "ok": "pinch",
    "no_gesture": "none",
}


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def find_annotations(root: Path, cls: str) -> dict:
    for p in [root / f"{cls}.json", *root.rglob(f"{cls}.json")]:
        if p.exists():
            return json.loads(p.read_text())
    return {}


def find_image(root: Path, cls: str, image_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        for p in (root / cls / f"{image_id}{ext}", root / f"train_val_{cls}" / f"{image_id}{ext}"):
            if p.exists():
                return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--out", default="data/hagrid_lm.npz")
    args = ap.parse_args()

    tracker = HandTracker(mode="image", num_hands=2, min_conf=0.4)
    lms, hands, labels, subjects = [], [], [], []
    for cls in [c for c in HAGRID_TO_OURS if c != "no_gesture"]:
        ann = find_annotations(args.annotations, cls)
        if not ann:
            print(f"skip {cls}: no annotation file found")
            continue
        got = {"gesture": 0, "none": 0}
        for image_id, a in ann.items():
            if got["gesture"] >= args.per_class:
                break
            img = find_image(args.images, cls, image_id)
            if img is None:
                continue
            det = tracker.detect_file(img)
            if not det:
                continue
            for box, lab in zip(a["bboxes"], a["labels"]):
                target = HAGRID_TO_OURS.get(lab)
                if target is None or (target == "none" and got["none"] >= args.per_class // 4):
                    continue
                gbox = (box[0], box[1], box[0] + box[2], box[1] + box[3])  # HaGRID: x, y, w, h (relative)
                best, best_iou = None, 0.1
                for d in det:
                    xy = np.array(d["landmarks"])[:, :2]
                    hb = (*xy.min(0), *xy.max(0))
                    s = iou(gbox, hb)
                    if s > best_iou:
                        best, best_iou = d, s
                if best is None:
                    continue
                # HaGRID photos are not mirrored; flip into selfie view so they match the live app
                arr = np.array(best["landmarks"])
                arr[:, 0] = 1 - arr[:, 0]
                lms.append(arr)
                lead = str(a.get("leading_hand", "right")).capitalize()
                hands.append(lead if target != "none" else best["handedness"])
                labels.append(target)
                subjects.append(str(a.get("user_id", image_id)))
                got["none" if target == "none" else "gesture"] += 1
        print(f"{cls:>8s} -> {HAGRID_TO_OURS[cls]:9s} {got['gesture']} samples (+{got['none']} none)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, landmarks=np.array(lms), handedness=np.array(hands), labels=np.array(labels),
                        subjects=np.array(subjects))
    print(f"saved {len(labels)} samples from {len(set(subjects))} people -> {args.out}")


if __name__ == "__main__":
    main()
