"""Train and evaluate static hand-pose classifiers, split by person.

Data sources (any combination):
  --clips   data/clips           clips recorded in the web demo ("Collect training data")
  --hagrid  data/hagrid_lm.npz   landmarks extracted with scripts/extract_hagrid.py
  --synthetic                    synthetic hands - pipeline smoke test ONLY, never report these numbers

Example:
  python scripts/train_static.py --clips data/clips --hagrid data/hagrid_lm.npz

Writes models/static_pose.joblib (picked up automatically by the app) and
reports/static_pose.md with macro-F1 per model, per-class scores, a confusion
matrix and inference time per frame.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from archimedes_gestures import synthetic  # noqa: E402
from archimedes_gestures.classifier import LABELS, RulePoseClassifier  # noqa: E402
from archimedes_gestures.features import extract_batch  # noqa: E402
from archimedes_gestures.recorder import load_clips  # noqa: E402


def load_data(args):
    parts = []
    if args.clips and Path(args.clips).exists():
        lm, hand, y, subj, _ = load_clips(args.clips)
        parts.append((lm, hand, y, subj))
        print(f"clips:  {len(y)} frames, {len(set(subj))} subjects")
    if args.hagrid:
        d = np.load(args.hagrid, allow_pickle=False)
        parts.append((d["landmarks"], d["handedness"], d["labels"], d["subjects"]))
        print(f"hagrid: {len(d['labels'])} images, {len(set(d['subjects']))} subjects")
    if args.synthetic:
        lm, hand, y, subj = synthetic.dataset(300, seed=0)
        parts.append((lm, np.array(hand), y, subj))
        print("synthetic: smoke-test data - do not report these numbers")
    if not parts:
        sys.exit("No data. Record clips in the web demo, extract HaGRID, or pass --synthetic for a smoke test.")
    lm = np.concatenate([p[0] for p in parts])
    hand = np.concatenate([np.asarray(p[1]) for p in parts])
    y = np.concatenate([np.asarray(p[2]) for p in parts])
    subj = np.concatenate([np.asarray(p[3]) for p in parts])
    keep = np.isin(y, LABELS)
    return lm[keep], hand[keep], y[keep], subj[keep]


def mirror_augment(lm, hand, y, subj):
    """Add an x-mirrored copy with swapped handedness.

    None of the poses depend on chirality, and datasets disagree on whether
    images are mirrored, so this makes the model indifferent to that.
    """
    m = lm.copy()
    m[..., 0] = 1 - m[..., 0]
    h2 = np.where(hand == "Left", "Right", "Left")
    return (np.concatenate([lm, m]), np.concatenate([hand, h2]), np.concatenate([y, y]), np.concatenate([subj, subj]))


MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=600, early_stopping=True,
                                                                 random_state=0)),
    "gboost": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--hagrid")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="models/static_pose.joblib")
    ap.add_argument("--report", default="reports/static_pose.md")
    args = ap.parse_args()

    lm, hand, y, subj = load_data(args)
    n_subj = len(set(subj))
    folds = min(args.folds, n_subj)
    if folds < 2:
        sys.exit("Need at least 2 subjects to evaluate on unseen people. Record a second person.")
    X = extract_batch(lm, list(hand))
    labels = sorted(set(y))
    print(f"{len(y)} samples, {len(labels)} classes, {n_subj} subjects, {folds}-fold split by subject")

    results = {}
    # rules baseline needs no training: score it on everything
    rules = RulePoseClassifier()
    t0 = time.perf_counter()
    pred = np.array([rules.predict(a, h)[0] for a, h in zip(lm, hand)])
    rules_ms = (time.perf_counter() - t0) / len(y) * 1000
    results["rules"] = (f1_score(y, pred, average="macro", labels=labels), y, pred, rules_ms)

    gkf = GroupKFold(n_splits=folds)
    for name, make in MODELS.items():
        ys, ps = [], []
        for tr, te in gkf.split(X, y, subj):
            ltr, htr, ytr, str_ = lm[tr], hand[tr], y[tr], subj[tr]
            if not args.no_mirror:
                ltr, htr, ytr, str_ = mirror_augment(ltr, htr, ytr, str_)
            m = make().fit(extract_batch(ltr, list(htr)), ytr)
            ys.append(y[te])
            ps.append(m.predict(X[te]))
        ys, ps = np.concatenate(ys), np.concatenate(ps)
        t0 = time.perf_counter()  # single-frame latency, as in the live app
        for i in range(200):
            m.predict(X[i % len(X)][None])
        ms = (time.perf_counter() - t0) / 200 * 1000
        results[name] = (f1_score(ys, ps, average="macro", labels=labels), ys, ps, ms)
        print(f"  {name:7s} macro-F1 {results[name][0]:.3f}   {ms:.2f} ms/frame")
    print(f"  {'rules':7s} macro-F1 {results['rules'][0]:.3f}   {rules_ms:.2f} ms/frame")

    best = max((k for k in MODELS), key=lambda k: results[k][0])
    fl, fh, fy, fs = (lm, hand, y, subj) if args.no_mirror else mirror_augment(lm, hand, y, subj)
    final = MODELS[best]().fit(extract_batch(fl, list(fh)), fy)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(final, args.out)
    print(f"best: {best} -> saved {args.out}")

    _, ys, ps, _ = results[best]
    cm = confusion_matrix(ys, ps, labels=labels)
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Static pose classifier report", "",
        f"Samples: {len(y)} · classes: {len(labels)} · subjects: {n_subj} · {folds}-fold GroupKFold by subject"
        + (" · **synthetic data (smoke test only)**" if args.synthetic else ""), "",
        "| model | macro-F1 (unseen subjects) | ms / frame |", "|---|---|---|",
    ]
    for k, (f1, *_rest, ms) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"| {k}{' (saved)' if k == best else ''} | {f1:.3f} | {ms:.2f} |")
    lines += ["", f"## Per-class ({best})", "", "```", classification_report(ys, ps, labels=labels, digits=3), "```",
              "", f"## Confusion matrix ({best}; rows = true, cols = predicted)", "",
              "| | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for lab, row in zip(labels, cm):
        lines.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
    rep.write_text("\n".join(lines) + "\n")
    print(f"report -> {rep}")


if __name__ == "__main__":
    main()
