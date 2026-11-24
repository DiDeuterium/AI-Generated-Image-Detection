"""Train a small, pixel-only forensic baseline on processed training images."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ssd.data import EnsureMinSize, read_splits
from ssd.domain_shift import legacy_degrade
from ssd.engine import compute_metrics
from ssd.target_profile import FixedTargetMatch, validation_groups

PATCH = 96
WINDOW = np.outer(np.hanning(PATCH), np.hanning(PATCH))
CENTERS = ((.5, .5), (.15, .15), (.85, .15), (.15, .85), (.85, .85))
Y_FREQ = np.fft.fftfreq(PATCH)[:, None]
X_FREQ = np.fft.rfftfreq(PATCH)[None, :]
RADIUS = np.hypot(Y_FREQ, X_FREQ)
AWAY_FROM_JPEG = ((np.abs(X_FREQ * 8 - np.rint(X_FREQ * 8)) > .08)
                  & (np.abs(Y_FREQ * 8 - np.rint(Y_FREQ * 8)) > .08))
BANDS = [(RADIUS >= lower) & (RADIUS < upper) & AWAY_FROM_JPEG
         for lower, upper in zip((.04, .12, .23, .36, .48), (.12, .23, .36, .48, .72))]


def patch_features(patch):
    pixels = patch.astype(np.float64)
    center = pixels[1:-1, 1:-1]
    neighbors = (pixels[:-2, 1:-1], pixels[2:, 1:-1],
                 pixels[1:-1, :-2], pixels[1:-1, 2:])
    residual = center - sum(neighbors) * .25
    magnitude = np.sqrt(np.mean(residual**2))
    normalized = residual / max(magnitude, .5)
    features = [np.log1p(magnitude), *np.quantile(np.abs(normalized), [.5, .9, .99])]
    for shift in (1, 2, 3):
        features.extend((np.mean(normalized[shift:, :] * normalized[:-shift, :]),
                         np.mean(normalized[:, shift:] * normalized[:, :-shift])))
    spectrum = np.abs(np.fft.rfft2((pixels - pixels.mean()) * WINDOW)) ** 2
    energy = sum(spectrum[band].sum() for band in BANDS)
    features.extend(np.log1p(1000 * spectrum[band].sum() / max(energy, 1.)) for band in BANDS)
    return np.asarray(features)


def local_features(image):
    result = []
    pixels = np.asarray(EnsureMinSize(PATCH)(image.convert("L")))
    height, width = pixels.shape
    for fx, fy in CENTERS:
        x, y = round((width - PATCH) * fx), round((height - PATCH) * fy)
        result.append(patch_features(pixels[y:y + PATCH, x:x + PATCH]))
    features = np.stack(result)
    return np.concatenate((np.median(features, axis=0), np.quantile(features, .8, axis=0)))


def extract(rows, transforms):
    features = {case: [] for case in transforms}
    for number, row in enumerate(rows, 1):
        with Image.open(row["path"]) as file:
            image = file.convert("RGB")
        for case, transform in transforms.items():
            features[case].append(local_features(transform(image, row["image_id"])))
        if number % 800 == 0:
            print(f"{','.join(transforms)}: {number}/{len(rows)}", flush=True)
    return {case: np.stack(values) for case, values in features.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train, val = read_splits(Path("splits/train_val.csv"), Path("dataset"))
    profiles = json.loads(Path("fqm/runs/target_profile_20261121/profiles.json").read_text())
    _, heldout = validation_groups(val)
    match = FixedTargetMatch(val, profiles, "both", seed=126)
    training = extract(train, {
        "both": FixedTargetMatch(train, profiles, "both", seed=310),
        "moderate": lambda image, key: legacy_degrade(image, key, "moderate"),
    })
    evaluation = extract(heldout, {
        "orig": lambda image, _: image,
        "both": match,
        "both_blur": FixedTargetMatch(val, profiles, "both", seed=126, other_blur_radius=.35),
        "legacy_strong": lambda image, key: legacy_degrade(image, key, "strong"),
    })
    x = np.concatenate(list(training.values()))
    y = np.tile([row["label"] for row in train], len(training))
    labels = np.array([row["label"] for row in heldout])
    cohorts = np.array([match.cohort(row["image_id"]) for row in heldout])
    models = {
        "linear": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
        "trees": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                                early_stopping=False, random_state=42),
    }
    reports = {}
    scores = {"ids": [row["image_id"] for row in heldout], "labels": labels,
              "cohorts": cohorts}
    for model_name, model in models.items():
        model.fit(x, y)
        model_report = {}
        for case, features in evaluation.items():
            probabilities = model.predict_proba(features)
            scores[f"{model_name}_{case}"] = probabilities[:, 1]
            metrics = {"all": compute_metrics(labels, probabilities)}
            for group in ("width500", "other"):
                keep = cohorts == group
                metrics[group] = compute_metrics(labels[keep], probabilities[keep])
            model_report[case] = metrics
            print(model_name, case, metrics["all"]["accuracy"], metrics["all"]["auroc"], flush=True)
        reports[model_name] = model_report
    (args.output / "metrics.json").write_text(json.dumps(reports, indent=2))
    np.savez(args.output / "scores.npz", **scores)


if __name__ == "__main__":
    main()
