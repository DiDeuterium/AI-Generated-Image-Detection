"""Compare held-out proxy results, quantify label-shift misfit, and export a candidate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from audit_proxy_shift import CASES, ROOT

PROXIES = CASES[1:]
FEATURE_CASES = ("orig", "q75", "moderate", "strong")


def accuracy(y, prediction):
    return float(np.mean(y == prediction))


def metrics(y, margin, threshold=0.):
    prediction = margin > threshold
    return {"acc": accuracy(y, prediction), "auc": float(roc_auc_score(y, margin)),
            "r0": accuracy(y[y == 0], prediction[y == 0]),
            "r1": accuracy(y[y == 1], prediction[y == 1])}


def describe(margin):
    prob = expit(margin)
    return {"mean": float(prob.mean()), "median": float(np.median(prob)),
            "quantiles_10_25_75_90_99": np.quantile(prob, [.1, .25, .75, .9, .99]).tolist(),
            "fake_rate": float(np.mean(margin > 0))}


def score_fit(val, test, y, rng):
    bins = np.r_[-np.inf, np.linspace(-6, 6, 15), np.inf]
    components = np.stack([np.histogram(val[y == label], bins)[0] + .5 for label in (0, 1)])
    components /= components.sum(axis=1, keepdims=True)
    observed = np.histogram(test, bins)[0]

    def fit(counts):
        def loss(weight):
            mixture = (1 - weight) * components[0] + weight * components[1]
            return float(-counts @ np.log(mixture))

        weight = minimize_scalar(loss, bounds=(0, 1), method="bounded").x
        expected = counts.sum() * ((1 - weight) * components[0] + weight * components[1])
        statistic = float(np.sum((counts - expected) ** 2 / expected))
        tv = float(np.abs(counts - expected).sum() / counts.sum() / 2)
        return float(weight), statistic, tv, expected / counts.sum()

    weight, statistic, tv, probabilities = fit(observed)
    simulated = [fit(rng.multinomial(int(observed.sum()), probabilities))[1] for _ in range(399)]
    return {"fit_fake_fraction": weight, "hist_tv": tv,
            "conditional_bootstrap_p": float((1 + np.sum(np.asarray(simulated) >= statistic)) / 400)}


def evaluate_prior_matching(labels, scores, threshold, rng):
    """Test known-prior rank decoding on rebalanced labeled holdout proxies."""
    indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    results = {}
    for fraction in (.34, .4, .5, .57):
        num_fake = round(600 * fraction)
        deltas = {key: [] for key in ("prior_match", "force_40pct", "force_50pct")}
        for _ in range(50):
            ids = np.r_[rng.choice(indices[0], 600 - num_fake, replace=False),
                        rng.choice(indices[1], num_fake, replace=False)]
            y = labels[ids]
            score = scores[ids]
            baseline = accuracy(y, score > threshold)
            order = np.argsort(score, kind="stable")
            for strategy, count in (("prior_match", num_fake), ("force_40pct", 240),
                                    ("force_50pct", 300)):
                pred = np.zeros(600, dtype=bool)
                pred[order[-count:]] = True
                deltas[strategy].append(accuracy(y, pred) - baseline)
        results[str(fraction)] = {
            key: {"mean_acc_delta": float(np.mean(values)),
                  "trial_quantiles_05_95": np.quantile(values, [.05, .95]).tolist()}
            for key, values in deltas.items()
        }
    return results


def knn(output, name, submission, *, report_dir=None):
    results = {}
    test = np.load(output / f"{name}__test_orig.npz")
    y_votes = {}
    for case in FEATURE_CASES:
        bank = np.load(output / f"{name}__train_{case}.npz")
        val = np.load(output / f"{name}__val_{case}.npz")
        x = bank["embedding"]
        scale = x.std(axis=0).clip(min=1e-4)
        mean = x.mean(axis=0)

        def preprocess(z):
            return normalize((z - mean) / scale)

        reference = preprocess(x)
        estimator = NearestNeighbors(n_neighbors=21, metric="cosine", n_jobs=4).fit(reference)
        val_features = preprocess(val["embedding"])
        test_features = preprocess(test["embedding"])
        val_distance, val_index = estimator.kneighbors(val_features)
        test_distance, test_index = estimator.kneighbors(test_features)
        case_result = {}
        for k in (5, 21):
            vp = bank["labels"][val_index[:, :k]].mean(axis=1)
            tp = bank["labels"][test_index[:, :k]].mean(axis=1)
            found_fake = tp > .5
            case_result[str(k)] = {"val": metrics(val["labels"], vp, .5),
                                   "test_fake_rate": float(found_fake.mean()),
                                   "test_soft_vote": float(tp.mean()),
                                   "agreement_old": float((found_fake == submission).mean()),
                                   "old_fake_confirmed": int(np.sum(found_fake & submission)),
                                   "extra_fake_vs_old": int(np.sum(found_fake & ~submission))}
            y_votes[(case, k)] = tp
        case_result["median_neighbour_distance"] = {
            "val": float(np.median(val_distance[:, 0])), "test": float(np.median(test_distance[:, 0])),
        }
        results[case] = case_result
    if report_dir is not None:
        with (report_dir / "knn_test_votes.csv").open("x", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image_id", "old_label", *(f"{case}_k{k}" for case, k in y_votes)])
            for i, image_id in enumerate(test["ids"]):
                writer.writerow([image_id, int(submission[i]), *(f"{votes[i]:.6f}" for votes in y_votes.values())])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--extra-output", type=Path,
                        help="Directory containing additional model score files")
    parser.add_argument("--report-dir", type=Path,
                        help="Fresh output directory for the JSON and kNN vote CSV")
    args = parser.parse_args()
    old = ROOT / "fqm/submissions/submission_cap_jpegaug_q75acc0.9150.csv"
    with old.open(newline="") as file:
        old_rows = list(csv.DictReader(file))
    with (ROOT / "ai-generated-image-detection-benchmark/sample_submission.csv").open(newline="") as file:
        template = list(csv.DictReader(file))
    ids = np.array([row["image_id"] for row in template])
    submission = np.array([int(row["label"]) for row in old_rows], dtype=bool)
    assert np.array_equal(ids, [row["image_id"] for row in old_rows]) and len(set(ids)) == 1000
    rng = np.random.default_rng(20261121)
    result = {"models": {}}
    predictions = {}
    calibration = {}
    holdout_scores = {}
    for name in args.models:
        score_dir = args.output
        if args.extra_output and not (score_dir / f"{name}__val_orig.npz").is_file():
            score_dir = args.extra_output
        cases = {case: np.load(score_dir / f"{name}__val_{case}.npz") for case in CASES}
        y = cases["orig"]["labels"]
        val_ids = cases["orig"]["ids"]
        fit_half = np.zeros(len(y), dtype=bool)
        for label in (0, 1):
            indices = np.flatnonzero(y == label)
            order = sorted(indices, key=lambda i: hashlib.sha256(f"audit:{val_ids[i]}".encode()).digest())
            fit_half[order[:len(order) // 2]] = True
        hold = ~fit_half
        val = {case: item["margin"] for case, item in cases.items()}
        proxy = np.stack([val[case] for case in PROXIES])
        thresholds = np.linspace(-2, 2, 81)
        def ordered_score(candidate):
            values = np.mean((proxy[:, fit_half] > candidate) == y[fit_half], axis=1)
            return (float(values.min()), float(values.mean()), -abs(candidate))

        threshold = float(max(thresholds, key=ordered_score))
        if ordered_score(threshold)[0] < ordered_score(0.)[0] + .005:
            threshold = 0.
        calibration[name] = ordered_score(threshold)
        for case, item in cases.items():
            assert np.array_equal(item["ids"], val_ids) and np.array_equal(item["labels"], y), case
        test = np.load(score_dir / f"{name}__test_orig.npz")
        assert np.array_equal(test["ids"], ids)
        scores = test["margin"]
        predictions[name] = scores
        holdout_scores[name] = (y[hold], {case: margin[hold] for case, margin in val.items()})
        result["models"][name] = {
            "full": {case: metrics(y, margin) for case, margin in val.items()},
            "holdout_default": {case: metrics(y[hold], margin[hold]) for case, margin in val.items()},
            "holdout_calibrated": {case: metrics(y[hold], margin[hold], threshold) for case, margin in val.items()},
            "threshold_margin": threshold, "calibration_worst_mean": calibration[name][:2],
            "test": describe(scores),
        }
    winner = max(calibration, key=calibration.get)
    threshold = result["models"][winner]["threshold_margin"]
    new = predictions[winner] > threshold
    result["candidate"] = {"checkpoint": winner, "threshold_margin": threshold,
                           "threshold_prob": float(expit(threshold)),
                           "fake_count": int(new.sum()), "diff_old": int(np.sum(new != submission))}
    holdout_labels, holdout_proxy = holdout_scores[winner]
    result["prior_matching"] = {
        case: evaluate_prior_matching(holdout_labels, holdout_proxy[case], threshold, rng)
        for case in PROXIES
    }
    feature_model = "w6_cap60_jpegaug"
    test_margin = predictions[feature_model]
    score = test_margin.astype(float).reshape(-1, 1)
    result["gmm"] = {}
    for n in (1, 2, 3):
        model = GaussianMixture(n_components=n, random_state=42, n_init=5).fit(score)
        order = np.argsort(model.means_.ravel())
        result["gmm"][str(n)] = {"bic": float(model.bic(score)),
                                "weights": model.weights_[order].tolist(),
                                "logit_means": model.means_.ravel()[order].tolist()}
    result["mixtures"] = {}
    for case in CASES:
        val = np.load(args.output / f"{feature_model}__val_{case}.npz")
        result["mixtures"][case] = score_fit(val["margin"], test_margin, val["labels"], rng)
    report_dir = args.report_dir or args.output
    if args.report_dir:
        report_dir.mkdir(parents=True, exist_ok=False)
    result["knn"] = knn(args.output, feature_model, submission, report_dir=report_dir)
    with (report_dir / "analysis.json").open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    with args.submission.open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "label"])
        writer.writerows(zip(ids, new.astype(int)))
    print(json.dumps(result["candidate"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
