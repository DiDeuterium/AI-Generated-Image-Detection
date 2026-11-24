"""Choose one checkpoint on labeled data and compare its standalone results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import partial
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from contrastive_texture_base import (
    CLASS_CONTRASTIVE_MODEL, CROSS_CLASS_CONTRASTIVE_MODEL, MODEL as CONTRASTIVE_MODEL,
    MOMENTUM_CONTRASTIVE_MODEL, ContrastiveTexture,
)
from degradation_anchor_texture import MODEL as DEGRADATION_MODEL, DegradationAnchorTexture
from detection.training import load_checkpoint
from lqy.codec_anchor import prepare_data
from lqy.codec_canonical import calibration_rules, transform, validation_halves
from lqy.codec_final import profile_gain
from lqy_compound_blend import COMPOUND, REPLAY, blend
from lqy_replay import PROFILES, metrics, read_json, write_csv
from lqy_resize_retrain import TEMPLATE, TEST_DIR, identity, prepare_views, read_test_ids, training_split
from lqy_specialist_gate import SCREEN
from multiscale_texture_base import MODEL as MULTISCALE_MODEL, MultiscaleTexture, native_scales
from native_texture_base import (
    DEEP_DEPTHS, DEEP_MODEL, MODEL, RANKED_MODEL, RESAMPLING_MODEL, SEED, WEIGHTED_MODEL, WIDE_CHANNELS, WIDE_MODEL,
    NativeTextureBase, ResamplingTexture, acquisition,
)
from patch_evidence_texture import (
    ADAPTIVE_CONTEXT_MODEL, ANCHOR_CONTEXT_MODEL, EVIDENCE_MODEL, GRID_MODEL, MULTIBAND_ANCHOR_MODEL,
    AdaptiveContextTexture, AnchorContextTexture, GridTexture, MultibandAnchorTexture,
    PatchEvidenceTexture, anchor_grid_input, grid_input,
)
from self_supervised_texture_base import (
    AUTOCORRELATION_MODEL, DIRECTIONAL_MODEL, DIRECTIONAL_TARGET_COUNT, MODEL as SELF_SUPERVISED_MODEL,
    PHASE_STEM_MODEL, PhaseStemTexture, SelfSupervisedTexture,
)
from two_stage_texture_base import FINETUNED_MODEL, FROZEN_MODEL

TRAINED = Path("fqm/runs/native_texture_base_20261122")
EPOCHS = (6, 12, 18, 24)
PATCH_MODELS = (GRID_MODEL, EVIDENCE_MODEL, ANCHOR_CONTEXT_MODEL, MULTIBAND_ANCHOR_MODEL,
                ADAPTIVE_CONTEXT_MODEL, DEGRADATION_MODEL)
EXTRA_PROBE = "resize_webp"


def calibrate_image(image, image_id):
    assignment = sha256(f"native-texture-anchor:{image_id}".encode()).digest()
    return acquisition(image, assignment)


def stress(profile, image, _image_id):
    return transform(image, profile)


def convert(samples):
    return [{"path": sample.path, "image_id": sample.image_id} for sample in samples]


def single_input(row, table, processing):
    first, second = prepare_views(row, table, processing)
    return (np.concatenate((first, second), axis=1),)


def multiscale_input(row, table, processing):
    with Image.open(row["path"]) as source:
        image = source.convert("RGB")
    return native_scales(processing(image, row["image_id"]), table)


@torch.inference_mode()
def score_many(models, rows, table, processing, device, *, render_row=single_input):
    prepare = partial(render_row, table=table, processing=processing)
    output = {epoch: [] for epoch in models}
    with ThreadPoolExecutor(max_workers=4) as threads:
        for offset in range(0, len(rows), 24):
            rendered = threads.map(prepare, rows[offset:offset + 24])
            inputs = [torch.from_numpy(np.stack(parts)).to(device) for parts in zip(*rendered, strict=True)]
            for epoch, model in models.items():
                score = model(*inputs).float().softmax(dim=1)[:, 1].cpu().numpy()
                output[epoch].append(score)
    return {epoch: np.concatenate(batches) for epoch, batches in output.items()}


def renderer_for(variant):
    return {MULTISCALE_MODEL: multiscale_input, GRID_MODEL: grid_input, EVIDENCE_MODEL: grid_input,
            ANCHOR_CONTEXT_MODEL: anchor_grid_input, MULTIBAND_ANCHOR_MODEL: anchor_grid_input,
            ADAPTIVE_CONTEXT_MODEL: anchor_grid_input, DEGRADATION_MODEL: anchor_grid_input}.get(variant, single_input)


def restore(directory, epoch, device, *, variant=MODEL):
    model_class = {MODEL: NativeTextureBase, MULTISCALE_MODEL: MultiscaleTexture,
                   RESAMPLING_MODEL: ResamplingTexture, WEIGHTED_MODEL: NativeTextureBase,
                   RANKED_MODEL: NativeTextureBase,
                   SELF_SUPERVISED_MODEL: SelfSupervisedTexture,
                   AUTOCORRELATION_MODEL: SelfSupervisedTexture,
                   DIRECTIONAL_MODEL: partial(SelfSupervisedTexture, target_count=DIRECTIONAL_TARGET_COUNT),
                   PHASE_STEM_MODEL: PhaseStemTexture,
                   FROZEN_MODEL: NativeTextureBase, FINETUNED_MODEL: NativeTextureBase,
                   CONTRASTIVE_MODEL: ContrastiveTexture,
                   CLASS_CONTRASTIVE_MODEL: ContrastiveTexture,
                   CROSS_CLASS_CONTRASTIVE_MODEL: ContrastiveTexture,
                   MOMENTUM_CONTRASTIVE_MODEL: ContrastiveTexture,
                   GRID_MODEL: GridTexture, EVIDENCE_MODEL: PatchEvidenceTexture,
                   ANCHOR_CONTEXT_MODEL: AnchorContextTexture,
                   MULTIBAND_ANCHOR_MODEL: MultibandAnchorTexture,
                   ADAPTIVE_CONTEXT_MODEL: AdaptiveContextTexture,
                   DEGRADATION_MODEL: DegradationAnchorTexture,
                   WIDE_MODEL: partial(NativeTextureBase, channels=WIDE_CHANNELS),
                   DEEP_MODEL: partial(NativeTextureBase, depths=DEEP_DEPTHS)}[variant]
    checkpoint = load_checkpoint(directory / f"epoch-{epoch}.pt")
    if (checkpoint["variant"], checkpoint["seed"], checkpoint["epoch"]) != (variant, SEED, epoch):
        raise ValueError("Unexpected training checkpoint")
    model = model_class().to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["weights"])
    return model.eval()


def decision_scores(probabilities, threshold):
    from lqy.codec_anchor import recenter
    return recenter(probabilities, threshold)


def reference_scores(ids, truth, evaluation, profiles=PROFILES):
    if evaluation is not None:
        with np.load(evaluation / "validation_scores.npz") as previous:
            np.testing.assert_array_equal(previous["ids"], ids)
            np.testing.assert_array_equal(previous["labels"], truth)
            return {profile: previous[f"{profile}_single"] for profile in profiles}
    with np.load(REPLAY / "validation_scores.npz") as ref, np.load(COMPOUND / "validation_scores.npz") as compound:
        for stored in (ref, compound):
            np.testing.assert_array_equal(stored["ids"], ids)
            np.testing.assert_array_equal(stored["labels"], truth)
        return {profile: blend(ref[f"{profile}_base"], ref[f"{profile}_replacement"], compound[profile])[1]
                for profile in profiles}


def matched_real_recall(truth, baseline, candidate):
    real = truth == 0
    target_count = int(np.count_nonzero(baseline[real] <= .5))

    def recalls(scores):
        # Keep tied real scores together and accept at least as many real images as the baseline.
        cutoff = np.sort(scores[real])[target_count - 1] if target_count else -np.inf
        return {"recall_real": float(np.mean(scores[real] <= cutoff)),
                "recall_fake": float(np.mean(scores[~real] > cutoff))}

    return {"target_recall_real": target_count / int(real.sum()),
            "baseline": recalls(baseline), "single_model": recalls(candidate)}


def size_cohort_results(truth, short_sides, models):
    groups = {"small_real": (truth == 0) & (short_sides < 160)}
    for side in np.unique(short_sides[truth == 1]):
        groups[f"generated_{side}"] = (truth == 1) & (short_sides == side)
    report = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        correct = {key: int(((scores[mask] > .5) == truth[mask]).sum()) for key, scores in models.items()}
        count = int(mask.sum())
        report[name] = {"count": count, "correct": correct,
                        "recall": {key: hits / count for key, hits in correct.items()}}
    return report


def export_candidate(path, score, model, epoch, threshold, table, device):
    rows = read_test_ids(TEMPLATE, TEST_DIR)
    ids = [row["image_id"] for row in rows]
    with TEMPLATE.open(newline="") as file:
        template_ids = [row["image_id"] for row in csv.DictReader(file)]
    if len(ids) != 1000 or len(set(ids)) != 1000 or ids != template_ids:
        raise ValueError("Test IDs must match the 1,000 unique template IDs in order")

    raw = score({epoch: model}, rows, table, identity, device)[epoch]
    if not np.isfinite(raw).all():
        raise ValueError("Invalid test probabilities")
    decisions = decision_scores(raw, threshold)
    write_csv(path, ids, decisions)
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    expected = [(image_id, str(int(fake))) for image_id, fake in zip(ids, decisions > .5, strict=True)]
    if reader.fieldnames != ["image_id", "label"] or [(row["image_id"], row["label"]) for row in rows] != expected:
        raise ValueError("Candidate CSV failed template or binary-label verification")
    print(f"Verified 1,000 template rows: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trained", default=TRAINED, type=Path)
    parser.add_argument("--training-job", default=7102, type=int)
    parser.add_argument("--reference-evaluation", type=Path)
    parser.add_argument("--comparison-evaluation", type=Path)
    parser.add_argument("--extra-probe-reference", type=Path, help="frozen training run for the resize-WebP probe")
    parser.add_argument("--baseline-online", default=.646, type=float)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--checkpoint-epochs", type=int, nargs="+", default=EPOCHS)
    args = parser.parse_args()
    if args.extra_probe_reference is not None and args.reference_evaluation is None:
        parser.error("--extra-probe-reference requires --reference-evaluation")
    if args.candidate is not None and args.candidate.exists():
        raise FileExistsError(args.candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    samples, labels, sizes, _, _, full, table, split = prepare_data()
    completed = read_json(args.trained / "summary.json")
    if completed["split_sha256"] != split or completed["stage"] != "training_complete":
        raise ValueError("Expected a complete single-model training run on the same split")
    variant = completed["variant"]
    table = completed.get("codec_table", table)
    epochs = args.checkpoint_epochs
    models = {epoch: restore(args.trained, epoch, device, variant=variant) for epoch in epochs}
    score = partial(score_many, render_row=renderer_for(variant))
    _, anchors = training_split([samples[i] for i in full.tolist()])
    anchor_labels = np.asarray([row.label for row in anchors])
    calibration = score(models, convert(anchors), table, calibrate_image, device)
    thresholds = {epoch: calibration_rules(values, anchor_labels)["both_class"]
                  for epoch, values in calibration.items()}
    selection, heldout = validation_halves(samples, labels, sizes, len(full))
    selection_rows = convert([samples[i] for i in selection])
    selection_labels = labels[selection].numpy()
    fit = {epoch: {} for epoch in epochs}
    for profile in SCREEN:
        print(f"Selecting checkpoint: {profile}", flush=True)
        values = score(models, selection_rows, table, partial(stress, profile), device)
        for epoch, probability in values.items():
            fit[epoch][profile] = metrics(selection_labels, decision_scores(probability, thresholds[epoch]))
    def ranking(epoch):
        accuracy = [scores["accuracy"] for scores in fit[epoch].values()]
        return min(accuracy), float(np.mean(accuracy)), -epoch
    best = max(epochs, key=ranking)
    print(f"Selected epoch {best}, threshold {thresholds[best]:.6f}", flush=True)

    truth = labels[heldout].numpy()
    rows = convert([samples[i] for i in heldout])
    ids = [row["image_id"] for row in rows]
    held, pairs, all_scores = {}, [], {}
    references = reference_scores(ids, truth, args.reference_evaluation)
    comparison = (reference_scores(ids, truth, args.comparison_evaluation)
                  if args.comparison_evaluation is not None else None)
    profiles = PROFILES
    if args.extra_probe_reference is not None:
        profiles = (*PROFILES, EXTRA_PROBE)
        folder = args.extra_probe_reference
        frozen_training = read_json(folder / "summary.json")
        if frozen_training["split_sha256"] != split:
            raise ValueError("Extra-probe reference must use the same labeled split")
        frozen_evaluation = read_json(args.reference_evaluation / "summary.json")
        epoch = frozen_evaluation["selected_epoch"]
        frozen_model = restore(folder, epoch, device, variant=frozen_training["variant"])
        raw = score_many({epoch: frozen_model}, rows, table, partial(stress, EXTRA_PROBE), device,
                         render_row=renderer_for(frozen_training["variant"]))[epoch]
        references[EXTRA_PROBE] = decision_scores(raw, frozen_evaluation["selected_threshold"])
        if comparison is not None:
            comparison.update(reference_scores(ids, truth, args.comparison_evaluation, (EXTRA_PROBE,)))
    for profile in profiles:
        print(f"Evaluating on separate images: {profile}", flush=True)
        probs = score({best: models[best]}, rows, table, partial(stress, profile), device)[best]
        candidate = decision_scores(probs, thresholds[best])
        baseline = references[profile]
        after, before = metrics(truth, candidate), metrics(truth, baseline)
        delta = {key: 100 * (after[key] - before[key]) for key in ("accuracy", "recall_real", "recall_fake")}
        held[profile] = {"single_model": after, "baseline": before, "delta_percentage_points": delta}
        if comparison is not None:
            held[profile]["comparison_model"] = metrics(truth, comparison[profile])
        if profile in ("alternate_resize_jpeg", EXTRA_PROBE):
            matched = matched_real_recall(truth, baseline, candidate)
            if comparison is not None:
                matched["comparison_model"] = matched_real_recall(truth, baseline, comparison[profile])["single_model"]
            held[profile]["matched_real_recall_diagnostic"] = matched
            if variant in PATCH_MODELS:
                short_sides = np.array([min(sizes[i]) for i in heldout])
                breakdown = {"baseline": baseline, "single_model": candidate}
                if comparison is not None:
                    breakdown["comparison_model"] = comparison[profile]
                held[profile]["size_cohorts"] = size_cohort_results(truth, short_sides, breakdown)
        if profile in PROFILES:
            pairs.append((baseline, candidate))
        all_scores[f"{profile}_single"] = candidate
        all_scores[f"{profile}_baseline"] = baseline
        print(f"{profile}: accuracy {after['accuracy']:.2%} ({delta['accuracy']:+.2f} pp), "
              f"real {after['recall_real']:.2%} ({delta['recall_real']:+.2f} pp), "
              f"generated {after['recall_fake']:.2%} ({delta['recall_fake']:+.2f} pp)", flush=True)
    report = {"training_job": args.training_job, "trained": str(args.trained),
              "baseline_online": args.baseline_online, "selection": fit,
              "selected_epoch": best, "selected_threshold": thresholds[best],
              "confirmation": held, "gain": profile_gain(truth, pairs)}
    if args.comparison_evaluation is not None:
        report["comparison_evaluation"] = str(args.comparison_evaluation)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(args.output / "validation_scores.npz", ids=ids, labels=truth, **all_scores)
    print(json.dumps({"selected_epoch": best, "gain": report["gain"]}), flush=True)
    if args.candidate is not None:
        export_candidate(args.candidate, score, models[best], best, thresholds[best], table, device)


if __name__ == "__main__":
    main()
