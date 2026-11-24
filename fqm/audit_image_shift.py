"""Audit image headers, pixel statistics, and cross-split perceptual neighbours."""
import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin
from scipy.fft import dctn
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
LABELS = ROOT / "ai-generated-image-detection-benchmark"


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def signature(path, item):
    with Image.open(path) as image:
        width, height = image.size
        encoding = image.format
        sampling = JpegImagePlugin.get_sampling(image) if encoding == "JPEG" else ""
        tables = getattr(image, "quantization", None)
        signature = hashlib.sha256(repr(tables).encode()).hexdigest()[:16] if tables else ""
        luminance = tables.get(0, []) if tables else []
        luma_sum = sum(luminance) if luminance else ""
        encoder = image.info.get("software", "")
        mode = image.mode
        gray = image.convert("L")
        center = ((width - min(320, width)) // 2, (height - min(320, height)) // 2)
        crop = gray.crop((*center, center[0] + min(320, width), center[1] + min(320, height)))
        a = np.asarray(crop, dtype=np.float32) / 255
        middle = a[1:-1, 1:-1]
        lap = np.abs(4 * middle - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:])
        low = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        coeffs = dctn(low, norm="ortho")[:8, :8].ravel()[1:]
        dh = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    with path.open("rb") as file:
        content_hash = hashlib.file_digest(file, "sha256").hexdigest()
    info = {**item, "width": width, "height": height, "encoding": encoding,
            "suffix": path.suffix, "sampling": sampling, "quant_table": signature,
            "luma_sum": luma_sum, "mode": mode, "software": encoder,
            "laplacian": float(lap.mean()), "gray_std": float(low.std()),
            "sha256": content_hash}
    return info, coeffs > np.median(coeffs), (dh[:, 1:] > dh[:, :-1]).ravel()


def nearest(query, reference, hashes, metadata):
    # Return unique best or tied nearest neighbour for both hash metrics.
    rows = []
    exact_references = {metadata[i]["sha256"] for i in reference}
    for start in range(0, len(query), 64):
        idx = query[start:start + 64]
        p = np.rint(cdist(hashes[0][idx], hashes[0][reference], "hamming") * 63).astype(np.uint8)
        d = np.rint(cdist(hashes[1][idx], hashes[1][reference], "hamming") * 64).astype(np.uint8)
        for offset, source in enumerate(idx):
            p_dist, d_dist = p[offset], d[offset]
            score = p_dist + d_dist
            combined = np.flatnonzero(score == score.min())
            best = combined[np.lexsort((d_dist[combined], p_dist[combined]))[0]]
            match = reference[best]
            rows.append({"image_id": metadata[source]["image_id"], "split": metadata[source]["split"],
                         "label": metadata[source]["label"], "nearest_id": metadata[match]["image_id"],
                         "nearest_split": metadata[match]["split"], "nearest_label": metadata[match]["label"],
                         "p_distance": int(p_dist[best]), "d_distance": int(d_dist[best]),
                         "min_p": int(p_dist.min()), "min_d": int(d_dist.min()),
                         "count_p_le4_d_le6": int(((p_dist <= 4) & (d_dist <= 6)).sum()),
                         "count_p_le8_d_le10": int(((p_dist <= 8) & (d_dist <= 10)).sum()),
                         "exact_file_match": metadata[source]["sha256"] in exact_references,
                         "exact_joint_hash": bool(((p_dist == 0) & (d_dist == 0)).any())})
    return rows


def write_csv(path, rows):
    with path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train_val = read_csv(ROOT / "splits/train_val.csv")
    test = [{"image_id": row["image_id"], "label": "", "split": "test",
             "image_path": f"dataset/test/{row['image_id']}"}
            for row in read_csv(LABELS / "sample_submission.csv")]
    items = train_val + test
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(lambda row: signature(DATASET / row["image_path"], row), items))
    metadata, phash, dhash = zip(*records)
    hashes = (np.asarray(phash), np.asarray(dhash))
    write_csv(args.output / "image_metadata.csv", metadata)
    split = np.array([row["split"] for row in items])
    for name, query, reference in [
        ("test_vs_train_all", np.flatnonzero(split == "test"), np.flatnonzero(split != "test")),
        ("val_vs_train", np.flatnonzero(split == "val"), np.flatnonzero(split == "train")),
    ]:
        write_csv(args.output / f"{name}.csv", nearest(query, reference, hashes, metadata))
        print(f"{name}: {len(query)} x {len(reference)}", flush=True)
    print(f"{len(records)} image records", flush=True)


if __name__ == "__main__":
    main()
