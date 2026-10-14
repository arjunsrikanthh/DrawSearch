"""Download and prepare a small, reproducible sample of Quick, Draw! data."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import yaml
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def split_source_indices(
    source_count: int,
    sample_count: int,
    split_counts: dict[str, int],
    seed: int,
) -> dict[str, np.ndarray]:
    """Choose distinct source rows and divide them into fixed split sizes."""
    if sum(split_counts.values()) != sample_count:
        raise ValueError("The split sizes must add up to the sample count.")
    if source_count < sample_count:
        raise ValueError(
            f"Need {sample_count:,} drawings, but the source has only {source_count:,}."
        )

    rng = np.random.default_rng(seed)
    chosen_rows = rng.choice(source_count, size=sample_count, replace=False)
    chosen_rows = rng.permutation(chosen_rows)

    result: dict[str, np.ndarray] = {}
    start = 0
    for split_name, count in split_counts.items():
        result[split_name] = chosen_rows[start : start + count]
        start += count

    all_rows = np.concatenate(list(result.values()))
    if len(np.unique(all_rows)) != sample_count:
        raise AssertionError("A source drawing was assigned more than once.")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> dict[str, Any]:
    """Download one source file safely, leaving no partial file at its final path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            digest = hashlib.sha256()
            downloaded = 0
            next_update = 0
            request = Request(url, headers={"User-Agent": "DrawSearch student project"})
            with urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                total = int(response.headers.get("Content-Length", "0"))
                etag = response.headers.get("ETag")
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if total and downloaded >= next_update:
                        percent = min(100, int(downloaded * 100 / total))
                        print(f"    {percent:3d}% ({downloaded / 1024**2:.0f} MiB)", flush=True)
                        next_update = downloaded + max(total // 10, 32 * 1024**2)
                output.flush()
                os.fsync(output.fileno())

            if total and downloaded != total:
                raise IOError(f"Download ended at {downloaded:,} bytes; expected {total:,}.")
            os.replace(temporary, destination)
            return {
                "sha256": digest.hexdigest(),
                "bytes": downloaded,
                "etag": etag,
            }
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                wait_seconds = attempt * 2
                print(f"    Download attempt {attempt} failed; retrying in {wait_seconds}s: {error}")
                time.sleep(wait_seconds)

    raise RuntimeError(f"Could not download {url}: {last_error}") from last_error


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        json.dump(value, temporary, indent=2)
        temporary.write("\n")
        temp_path = Path(temporary.name)
    os.replace(temp_path, path)


def _save_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        temp_path = Path(temporary.name)
        np.savez_compressed(temporary, **arrays)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temp_path, path)


def _seed_for_category(seed: int, category: str) -> int:
    """Give each category a stable seed, independent of category list position."""
    token = f"{seed}:{category}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def _create_sample_grid(
    preview_images: dict[str, np.ndarray],
    destination: Path,
    image_shape: tuple[int, int],
) -> None:
    """Save four unmodified source drawings for every class for visual inspection."""
    cell_size = 116
    label_width = 112
    columns = 4
    categories = list(preview_images)
    sheet = Image.new("RGB", (label_width + columns * cell_size, len(categories) * cell_size), "white")
    draw = ImageDraw.Draw(sheet)

    for row, category in enumerate(categories):
        top = row * cell_size
        draw.text((6, top + 48), category, fill="black")
        for column, pixels in enumerate(preview_images[category][:columns]):
            image = Image.fromarray(pixels.reshape(image_shape))
            image = image.resize((84, 84), Image.Resampling.NEAREST).convert("RGB")
            left = label_width + column * cell_size + 16
            sheet.paste(image, (left, top + 16))

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def _count_cross_split_duplicates(splits: dict[str, dict[str, np.ndarray]]) -> dict[str, int]:
    """Count identical pixel images that would appear in more than one split."""
    image_splits: dict[bytes, set[str]] = {}
    for split_name, data in splits.items():
        for pixels in data["images"]:
            image_hash = hashlib.blake2b(pixels, digest_size=16).digest()
            image_splits.setdefault(image_hash, set()).add(split_name)

    pair_counts = {"train-val": 0, "train-test": 0, "val-test": 0}
    for assigned_splits in image_splits.values():
        if "train" in assigned_splits and "val" in assigned_splits:
            pair_counts["train-val"] += 1
        if "train" in assigned_splits and "test" in assigned_splits:
            pair_counts["train-test"] += 1
        if "val" in assigned_splits and "test" in assigned_splits:
            pair_counts["val-test"] += 1
    return pair_counts


def prepare_dataset(
    config_path: Path,
    *,
    force: bool = False,
    root: Path = ROOT,
) -> tuple[Path, Path]:
    """Download the configured classes, save fixed splits, and write audit files."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    categories = config["categories"]
    split_counts = config["split_counts_per_class"]
    sample_count = int(config["samples_per_class"])
    seed = int(config["seed"])
    image_shape = tuple(int(value) for value in config["image_shape"])

    if len(categories) != len(set(categories)):
        raise ValueError("Category names must not repeat.")
    if sample_count != sum(split_counts.values()):
        raise ValueError("Split counts must add up to samples_per_class in configs/data.yaml.")

    raw_dir = root / config["raw_dir"]
    processed_dir = root / config["processed_dir"]
    archive_paths = {name: processed_dir / f"{name}.npz" for name in split_counts}
    manifest_path = processed_dir / "manifest.json"
    inspection_path = processed_dir / "inspection.json"
    if not force and any(path.exists() for path in archive_paths.values()):
        raise FileExistsError(
            f"Prepared data already exists under {processed_dir}. Use --force to rebuild it."
        )

    split_parts: dict[str, dict[str, list[np.ndarray]]] = {
        split: {"images": [], "labels": [], "source_indices": []} for split in split_counts
    }
    preview_images: dict[str, np.ndarray] = {}
    source_records: dict[str, Any] = {}

    for label_id, category in enumerate(categories):
        source_path = raw_dir / f"{category}.npy"
        source_url = f"{config['source_base_url'].rstrip('/')}/{category}.npy"
        print(f"[{label_id + 1}/{len(categories)}] {category}", flush=True)

        if source_path.exists():
            print(f"    Using existing source file: {source_path}")
            source_hash = _sha256_file(source_path)
            download_record = {"sha256": source_hash, "bytes": source_path.stat().st_size, "etag": None}
        else:
            print(f"    Downloading {source_url}", flush=True)
            download_record = _download_file(source_url, source_path)

        try:
            source_images = np.load(source_path, mmap_mode="r", allow_pickle=False)
        except Exception as error:
            raise RuntimeError(
                f"Could not read {source_path}. Remove that damaged file and rerun."
            ) from error
        if source_images.ndim != 2 or source_images.shape[1] != int(np.prod(image_shape)):
            raise ValueError(f"Unexpected array shape for {category}: {source_images.shape}")
        if source_images.dtype != np.uint8:
            raise ValueError(f"Expected uint8 pixels for {category}; got {source_images.dtype}.")

        indices = split_source_indices(
            source_count=source_images.shape[0],
            sample_count=sample_count,
            split_counts=split_counts,
            seed=_seed_for_category(seed, category),
        )
        preview_images[category] = np.asarray(source_images[indices["train"][:4]], dtype=np.uint8)
        source_records[category] = {
            "url": source_url,
            "sha256": download_record["sha256"],
            "bytes": download_record["bytes"],
            "etag": download_record["etag"],
            "source_row_count": int(source_images.shape[0]),
            "label_id": label_id,
            "selected_source_rows": {},
        }

        for split_name in split_counts:
            row_indices = indices[split_name]
            source_records[category]["selected_source_rows"][split_name] = row_indices.tolist()
            split_parts[split_name]["images"].append(np.asarray(source_images[row_indices], dtype=np.uint8))
            split_parts[split_name]["labels"].append(np.full(len(row_indices), label_id, dtype=np.int64))
            split_parts[split_name]["source_indices"].append(row_indices.astype(np.int64, copy=False))

    prepared_splits: dict[str, dict[str, np.ndarray]] = {}
    archive_records: dict[str, Any] = {}
    for shuffle_number, split_name in enumerate(split_counts, start=1):
        arrays = {
            key: np.concatenate(parts)
            for key, parts in split_parts[split_name].items()
        }
        order = np.random.default_rng(seed + shuffle_number).permutation(len(arrays["labels"]))
        arrays = {key: values[order] for key, values in arrays.items()}
        _save_archive(archive_paths[split_name], arrays)
        archive_records[split_name] = {
            "file": str(archive_paths[split_name].relative_to(root)),
            "sha256": _sha256_file(archive_paths[split_name]),
            "drawings": len(arrays["labels"]),
            "fields": {
                "images": list(arrays["images"].shape),
                "labels": list(arrays["labels"].shape),
                "source_indices": list(arrays["source_indices"].shape),
            },
        }
        prepared_splits[split_name] = arrays
        print(f"    Saved {archive_records[split_name]['drawings']:,} {split_name} drawings.")

    all_pixel_values = np.unique(
        np.concatenate([data["images"].ravel() for data in prepared_splits.values()])
    )
    duplicate_counts = _count_cross_split_duplicates(prepared_splits)
    counts_by_split = {
        split_name: {
            category: int(np.count_nonzero(data["labels"] == label_id))
            for label_id, category in enumerate(categories)
        }
        for split_name, data in prepared_splits.items()
    }
    inspection = {
        "image_shape": list(image_shape),
        "pixel_dtype": str(prepared_splits["train"]["images"].dtype),
        "pixel_min": min(int(data["images"].min()) for data in prepared_splits.values()),
        "pixel_max": max(int(data["images"].max()) for data in prepared_splits.values()),
        "pixel_values_found": all_pixel_values.tolist(),
        "fraction_of_pixels_greater_than_zero": {
            split_name: float(np.mean(data["images"] > 0))
            for split_name, data in prepared_splits.items()
        },
        "fraction_of_pixels_less_than_255": {
            split_name: float(np.mean(data["images"] < 255))
            for split_name, data in prepared_splits.items()
        },
        "drawings_per_class": counts_by_split,
        "exact_images_appearing_in_multiple_splits": duplicate_counts,
        "sample_grid": "outputs/figures/quickdraw_samples.png",
        "polarity_note": "Pixel values are kept unchanged. Inspect the sample grid before preprocessing for the drawing app.",
    }
    _write_json(inspection_path, inspection)
    _create_sample_grid(preview_images, root / inspection["sample_grid"], image_shape)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config["dataset"],
        "source": "Google Quick, Draw! numpy_bitmap files (CC BY 4.0)",
        "seed": seed,
        "class_order": categories,
        "label_ids": {str(index): name for index, name in enumerate(categories)},
        "image_shape": list(image_shape),
        "pixel_dtype": "uint8",
        "samples_per_class": sample_count,
        "split_counts_per_class": split_counts,
        "split_archives": archive_records,
        "source_files": source_records,
        "inspection_report": str(inspection_path.relative_to(root)),
    }
    _write_json(manifest_path, manifest)
    print(f"\nFinished. Dataset record: {manifest_path}")
    print(f"Inspection details: {inspection_path}")
    print(f"Sample grid: {root / inspection['sample_grid']}")
    if any(duplicate_counts.values()):
        print("WARNING: identical images occur in multiple splits; review inspection.json before training.")
    return manifest_path, inspection_path
