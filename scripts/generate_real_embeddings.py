import argparse
import hashlib
import json
import os

from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from datasets import load_dataset
from fastembed import TextEmbedding


SOURCE = "fancyzhx/ag_news"
MODEL = "BAAI/bge-small-en-v1.5"

BASE_COUNT = 100000
QUERY_COUNT = 200
DIMENSIONS = 384
SEED = 42

CHUNK_SIZE = 256
BATCH_SIZE = 64
THREADS = 4

OUTPUT = Path("datasets/agnews_100k_d384.npz")
CHECKPOINT = Path("datasets/agnews_100k_d384_checkpoint")

VECTOR_FILE = CHECKPOINT / "embeddings.npy"
PROGRESS_FILE = CHECKPOINT / "progress.json"

TOTAL = BASE_COUNT + QUERY_COUNT


def text_sha256(texts):
    digest = hashlib.sha256()

    for text in texts:
        raw = text.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)

    return digest.hexdigest()


def save_progress(config, next_row):
    state = {
        "config": config,
        "next_row": next_row,
    }

    temporary = CHECKPOINT / "progress.json.tmp"

    temporary.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )

    os.replace(temporary, PROGRESS_FILE)


def open_checkpoint(config):
    CHECKPOINT.mkdir(parents=True, exist_ok=True)

    has_vectors = VECTOR_FILE.exists()
    has_progress = PROGRESS_FILE.exists()

    if has_vectors != has_progress:
        raise RuntimeError(
            "Incomplete checkpoint files. "
            "Inspect them before continuing."
        )

    if has_progress:
        state = json.loads(
            PROGRESS_FILE.read_text(encoding="utf-8")
        )

        if state["config"] != config:
            raise RuntimeError(
                "Checkpoint configuration mismatch"
            )

        next_row = int(state["next_row"])

        if not 0 <= next_row <= TOTAL:
            raise RuntimeError(
                "Invalid checkpoint position"
            )

        vectors = np.lib.format.open_memmap(
            VECTOR_FILE,
            mode="r+",
        )

        if vectors.shape != (TOTAL, DIMENSIONS):
            raise RuntimeError(
                "Incorrect checkpoint shape"
            )

        if vectors.dtype != np.float32:
            raise RuntimeError(
                "Incorrect checkpoint dtype"
            )

        print(
            f"Resuming from {next_row}/{TOTAL}",
            flush=True,
        )

        return vectors, next_row

    vectors = np.lib.format.open_memmap(
        VECTOR_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(TOTAL, DIMENSIONS),
    )

    vectors.flush()
    save_progress(config, 0)

    print("New checkpoint created", flush=True)

    return vectors, 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="0 means generate all remaining batches",
    )

    args = parser.parse_args()

    if args.max_batches < 0:
        parser.error(
            "--max-batches must be >= 0"
        )

    if OUTPUT.exists():
        raise SystemExit(
            f"Final dataset already exists: {OUTPUT}"
        )

    print(
        "=== REAL EMBEDDINGS GENERATOR ===",
        flush=True,
    )
    print(f"Dataset: {SOURCE}")
    print(f"Model: {MODEL}")
    print(f"Base vectors: {BASE_COUNT}")
    print(f"Queries: {QUERY_COUNT}")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Checkpoint size: {CHUNK_SIZE}")

    print("\nLoading AG News...", flush=True)

    train = load_dataset(
        SOURCE,
        split=f"train[:{BASE_COUNT}]",
    )

    test = load_dataset(
        SOURCE,
        split=f"test[:{QUERY_COUNT}]",
    )

    train_texts = list(train["text"])
    query_texts = list(test["text"])

    labels = np.asarray(
        train["label"],
        dtype=np.int16,
    )

    query_labels = np.asarray(
        test["label"],
        dtype=np.int16,
    )

    if len(train_texts) != BASE_COUNT:
        raise RuntimeError(
            "Incorrect train count"
        )

    if len(query_texts) != QUERY_COUNT:
        raise RuntimeError(
            "Incorrect query count"
        )

    config = {
        "source": SOURCE,
        "model": MODEL,
        "base_count": BASE_COUNT,
        "query_count": QUERY_COUNT,
        "dimensions": DIMENSIONS,
        "seed": SEED,
        "chunk_size": CHUNK_SIZE,
        "batch_size": BATCH_SIZE,
        "threads": THREADS,
        "fastembed_version": version("fastembed"),
        "datasets_version": version("datasets"),
        "numpy_version": np.__version__,
        "train_sha256": text_sha256(train_texts),
        "query_sha256": text_sha256(query_texts),
    }

    all_texts = train_texts + query_texts

    vectors, next_row = open_checkpoint(config)

    if next_row < TOTAL:
        print("\nLoading embedding model...", flush=True)

        model = TextEmbedding(
            model_name=MODEL,
            threads=THREADS,
        )

        started = perf_counter()
        batches_done = 0

        for start in range(
            next_row,
            TOTAL,
            CHUNK_SIZE,
        ):
            end = min(
                start + CHUNK_SIZE,
                TOTAL,
            )

            embeddings = np.asarray(
                list(
                    model.embed(
                        all_texts[start:end],
                        batch_size=BATCH_SIZE,
                    )
                ),
                dtype=np.float32,
            )

            expected_shape = (
                end - start,
                DIMENSIONS,
            )

            if embeddings.shape != expected_shape:
                raise RuntimeError(
                    f"Wrong shape at row {start}: "
                    f"{embeddings.shape}"
                )

            if not np.isfinite(embeddings).all():
                raise RuntimeError(
                    f"Non-finite embedding at row {start}"
                )

            vectors[start:end] = embeddings

            # Save data before recording progress.
            vectors.flush()
            save_progress(config, end)

            batches_done += 1

            print(
                f"Checkpoint saved: {end}/{TOTAL}",
                flush=True,
            )

            if (
                args.max_batches > 0
                and batches_done >= args.max_batches
            ):
                break

        elapsed = perf_counter() - started

        print(
            f"Session elapsed: {elapsed:.2f} seconds",
            flush=True,
        )

    state = json.loads(
        PROGRESS_FILE.read_text(encoding="utf-8")
    )

    completed = int(state["next_row"])

    if completed < TOTAL:
        print("\n=== PARTIAL GENERATION ===")
        print(
            f"Embeddings generated: {completed}/{TOTAL}"
        )
        print(f"Checkpoint: {CHECKPOINT}")
        print(
            "Run the script again to continue."
        )
        print("CHECKPOINT TEST PASSED")
        return

    print(
        "\nPreparing final dataset...",
        flush=True,
    )

    ids = np.asarray([
        f"v{i:07d}"
        for i in range(BASE_COUNT)
    ])

    rng = np.random.default_rng(SEED)

    buckets = np.repeat(
        np.arange(100, dtype=np.int16),
        BASE_COUNT // 100,
    )

    rng.shuffle(buckets)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = OUTPUT.with_suffix(".npz.tmp")

    with temporary.open("wb") as file:
        np.savez_compressed(
            file,
            vectors=np.asarray(
                vectors[:BASE_COUNT]
            ),
            queries=np.asarray(
                vectors[BASE_COUNT:]
            ),
            ids=ids,
            buckets=buckets,
            labels=labels,
            query_labels=query_labels,
            seed=np.int64(SEED),
            source_name=np.array(SOURCE),
            model_name=np.array(MODEL),
            metadata_json=np.array(
                json.dumps(config)
            ),
        )

    os.replace(temporary, OUTPUT)

    print("\n=== FINAL DATASET ===")
    print(f"Output: {OUTPUT}")
    print(
        f"Vectors: ({BASE_COUNT}, {DIMENSIONS})"
    )
    print(
        f"Queries: ({QUERY_COUNT}, {DIMENSIONS})"
    )
    print(
        f"Size: "
        f"{OUTPUT.stat().st_size / 1024**2:.1f} MiB"
    )
    print("REAL EMBEDDINGS GENERATION PASSED")


if __name__ == "__main__":
    main()
