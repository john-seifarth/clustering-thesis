"""Generate normalized embeddings from a one-column, headerless CSV."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("slurm/response_text.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/response_embeddings.npy"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    return parser.parse_args()


def read_responses(path: Path) -> list[str]:
    """Read a headerless CSV containing exactly one text column."""

    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        data = pd.read_csv(
            path,
            header=None,
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=False,
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Input file is empty: {path}") from exc

    if data.shape[1] != 1:
        raise ValueError(
            f"Expected exactly one CSV column, but found {data.shape[1]}. "
            "Responses containing commas must be CSV-quoted."
        )

    return data.iloc[:, 0].tolist()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable. Check the Slurm GPU request "
            "and PyTorch installation."
        )

    responses = read_responses(args.input)

    # Blank rows remain in the data so row alignment is preserved.
    empty_mask = np.asarray(
        [not response.strip() for response in responses],
        dtype=bool,
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
    )

    embeddings = model.encode(
        responses,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Row i corresponds to row i in response_text.csv.
    np.save(args.output, embeddings)

    mask_path = args.output.with_name(
        f"{args.output.stem}_empty_mask.npy"
    )
    np.save(mask_path, empty_mask)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "input_file": str(args.input.resolve()),
        "embedding_file": str(args.output.resolve()),
        "empty_mask_file": str(mask_path.resolve()),
        "rows": len(responses),
        "empty_rows": int(empty_mask.sum()),
        "embedding_dimension": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "normalized": True,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "sentence_transformers_version": (
            sentence_transformers.__version__
        ),
    }

    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved embeddings: {args.output}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved empty-row mask: {mask_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()