"""Inference entry point for Project 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import pruned_sft


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LEGACY_ADAPTER_PATH = (
    ROOT
    / "rapidfire_experiments"
    / "project2_qwen25_coder15b_sft_pruned_dbbal12_lr8p5e5_r16a32_ep7_retry_detached_len8192"
    / "runs"
    / "1"
    / "checkpoints"
    / "final_checkpoint"
)
DEFAULT_ADAPTER_PATH = ROOT / "adapter"


def buildParser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--schemas_dir", default="./schemas")

    ap.add_argument("--adapter_path", default=str(DEFAULT_ADAPTER_PATH))
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--train_json", default="train.json")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")

    # Best validation pruning setup.
    ap.add_argument("--top_tables", type=int, default=14)
    ap.add_argument("--max_schema_columns", type=int, default=320)
    ap.add_argument("--max_columns_per_table", type=int, default=48)
    ap.add_argument("--max_input_tokens", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=384)

    # Local validation only; the hidden test runner will not pass this.
    ap.add_argument("--gold", default=None)
    return ap


def main() -> None:
    args = buildParser().parse_args()
    adapter_path = Path(args.adapter_path)
    using_default_adapter = str(adapter_path) == str(DEFAULT_ADAPTER_PATH)
    if using_default_adapter and not adapter_path.exists() and LEGACY_ADAPTER_PATH.exists():
        adapter_path = LEGACY_ADAPTER_PATH
        args.adapter_path = str(adapter_path)
    if not adapter_path.exists():
        raise SystemExit(
            "Adapter checkpoint not found at "
            f"{adapter_path}. Put the PEFT adapter in ./adapter or use --adapter_path."
        )

    # Keep inference on the pruned-schema path used by the best run.
    args.include_all_selected_table_columns = False
    pruned_sft.predict(args)


if __name__ == "__main__":
    main()
