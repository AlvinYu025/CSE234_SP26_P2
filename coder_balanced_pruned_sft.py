"""Coder-model variant of the balanced pruned SFT setup."""

from __future__ import annotations

import argparse
from typing import Any

import balanced_pruned_sft
import pruned_sft
import sft


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_RUN = "sft_pruned_coder_dbbal12_lr1e-4_r16_a32_ep6_len8192"
DEFAULT_EXPERIMENT = "project2_qwen25_coder15b_sft_pruned_dbbal12_r16a32_ep6_len8192"


def torchDtype(name: str) -> Any:
    import torch

    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def createModel(configLeaf: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    modelName = configLeaf["model_name"]
    tokenizerName = configLeaf.get("tokenizer") or modelName
    tokenizerKwargs = dict(configLeaf.get("tokenizer_kwargs") or {})
    modelKwargs = dict(configLeaf.get("model_kwargs") or {})

    dtype = modelKwargs.get("torch_dtype")
    if isinstance(dtype, str) and dtype != "auto":
        modelKwargs["torch_dtype"] = torchDtype(dtype)

    loadIn4bit = bool(modelKwargs.pop("load_in_4bit", False))
    if loadIn4bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        computeDtype = modelKwargs.get("torch_dtype")
        if computeDtype == "auto":
            computeDtype = torch.bfloat16
        modelKwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=computeDtype,
        )
        modelKwargs.setdefault("device_map", "auto")

    tokenizer = AutoTokenizer.from_pretrained(tokenizerName, **tokenizerKwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(modelName, **modelKwargs)
    if loadIn4bit:
        model = prepare_model_for_kbit_training(model)
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    # With gradient_checkpointing=True on recent transformers, gradients do not
    # reach the LoRA adapters unless the input embeddings are made to require
    # grad (otherwise grad_norm stays 0.0 and lora_B never moves off its zero
    # init -> the adapter trains to a no-op). This explicit call replicates the
    # behaviour older TRL versions applied automatically.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


def buildRapidFireConfig(args: argparse.Namespace) -> Any:
    sft.requireSftDependencies()

    from rapidfireai.automl.datatypes import List
    from rapidfireai.automl.grid_search import RFGridSearch
    from rapidfireai.automl.model_config import RFModelConfig, RFLoraConfig, RFSFTConfig

    trainingArgs = RFSFTConfig(
        output_dir=str(sft.ROOT / "rapidfire_trainer_output"),
        dataset_text_field="text",
        max_length=args.max_length,
        packing=False,
        completion_only_loss=args.completion_only_loss,
        assistant_only_loss=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=List(args.learning_rates),
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
    )

    peftConfig = RFLoraConfig(
        r=List(args.lora_ranks),
        lora_alpha=List(args.lora_alphas),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )

    modelKwargs = {
        "torch_dtype": args.model_dtype,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.load_in_4bit:
        modelKwargs["load_in_4bit"] = True

    modelConfig = RFModelConfig(
        model_name=args.base_model,
        tokenizer=args.base_model,
        tokenizer_kwargs={
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
            "use_fast": True,
        },
        model_type="causal_lm",
        model_kwargs=modelKwargs,
        peft_config=peftConfig,
        training_args=trainingArgs,
        num_gpus=args.num_gpus,
    )
    return RFGridSearch(configs=[modelConfig], trainer_type="SFT")


def setDefaults(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    choices = parser._subparsers._group_actions[0].choices
    for command in ("prepare", "train", "predict"):
        choices[command].set_defaults(
            base_model=DEFAULT_MODEL,
            top_tables=14,
            max_schema_columns=320,
            max_columns_per_table=48,
            max_length=8192,
            prompt_keep_head_tokens=800,
        )
    for command in ("prepare", "train"):
        choices[command].set_defaults(
            run_name=DEFAULT_RUN,
            experiment_name=DEFAULT_EXPERIMENT,
            epochs=6.0,
            learning_rates="0.0001",
            lora_ranks="16",
            lora_alphas="32",
            lora_dropout=0.05,
            warmup_ratio=0.05,
            completion_only_loss=True,
            db_balance_target=12,
            db_balance_cap=0,
            gradient_accumulation_steps=8,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            model_dtype="bfloat16",
            bf16=True,
            fp16=False,
        )
        choices[command].add_argument(
            "--load_in_4bit",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use 4-bit QLoRA-style model loading for training.",
        )
    return parser


def buildParser() -> argparse.ArgumentParser:
    return setDefaults(balanced_pruned_sft.buildParser())


def main() -> None:
    args = buildParser().parse_args()
    pruned_sft.prepareData = balanced_pruned_sft.prepareData
    sft.buildRapidfireConfig = buildRapidFireConfig
    sft.createModelFn = createModel

    if args.command in {"prepare", "train"}:
        args.learning_rates = pruned_sft.parseCsvNumbers(args.learning_rates, float)
        args.lora_ranks = pruned_sft.parseCsvNumbers(args.lora_ranks, int)
        args.lora_alphas = pruned_sft.parseCsvNumbers(args.lora_alphas, int)
        if args.command == "prepare":
            paths = balanced_pruned_sft.prepareData(args)
            print(f"Wrote Coder balanced train data to {paths['train_jsonl']}")
            print(f"Wrote Coder balanced eval data to {paths['eval_jsonl']}")
            print(f"Wrote manifest to {paths['manifest']}")
        else:
            pruned_sft.runTrain(args)
    else:
        pruned_sft.predict(args)


if __name__ == "__main__":
    main()
