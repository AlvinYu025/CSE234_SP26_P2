# Adapter

This folder contains the final LoRA adapter used by `main.py`.

- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Adapter type: LoRA / QLoRA training, PEFT loading
- Rank: 16
- Alpha: 32
- Dropout: 0.05

`main.py` loads the public base model from Hugging Face and then attaches this
adapter with PEFT. No full model checkpoint is required.
