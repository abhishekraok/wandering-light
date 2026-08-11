# Project conventions

- Use only Abhishek's personal services: GitHub/Hugging Face `abhishekraok` and W&B `abhishekraok-na`. Never use Ai2 organization resources.
- `checkpoints/saved/` contains the curated checkpoints worth retaining; treat other checkpoint directories as transient.
- Keep saved checkpoints weights-only unless training resumption is explicitly planned. Preserve model/tokenizer/config, trainer metadata, and `wandb_run.url`; optimizer, scheduler, and RNG state are normally disposable.
- `wandb_run.url` should point to the checkpoint's run in the personal W&B entity so model lineage remains recoverable.
