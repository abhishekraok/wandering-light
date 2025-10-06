import os

from transformers import TrainerCallback, TrainerControl, TrainerState

URL_FILE_CONTENT = """[InternetShortcut]
URL={wandb_url}
"""


class WandbRunLinkCallback(TrainerCallback):
    """Callback to record the wandb run URL for each checkpoint."""

    def __init__(self, wandb_url: str | None):
        self.wandb_url = wandb_url

    def _write_link(self, directory: str) -> None:
        if self.wandb_url is None:
            return
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "wandb_run.url")
        with open(path, "w") as f:
            f.write(URL_FILE_CONTENT.format(wandb_url=self.wandb_url))

    def on_train_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        self._write_link(args.output_dir)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        if os.path.isdir(checkpoint_dir):
            self._write_link(checkpoint_dir)
        return control

    def write_link(self, directory: str) -> None:
        self._write_link(directory)
