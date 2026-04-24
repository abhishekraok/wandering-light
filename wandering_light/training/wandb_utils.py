import os
from urllib.parse import urlparse

import wandb
from transformers import TrainerCallback, TrainerControl, TrainerState


def define_wandb_step_metric() -> None:
    """Bind training/eval/etc. metrics to a custom "step" field for x-axis.

    trl/HF advance wandb's internal step counter via their own logging,
    which can race with our slow eval callbacks and cause our wandb.log
    calls (using state.global_step) to be silently ignored. Defining a
    dedicated step metric and dropping the explicit `step=` kwarg lets
    us log out of order against our own step field.
    """
    if wandb.run is None:
        return
    wandb.define_metric("step")
    for prefix in ("training", "eval", "interval_metadata", "final"):
        wandb.define_metric(f"{prefix}/*", step_metric="step")

URL_FILE_CONTENT = """[InternetShortcut]
URL={wandb_url}
"""


def read_wandb_url_file(path: str) -> str | None:
    """Read the URL from an InternetShortcut file written by WandbRunLinkCallback."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("URL="):
                    return line[4:].strip()
    except FileNotFoundError:
        return None
    return None


def parse_wandb_run_url(url: str) -> tuple[str, str, str] | None:
    """Parse a wandb run URL into (entity, project, run_id), or None if unparseable.

    Expected shape: https://wandb.ai/<entity>/<project>/runs/<run_id>
    """
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) >= 4 and parts[-2] == "runs":
        return parts[0], parts[1], parts[-1]
    return None


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
