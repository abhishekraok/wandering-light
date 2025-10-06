import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from transformers import TrainerControl, TrainerState

from wandering_light.training import rl_grpo, sft


def _create_dummy_config(output_dir):
    def _factory(**kwargs):
        kwargs["output_dir"] = output_dir
        return SimpleNamespace(**kwargs)

    return _factory


def _parse_wandb_url_file(file_path):
    """Parse wandb URL from Internet Shortcut format."""
    content = file_path.read_text().strip()
    # Expected format:
    # [InternetShortcut]
    # URL=http://wandb.test/sft
    for line in content.split("\n"):
        if line.startswith("URL="):
            return line[4:]  # Remove 'URL=' prefix
    raise ValueError(f"Could not find URL in content: {content}")


class DummyTrainer:
    def __init__(self, *args, **kwargs):
        self.args = kwargs.get("args")
        self.callback_handler = SimpleNamespace(callbacks=kwargs.get("callbacks", []))

    def train(self):
        state = TrainerState(global_step=1)
        control = TrainerControl()
        for cb in self.callback_handler.callbacks:
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin(self.args, state, control)
            if hasattr(cb, "on_save"):
                os.makedirs(
                    os.path.join(self.args.output_dir, "checkpoint-1"), exist_ok=True
                )
                cb.on_save(self.args, state, control)

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)


@patch("wandering_light.training.sft.induction_dataset", return_value=[])
@patch("wandering_light.training.sft.SFTTrainer", side_effect=DummyTrainer)
@patch("wandering_light.training.sft.SFTConfig")
@patch("wandering_light.training.sft.wandb")
def test_sft_wandb_run_link(
    mock_wandb, mock_config, mock_trainer, mock_dataset, tmp_path
):
    run = MagicMock()
    run.get_url.return_value = "http://wandb.test/sft"
    mock_wandb.init.return_value = run
    mock_wandb.run = run
    mock_wandb.log = MagicMock()
    mock_wandb.finish = MagicMock()

    mock_config.side_effect = _create_dummy_config(str(tmp_path / "out"))

    sft.sft_main(
        model_name="tiny",
        full_run=False,
        run_eval=False,
        wandb_project="p",
        wandb_run_name="r",
    )

    out_dir = tmp_path / "out"
    link_file = out_dir / "wandb_run.url"
    assert link_file.exists()
    assert _parse_wandb_url_file(link_file) == "http://wandb.test/sft"
    ckpt_file = out_dir / "checkpoint-1" / "wandb_run.url"
    assert ckpt_file.exists()
    assert _parse_wandb_url_file(ckpt_file) == "http://wandb.test/sft"
    final_file = out_dir / "final_model" / "wandb_run.url"
    assert final_file.exists()
    assert _parse_wandb_url_file(final_file) == "http://wandb.test/sft"


@patch(
    "wandering_light.training.rl_grpo.induction_dataset_rl", return_value=([], {}, {})
)
@patch("wandering_light.training.rl_grpo.AutoTokenizer")
@patch("wandering_light.training.rl_grpo.GRPOTrainer", side_effect=DummyTrainer)
@patch("wandering_light.training.rl_grpo.GRPOConfig")
@patch("wandering_light.training.rl_grpo.wandb")
def test_rl_grpo_wandb_run_link(
    mock_wandb, mock_config, mock_trainer, mock_tokenizer, mock_dataset, tmp_path
):
    run = MagicMock()
    run.get_url.return_value = "http://wandb.test/rl"
    mock_wandb.init.return_value = run
    mock_wandb.run = run
    mock_wandb.log = MagicMock()
    mock_wandb.finish = MagicMock()

    mock_config.side_effect = _create_dummy_config(str(tmp_path / "out"))
    mock_tokenizer.from_pretrained.return_value = MagicMock(
        pad_token=None, eos_token="<eos>"
    )

    rl_grpo.rl_grpo_main(
        model_name="tiny",
        model=None,
        full_run=False,
        eval_steps=1,
        eval_samples=1,
        num_train_epochs=0.1,
        learning_rate=1e-5,
        max_length=8,
        per_device_train_batch_size=1,
        wandb_project="p",
        wandb_run_name="r",
    )

    out_dir = tmp_path / "out"
    link_file = out_dir / "wandb_run.url"
    assert link_file.exists()
    assert _parse_wandb_url_file(link_file) == "http://wandb.test/rl"
    ckpt_file = out_dir / "checkpoint-1" / "wandb_run.url"
    assert ckpt_file.exists()
    assert _parse_wandb_url_file(ckpt_file) == "http://wandb.test/rl"
    final_model_dir = Path("./checkpoints/rl/grpo/induction/tiny/final_model")
    final_file = final_model_dir / "wandb_run.url"
    assert final_file.exists()
    assert _parse_wandb_url_file(final_file) == "http://wandb.test/rl"
