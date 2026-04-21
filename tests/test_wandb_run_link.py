import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
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
        self.last_train_kwargs: dict | None = None

    def train(self, **kwargs):
        self.last_train_kwargs = kwargs
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
    run.url = "http://wandb.test/sft"
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
    run.url = "http://wandb.test/rl"
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


def test_sft_resume_errors_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="no checkpoint found"):
        sft.sft_main(model_name="unknown/model", resume=True, run_eval=False)


def test_sft_resume_rejects_from_scratch():
    with pytest.raises(ValueError, match="mutually exclusive"):
        sft.sft_main(resume=True, from_scratch=True)


@patch("wandering_light.training.sft.induction_dataset", return_value=[])
@patch("wandering_light.training.sft.SFTConfig")
@patch("wandering_light.training.sft.wandb")
def test_sft_resume_reuses_wandb_run(
    mock_wandb, mock_config, mock_dataset, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    # Seed a prior checkpoint with a wandb URL file, matching HF's layout.
    model_name = "tiny"
    ckpt_root = tmp_path / "checkpoints" / model_name
    ckpt_dir = ckpt_root / "checkpoint-100"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "wandb_run.url").write_text(
        "[InternetShortcut]\nURL=https://wandb.ai/myent/myproj/runs/abc123\n"
    )

    run = MagicMock()
    run.url = "https://wandb.ai/myent/myproj/runs/abc123"
    mock_wandb.init.return_value = run
    mock_wandb.run = run
    mock_wandb.log = MagicMock()
    mock_wandb.finish = MagicMock()

    mock_config.side_effect = _create_dummy_config(str(tmp_path / "out"))

    created_trainers: list[DummyTrainer] = []

    def trainer_factory(*args, **kwargs):
        t = DummyTrainer(*args, **kwargs)
        created_trainers.append(t)
        return t

    with patch(
        "wandering_light.training.sft.SFTTrainer", side_effect=trainer_factory
    ):
        sft.sft_main(
            model_name=model_name,
            full_run=False,
            run_eval=False,
            resume=True,
        )

    # wandb.init should have been called with the parsed run identifiers.
    mock_wandb.init.assert_called_once()
    init_kwargs = mock_wandb.init.call_args.kwargs
    assert init_kwargs["id"] == "abc123"
    assert init_kwargs["resume"] == "must"
    assert init_kwargs["entity"] == "myent"
    assert init_kwargs["project"] == "myproj"

    # Trainer.train should have received the checkpoint path (as HF returns it:
    # relative to the output_dir the script derives from model_name).
    assert len(created_trainers) == 1
    assert created_trainers[0].last_train_kwargs == {
        "resume_from_checkpoint": f"./checkpoints/{model_name}/checkpoint-100"
    }
