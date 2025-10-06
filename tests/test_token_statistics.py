import training.data_generator as data_gen
from datasets import Dataset


class DummyTokenizer:
    eos_token = "<eos>"

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[0] * len(t) for t in texts]}


def test_token_statistics(monkeypatch, capsys):
    def fake_dataset():
        return Dataset.from_list(
            [
                {"prompt": "a", "completion": "b"},
                {"prompt": "cc", "completion": "d"},
            ]
        )

    monkeypatch.setattr(data_gen, "induction_dataset", fake_dataset)
    monkeypatch.setattr(
        data_gen,
        "AutoTokenizer",
        type(
            "Tok",
            (),
            {
                "from_pretrained": staticmethod(
                    lambda name, use_fast=True: DummyTokenizer()
                )
            },
        ),
    )

    data_gen.token_statistics("dummy-model")

    out = capsys.readouterr().out
    assert "# samples" in out
    assert "Suggested max_seq_length" in out
    assert "7.50" in out  # mean length for our dummy data
