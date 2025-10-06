import fire
import numpy as np
from transformers import PreTrainedTokenizerFast

from wandering_light.training.data_generator import induction_dataset


def evaluate_tokenizer(tokenizer_path: str):
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    dataset = induction_dataset()

    chars, tokens, unk = 0, 0, 0
    lengths = []

    for prompt, completion in zip(
        dataset["prompt"], dataset["completion"], strict=False
    ):
        line = f"{prompt}\n{completion}{tok.eos_token}"
        enc = tok.encode(line)
        chars += len(line)
        tokens += len(enc)
        unk += enc.count(tok.unk_token_id)
        lengths.append(len(enc))

    print(f"Average length: {sum(lengths) / len(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Median length: {np.median(lengths)}")
    # 95th percentile
    print(f"95th percentile length: {np.percentile(lengths, 95)}")

    ratio = chars / tokens  # “chars per token”
    unk_pct = 100 * unk / tokens
    print(
        f"{tokenizer_path}:  vocab={tok.vocab_size:,}  "
        f"ratio={ratio:.2f}  unk%={unk_pct:.2f}"
    )


def roundtrip_tokenize(tokenizer_path: str, limit: int = 10):
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    ds = induction_dataset()
    for i in range(limit):
        enc_prompt = tok.encode(ds["prompt"][i])
        enc_completion = tok.encode(ds["completion"][i])
        decoded_prompt = tok.decode(enc_prompt)
        if decoded_prompt != ds["prompt"][i]:
            print(f"Prompt mismatch: {decoded_prompt} != {ds['prompt'][i]}")
            break
        decoded_completion = tok.decode(enc_completion)
        if decoded_completion != ds["completion"][i]:
            print(f"Completion mismatch: {decoded_completion} != {ds['completion'][i]}")
            break
    print("✅  roundtrip tokenization successful")


if __name__ == "__main__":
    fire.Fire()
