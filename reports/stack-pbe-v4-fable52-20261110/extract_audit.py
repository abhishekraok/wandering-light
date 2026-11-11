"""Export completed Claude commands/stdout and stderr digests from local raw files."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = "fable52_low_v4_sample40"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    directory = args.data_dir / "raw" / STEM
    expected = [f"call-{i:03d}" for i in range(1, 41)]
    paths = sorted(directory.glob("*.stdout.json"))
    assert [p.name.removesuffix(".stdout.json") for p in paths] == expected
    calls = []
    for stem, path in zip(expected, paths, strict=True):
        stdout = path.read_text()
        json.loads(stdout)  # Reject an incomplete JSON write before exporting.
        calls.append(
            {
                "call": stem,
                "command": json.loads((directory / f"{stem}.command.json").read_text()),
                "stdout": stdout,
                "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(
                    (directory / f"{stem}.stderr.txt").read_bytes()
                ).hexdigest(),
            }
        )
    audit = {"artifact": f"{STEM}.json", "calls": calls}
    text = json.dumps(audit, indent=2) + "\n"
    destination = args.out or args.data_dir / "cli-audit.json"
    if destination.exists():
        assert destination.read_text() == text, "existing audit differs from raw"
    else:
        destination.write_text(text)
    print(f"Exported {len(calls)} original CLI commands/stdout: {destination}")


if __name__ == "__main__":
    main()
