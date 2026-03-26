"""Filter SS-v2 labels to keep only negatives and their paired positives.

Writes:
  ss-v2/labels/negative_pairs_train.json
  ss-v2/labels/negative_pairs_val.json
"""

import json
from pathlib import Path

LABELS_DIR = Path(__file__).parent / "labels"


def filter_negative_pairs(src_path: Path, dst_path: Path) -> None:
    with open(src_path) as f:
        data = json.load(f)

    neg = [e for e in data if e.get("is_negative")]
    neg_pair_ids = {e["pair_id"] for e in neg}
    paired_pos = [e for e in data if not e.get("is_negative") and e.get("pair_id") in neg_pair_ids]

    result = neg + paired_pos
    with open(dst_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{src_path.name}: {len(data)} → {len(result)} "
          f"({len(neg)} neg + {len(paired_pos)} paired pos)")


if __name__ == "__main__":
    filter_negative_pairs(
        LABELS_DIR / "filtered_train.json",
        LABELS_DIR / "negative_pairs_train.json",
    )
    filter_negative_pairs(
        LABELS_DIR / "filtered_validation.json",
        LABELS_DIR / "negative_pairs_val.json",
    )
