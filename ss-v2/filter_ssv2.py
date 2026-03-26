"""
Filter SS-v2 train.json to keep only robot-relevant categories.

Output: filtered_train.json with added fields:
  - is_negative: bool (True for failure-side of natural pairs)
  - pair_id: str | null (links success/failure pairs)
"""

import json
from pathlib import Path

LABELS_DIR = Path(__file__).parent / "labels"

# Natural success/failure pairs: (positive_template, negative_template, pair_id)
NATURAL_PAIRS = [
    (
        "Putting [something] into [something]",
        "Failing to put [something] into [something] because [something] does not fit",
        "put_into",
    ),
    (
        "Attaching [something] to [something]",
        "Trying but failing to attach [something] to [something] because it doesn't stick",
        "attach",
    ),
    (
        "Poking a stack of [something] so the stack collapses",
        "Poking a stack of [something] without the stack collapsing",
        "poke_stack",
    ),
    (
        "Pulling two ends of [something] so that it separates into two pieces",
        "Pulling two ends of [something] but nothing happens",
        "pull_apart",
    ),
    (
        "Putting [something] upright on the table",
        "Putting [something that cannot actually stand upright] upright on the table, so it falls on its side",
        "upright",
    ),
    (
        "Tilting [something] with [something] on it until it falls off",
        "Tilting [something] with [something] on it slightly so it doesn't fall down",
        "tilt",
    ),
    (
        "Putting [something] that can't roll onto a slanted surface, so it slides down",
        "Putting [something] that can't roll onto a slanted surface, so it stays where it is",
        "slant_slide",
    ),
    (
        "Plugging [something] into [something]",
        "Plugging [something] into [something] but pulling it right out as you remove your hand",
        "plug",
    ),
]

# Curated single categories (robot-like manipulation + spatial language)
# Excludes: Pretending to..., camera actions, Showing..., human social actions
CURATED_SINGLES = {
    "Pushing [something] from left to right",
    "Pushing [something] from right to left",
    "Pulling [something] from left to right",
    "Pulling [something] from right to left",
    "Pulling [something] onto [something]",
    "Pulling [something] from behind of [something]",
    "Pulling [something] out of [something]",
    "Putting [something] next to [something]",
    "Putting [something] behind [something]",
    "Putting [something] in front of [something]",
    "Putting [something] underneath [something]",
    "Putting [something] on a surface",
    "Putting [something] onto [something]",
    "Putting [something] on a flat surface without letting it roll",
    "Putting [something] and [something] on the table",
    "Putting number of [something] onto [something]",
    "Dropping [something] behind [something]",
    "Dropping [something] in front of [something]",
    "Dropping [something] into [something]",
    "Dropping [something] next to [something]",
    "Dropping [something] onto [something]",
    "Holding [something] behind [something]",
    "Holding [something] in front of [something]",
    "Holding [something] next to [something]",
    "Holding [something] over [something]",
    "Moving [something] up",
    "Moving [something] down",
    "Moving [something] away from [something]",
    "Moving [something] closer to [something]",
    "Moving [something] across a surface without it falling down",
    "Moving [something] and [something] away from each other",
    "Moving [something] and [something] closer to each other",
    "Picking [something] up",
    "Lifting [something] up completely without letting it drop down",
    "Lifting [something] with [something] on it",
    "Pushing [something] onto [something]",
    "Pushing [something] off of [something]",
    "Pushing [something] so it slightly moves",
    "Pushing [something] with [something]",
    "Pouring [something] into [something]",
    "Pouring [something] onto [something]",
    "Pouring [something] out of [something]",
    "Scooping [something] up with [something]",
    "Spreading [something] onto [something]",
    "Stacking number of [something]",
    "Stuffing [something] into [something]",
    "Taking [something] out of [something]",
    "Taking [something] from somewhere",
    "Uncovering [something]",
    "Removing [something], revealing [something] behind",
}


def _fill_template(template: str, placeholders: list[str]) -> str:
    """Fill [something] slots in template with placeholders in order."""
    result = template
    for ph in placeholders:
        result = result.replace("[something]", ph, 1)
    return result


def build_template_lookup():
    """Map template string → (is_negative, pair_id, pos_template | None) for natural pairs."""
    lookup = {}
    for pos_tmpl, neg_tmpl, pair_id in NATURAL_PAIRS:
        lookup[pos_tmpl] = (False, pair_id, None)
        lookup[neg_tmpl] = (True, pair_id, pos_tmpl)
    return lookup


def filter_split(data: list, pair_lookup: dict, all_kept_templates: set) -> tuple[list, dict]:
    filtered = []
    skipped = 0
    for entry in data:
        tmpl = entry["template"]
        if tmpl not in all_kept_templates:
            skipped += 1
            continue

        is_negative, pair_id, pos_tmpl = pair_lookup.get(tmpl, (False, None, None))

        positive_label = None
        if is_negative and pos_tmpl is not None:
            n_pos_slots = pos_tmpl.count("[something]")
            placeholders = entry.get("placeholders", [])[:n_pos_slots]
            positive_label = _fill_template(pos_tmpl, placeholders)

        filtered.append({
            **entry,
            "is_negative": is_negative,
            "pair_id": pair_id,
            "positive_label": positive_label,
        })

    n_neg = sum(1 for e in filtered if e["is_negative"])
    stats = {
        "total": len(data),
        "kept": len(filtered),
        "skipped": skipped,
        "negatives": n_neg,
        "positives": len(filtered) - n_neg,
    }
    return filtered, stats


def main():
    pair_lookup = build_template_lookup()
    all_pair_templates = set(pair_lookup.keys())
    all_kept_templates = all_pair_templates | CURATED_SINGLES

    for in_name, out_name in [("train.json", "filtered_train.json"), ("validation.json", "filtered_validation.json")]:
        in_path  = LABELS_DIR / in_name
        out_path = LABELS_DIR / out_name

        with open(in_path) as f:
            data = json.load(f)

        filtered, stats = filter_split(data, pair_lookup, all_kept_templates)

        with open(out_path, "w") as f:
            json.dump(filtered, f, indent=2)

        print(f"\n── {in_name} → {out_name} ──")
        print(f"  Total:     {stats['total']:,}")
        print(f"  Kept:      {stats['kept']:,} ({stats['kept']/stats['total']*100:.1f}%)")
        print(f"    positives: {stats['positives']:,}")
        print(f"    negatives: {stats['negatives']:,}")
        print(f"  Skipped:   {stats['skipped']:,}")
        print(f"  Output:    {out_path}")


if __name__ == "__main__":
    main()
