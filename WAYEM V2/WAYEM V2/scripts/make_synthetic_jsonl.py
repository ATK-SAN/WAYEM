#!/usr/bin/env python3
"""
Synthesize JSONL datasets (train/val/test/challenge) from your ontology.

- Uses ontology/moods.yml canonicals
- Uses ontology/aliases.csv to generate natural prompts
- Optionally uses ontology/mood_prompts.yml for richer phrasing
- Produces JSONL lines: {"text": "...", "labels": ["Mood1", "Mood2", ...]}

CLI:
    python scripts/make_synthetic_jsonl.py \
        --output-dir data \
        --per-class 120 \
        --val-ratio 0.1 \
        --test-ratio 0.1 \
        --challenge-per-class 20 \
        --seed 42
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import yaml


def load_canonicals(moods_path: str) -> List[str]:
    if not os.path.exists(moods_path):
        raise FileNotFoundError(f"Missing moods file: {moods_path}")
    with open(moods_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    canonicals = data.get("canonicals") or []
    out, seen = [], set()
    for c in canonicals:
        c = (c or "").strip()
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    if not out:
        raise ValueError("canonicals is empty.")
    return out


def load_aliases(aliases_path: str, canonicals_set: set[str]) -> Dict[str, str]:
    if not os.path.exists(aliases_path):
        raise FileNotFoundError(f"Missing aliases file: {aliases_path}")
    alias2canon: Dict[str, str] = {}
    with open(aliases_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = []
        if header and len(header) >= 2 and header[0].lower() == "alias":
            rows = list(reader)
        else:
            rows = ([header] if header else []) + list(reader)
        for row in rows:
            if not row or len(row) < 2:
                continue
            alias = (row[0] or "").strip().lower()
            canon = (row[1] or "").strip()
            if not alias or not canon or canon not in canonicals_set:
                continue
            alias2canon[alias] = canon
    return alias2canon


def load_prompts(prompts_path: str) -> Dict[str, str]:
    if not os.path.exists(prompts_path):
        return {}
    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return {k: (v or "").strip() for k, v in data.items()}
    except Exception:
        return {}


# --- text templates -----------------------------------------------------------

SINGLE_TEMPLATES = [
    "{alias}",
    "i need {alias}",
    "music to {alias}",
    "{alias} playlist",
    "{alias} vibes",
    "{alias} only",
    "looking for {alias} songs",
    "give me something {alias}",
    "set the mood: {alias}",
    "need {alias} background",
    "perfect for {alias}",
]

# For multi-label examples (two moods in one prompt)
DUAL_TEMPLATES = [
    "{alias1} but also {alias2}",
    "{alias1} with a touch of {alias2}",
    "{alias1} / {alias2} mix",
    "{alias1} yet still {alias2}",
    "between {alias1} and {alias2}",
]


def clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def synthesize_for_label(
    canon: str,
    aliases_for_label: List[str],
    per_label: int,
    prompts_hint: str | None,
    rnd: random.Random,
) -> List[Dict]:
    """Create positive single-label examples for one canonical label."""
    out: List[Dict] = []
    if not aliases_for_label:
        # fallback: use canonical as alias
        aliases_for_label = [canon.lower()]

    # include some raw alias strings directly
    base = []
    for a in aliases_for_label:
        base.append({"text": a, "labels": [canon]})

    # template-generated
    gen: List[Dict] = []
    all_aliases = list(aliases_for_label)
    rnd.shuffle(all_aliases)

    # Boost coverage with variety
    for _ in range(max(0, per_label - len(base))):
        a = rnd.choice(all_aliases)
        tpl = rnd.choice(SINGLE_TEMPLATES)
        gen.append({"text": clean_space(tpl.format(alias=a)), "labels": [canon]})

    # sprinkle prompt hint paraphrases if available
    if prompts_hint:
        phrases = [p.strip() for p in prompts_hint.split(",") if p.strip()]
        for ph in phrases[:4]:
            gen.append({"text": clean_space(ph), "labels": [canon]})

    out.extend(base)
    out.extend(gen)
    # trim or pad:
    if len(out) > per_label:
        out = rnd.sample(out, per_label)
    elif len(out) < per_label:
        # duplicate with small perturbations (cheap)
        while len(out) < per_label:
            a = rnd.choice(all_aliases)
            tpl = rnd.choice(SINGLE_TEMPLATES)
            out.append({"text": clean_space(tpl.format(alias=a)), "labels": [canon]})
    return out


def synthesize_dual_label(
    canon1: str, canon2: str,
    aliases1: List[str], aliases2: List[str],
    count: int,
    rnd: random.Random,
) -> List[Dict]:
    out: List[Dict] = []
    if not aliases1:
        aliases1 = [canon1.lower()]
    if not aliases2:
        aliases2 = [canon2.lower()]
    for _ in range(count):
        a1 = rnd.choice(aliases1)
        a2 = rnd.choice(aliases2)
        tpl = rnd.choice(DUAL_TEMPLATES)
        txt = tpl.format(alias1=a1, alias2=a2)
        out.append({"text": clean_space(txt), "labels": [canon1, canon2]})
    return out


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # utf-8 WITHOUT BOM by default
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")


def make_splits(
    rows: List[Dict],
    val_ratio: float,
    test_ratio: float,
    rnd: random.Random,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rnd.shuffle(rows)
    n = len(rows)
    n_test = int(n * test_ratio)
    n_val = int((n - n_test) * val_ratio)
    test_rows = rows[:n_test]
    val_rows = rows[n_test:n_test + n_val]
    train_rows = rows[n_test + n_val:]
    return train_rows, val_rows, test_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moods", default="ontology/moods.yml")
    ap.add_argument("--aliases", default="ontology/aliases.csv")
    ap.add_argument("--prompts", default="ontology/mood_prompts.yml")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--dual-per-pair", type=int, default=8,
                    help="number of dual-label samples per randomly selected pair; total pairs limited")
    ap.add_argument("--dual-pairs", type=int, default=60,
                    help="how many label pairs to synthesize (random)")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--challenge-per-class", type=int, default=20,
                    help="harder prompts per class for challenge split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)

    canonicals = load_canonicals(args.moods)
    alias2canon = load_aliases(args.aliases, set(canonicals))
    prompts_map = load_prompts(args.prompts)

    # group aliases by canonical
    canon2aliases: Dict[str, List[str]] = defaultdict(list)
    for a, c in alias2canon.items():
        canon2aliases[c].append(a)

    # --- single-label samples
    all_rows: List[Dict] = []
    for c in canonicals:
        rows = synthesize_for_label(
            c,
            canon2aliases.get(c, []),
            per_label=args.per_class,
            prompts_hint=prompts_map.get(c),
            rnd=rnd,
        )
        all_rows.extend(rows)

    # --- dual-label samples (create synergy pairs, randomized)
    pairs = []
    seed_pairs = [
        ("Focus", "Chill"),
        ("Energetic", "Euphoric"),
        ("Somber", "Melancholic"),
        ("Dreamy", "Ethereal"),
        ("Romantic", "Sensual"),
        ("Nostalgic", "Bittersweet"),
        ("Confident", "Groovy"),
        ("Epic", "Triumphant"),
        ("Atmospheric", "Hypnotic"),
    ]
    for p in seed_pairs:
        if p[0] in canonicals and p[1] in canonicals:
            pairs.append(p)

    # fill remaining pairs randomly
    left = max(0, args.dual_pairs - len(pairs))
    candidates = canonicals[:]
    rnd.shuffle(candidates)
    while left > 0 and len(candidates) >= 2:
        c1 = candidates.pop()
        c2 = rnd.choice(canonicals)
        if c1 == c2:
            continue
        pair = (c1, c2)
        if pair not in pairs and (c2, c1) not in pairs:
            pairs.append(pair)
            left -= 1

    for (c1, c2) in pairs:
        rows = synthesize_dual_label(
            c1, c2,
            canon2aliases.get(c1, []),
            canon2aliases.get(c2, []),
            count=args.dual_per_pair,
            rnd=rnd,
        )
        all_rows.extend(rows)

    # dedupe by text (case-insensitive)
    seen = set()
    deduped = []
    for r in all_rows:
        key = r["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # --- splits
    train_rows, val_rows, test_rows = make_splits(deduped, args.val_ratio, args.test_ratio, rnd)

    # --- challenge split (harder templates: negations/contrasts)
    challenge_rows: List[Dict] = []
    NEG_TEMPLATES = [
        "not too {alias_bad}, keep it {alias_good}",
        "less {alias_bad}, more {alias_good}",
        "somewhere between {alias_good} and not {alias_bad}",
    ]
    for c in canonicals:
        aliases = canon2aliases.get(c, [c.lower()])
        good = rnd.choice(aliases)
        bad_cand = rnd.choice([x for x in canonicals if x != c]) if len(canonicals) > 1 else c
        bad_aliases = canon2aliases.get(bad_cand, [bad_cand.lower()])
        bad = rnd.choice(bad_aliases)
        neg = NEG_TEMPLATES[rnd.randrange(len(NEG_TEMPLATES))].format(alias_bad=bad, alias_good=good)
        challenge_rows.append({"text": clean_space(neg), "labels": [c]})

    # top up challenge set per class
    target_challenge = args.challenge_per_class * len(canonicals)
    while len(challenge_rows) < target_challenge:
        c = rnd.choice(canonicals)
        aliases = canon2aliases.get(c, [c.lower()])
        a1 = rnd.choice(aliases)
        a2 = rnd.choice(aliases)
        tpl = rnd.choice([
            "{alias1}, {alias2}, keep it consistent",
            "definitely {alias1}, maybe {alias2}",
            "strictly {alias1}, ideally {alias2}",
        ])
        challenge_rows.append({"text": clean_space(tpl.format(alias1=a1, alias2=a2)), "labels": [c]})

    # --- write files (UTF-8, no BOM)
    outdir = args.output_dir
    write_jsonl(os.path.join(outdir, "train.jsonl"), train_rows)
    write_jsonl(os.path.join(outdir, "val.jsonl"), val_rows)
    write_jsonl(os.path.join(outdir, "test.jsonl"), test_rows)
    write_jsonl(os.path.join(outdir, "challenge.jsonl"), challenge_rows[:target_challenge])

    print(f"Wrote {len(train_rows)} train, {len(val_rows)} val, {len(test_rows)} test, {target_challenge} challenge rows.")


if __name__ == "__main__":
    main()
