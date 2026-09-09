from capping_common import (
    load_jsonl, write_jsonl, dedup, total_tokens,
    apply_oversample_weight, greedy_budget_downsample, collapse_trivial_loops,
)

IN_PATH = "/home/aahav/parser/output/plaid_parsed.jsonl"
OUT_PATH = "/home/aahav/parser/output/capped/plaid_capped.jsonl"
BENIGN_TARGET = 5_400_000
ATTACK_TARGET = 85_000
FLOOR = 10
SEED = 42

records = list(load_jsonl(IN_PATH))
benign_raw = [r for r in records if r["label"] == "benign"]
attack_raw = [r for r in records if r["label"] == "attack"]

output = []
report = []

# --- benign: pooled across all 18 baseline apps, downsampled to 5.4M ---
benign_unique, benign_dups = dedup(benign_raw)
benign_capped, benign_total = greedy_budget_downsample(
    benign_unique, BENIGN_TARGET, seed=SEED, note="PLAID benign, per-source downsample (greedy budget, pooled across 18 baseline apps)"
)
output.extend(benign_capped)
report.append(("benign", len(benign_raw), len(benign_unique), benign_dups, total_tokens(benign_unique), "downsample", len(benign_capped), benign_total))

for cat in sorted(set(r["attack_type"] for r in attack_raw)):
    cat_raw = [r for r in attack_raw if r["attack_type"] == cat]
    cat_unique, cat_dups = dedup(cat_raw)
    dedup_tok = total_tokens(cat_unique)

    if len(cat_unique) < FLOOR:
        capped = []
        for r in cat_unique:
            r = dict(r)
            r["sample_weight"] = 1.0
            r["capping_note"] = f"PLAID/{cat}: below {FLOOR}-record floor — excluded from target, kept at real size, not inflated"
            capped.append(r)
        action = f"excluded (< {FLOOR}-record floor)"

    elif cat == "cowroot":
        small = [r for r in cat_unique if r["seq_len"] < ATTACK_TARGET]
        large = [r for r in cat_unique if r["seq_len"] >= ATTACK_TARGET]
        kept_large, unresolved = collapse_trivial_loops(large, score_threshold=0.9, max_period=10)
        capped = []
        for r in small:
            r = dict(r)
            r["sample_weight"] = 1.0
            r["capping_note"] = "PLAID/cowroot: tiny-tier record, kept as-is"
            capped.append(r)
        for r in kept_large:
            r["sample_weight"] = 1.0
            r["capping_note"] = "PLAID/cowroot: smallest record of a distinct trivial-loop signature — larger repeats of the same loop dropped as provably redundant"
            capped.append(r)
        for r in unresolved:
            r = dict(r)
            r["sample_weight"] = 1.0
            r["capping_note"] = "PLAID/cowroot: large record with NO clean loop signature found (score < 0.90) — kept, needs manual review"
            capped.append(r)
        action = f"special-case: {len(small)} tiny + {len(kept_large)} loop-representatives kept, {len(large)-len(kept_large)-len(unresolved)} redundant repeats dropped" + (f", {len(unresolved)} UNRESOLVED" if unresolved else "")

    elif dedup_tok >= ATTACK_TARGET:
        capped, kept_total = greedy_budget_downsample(
            cat_unique, ATTACK_TARGET, seed=SEED, note=f"PLAID/{cat}: downsampled (greedy budget)"
        )
        action = "downsample"

    else:
        capped, weight = apply_oversample_weight(
            cat_unique, ATTACK_TARGET, note=f"PLAID/{cat}: oversample x{ATTACK_TARGET/dedup_tok:.2f}"
        )
        action = f"oversample x{ATTACK_TARGET/dedup_tok:.2f}"

    output.extend(capped)
    report.append((cat, len(cat_raw), len(cat_unique), cat_dups, dedup_tok, action, len(capped), total_tokens(capped)))

write_jsonl(OUT_PATH, output)

print(f"{'bucket':<10} {'raw':>6} {'unique':>7} {'dups':>6} {'dedup_tokens':>13} {'action':<75} {'kept_recs':>10} {'kept_tokens':>12}")
for row in report:
    print(f"{row[0]:<10} {row[1]:>6} {row[2]:>7} {row[3]:>6} {row[4]:>13} {row[5]:<75} {row[6]:>10} {row[7]:>12}")
print(f"\nTotal output records: {len(output)}  -> {OUT_PATH}")
