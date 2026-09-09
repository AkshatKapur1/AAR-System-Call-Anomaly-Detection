from capping_common import load_jsonl, write_jsonl, dedup, total_tokens, apply_oversample_weight, greedy_budget_downsample

IN_PATH = "/home/aahav/parser/output/liddds_parsed.jsonl"
OUT_PATH = "/home/aahav/parser/output/capped/liddds_capped.jsonl"
BENIGN_TARGET = 5_400_000
ATTACK_TARGET = 85_000
SEED = 42

records = list(load_jsonl(IN_PATH))
benign_raw = [r for r in records if r["label"] == "benign"]
attack_raw = [r for r in records if r["label"] == "attack"]

output = []
report = []

# --- benign: pooled across both CVE recording pools, downsampled to 5.4M ---
benign_unique, benign_dups = dedup(benign_raw)
benign_capped, benign_total = greedy_budget_downsample(
    benign_unique, BENIGN_TARGET, seed=SEED, note="LID-DS benign, per-source downsample (greedy budget)"
)
output.extend(benign_capped)
report.append(("benign", len(benign_raw), len(benign_unique), benign_dups, total_tokens(benign_unique), "downsample", len(benign_capped), benign_total))

# --- CVE-2014-0160: single subtype, straightforward oversample ---
cve0160_raw = [r for r in attack_raw if r["attack_type"] == "CVE-2014-0160"]
cve0160_unique, cve0160_dups = dedup(cve0160_raw)
cve0160_capped, cve0160_weight = apply_oversample_weight(
    cve0160_unique, ATTACK_TARGET, note=f"CVE-2014-0160 attack, per-category oversample x{ATTACK_TARGET/total_tokens(cve0160_unique):.2f}"
)
output.extend(cve0160_capped)
report.append(("CVE-2014-0160", len(cve0160_raw), len(cve0160_unique), cve0160_dups, total_tokens(cve0160_unique), f"oversample x{cve0160_weight:.2f}", len(cve0160_capped), total_tokens(cve0160_capped)))

# --- CVE-2012-2122: 5 subtypes, 3 of which have a genuinely-diverse large tier ---
cve2122_raw = [r for r in attack_raw if r["attack_type"] == "CVE-2012-2122"]
cve2122_unique, cve2122_dups = dedup(cve2122_raw)

FORCE_SUBTYPES = {"slow-extract", "slow-no-extract", "very-slow-no-extract"}
small_pool = []
forced_large = []
for subtype in set(r["attack_subtype"] for r in cve2122_unique):
    group = [r for r in cve2122_unique if r["attack_subtype"] == subtype]
    if subtype in FORCE_SUBTYPES:
        small = [r for r in group if r["seq_len"] < ATTACK_TARGET]
        large = sorted([r for r in group if r["seq_len"] >= ATTACK_TARGET], key=lambda x: x["seq_len"])
        small_pool.extend(small)
        if large:
            chosen = dict(large[0])
            chosen["sample_weight"] = 1.0
            chosen["capping_note"] = f"CVE-2012-2122/{subtype}: forced inclusion (smallest real large record) — subtype otherwise unrepresented, real transfer data (not a redundant loop), overshoot accepted"
            forced_large.append(chosen)
    else:
        small_pool.extend(group)

small_capped, small_total = greedy_budget_downsample(
    small_pool, ATTACK_TARGET, seed=SEED, note="CVE-2012-2122: greedy token-budget over normal-extract/normal-no-extract + tiny fragments from slow variants"
)
cve2122_capped = small_capped + forced_large
output.extend(cve2122_capped)
report.append((
    "CVE-2012-2122", len(cve2122_raw), len(cve2122_unique), cve2122_dups, total_tokens(cve2122_unique),
    f"budget+forced ({len(forced_large)} forced)", len(cve2122_capped), total_tokens(cve2122_capped)
))

write_jsonl(OUT_PATH, output)

print(f"{'bucket':<16} {'raw':>6} {'unique':>7} {'dups':>6} {'dedup_tokens':>13} {'action':<45} {'kept_recs':>10} {'kept_tokens':>12}")
for row in report:
    print(f"{row[0]:<16} {row[1]:>6} {row[2]:>7} {row[3]:>6} {row[4]:>13} {row[5]:<45} {row[6]:>10} {row[7]:>12}")
print(f"\nTotal output records: {len(output)}  -> {OUT_PATH}")
