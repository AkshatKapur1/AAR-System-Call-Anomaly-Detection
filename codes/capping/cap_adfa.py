from capping_common import load_jsonl, write_jsonl, dedup, total_tokens, apply_oversample_weight

IN_PATH = "/home/aahav/parser/output/adfa_parsed.jsonl"
OUT_PATH = "/home/aahav/parser/output/capped/adfa_capped.jsonl"
BENIGN_TARGET = 5_400_000
ATTACK_TARGET = 85_000

records = list(load_jsonl(IN_PATH))
benign_raw = [r for r in records if r["label"] == "benign"]
attack_raw = [r for r in records if r["label"] == "attack"]

output = []
report = []

benign_unique, benign_dups = dedup(benign_raw)
benign_capped, benign_weight = apply_oversample_weight(
    benign_unique, BENIGN_TARGET, note=f"ADFA-LD benign, per-source oversample x{BENIGN_TARGET/total_tokens(benign_unique):.2f}"
)
output.extend(benign_capped)
report.append(("benign", len(benign_raw), len(benign_unique), benign_dups, total_tokens(benign_unique), benign_weight, len(benign_capped)))

for cat in sorted(set(r["attack_type"] for r in attack_raw)):
    cat_raw = [r for r in attack_raw if r["attack_type"] == cat]
    cat_unique, cat_dups = dedup(cat_raw)
    cat_capped, cat_weight = apply_oversample_weight(
        cat_unique, ATTACK_TARGET, note=f"ADFA-LD/{cat} attack, per-category oversample x{ATTACK_TARGET/total_tokens(cat_unique):.2f}"
    )
    output.extend(cat_capped)
    report.append((cat, len(cat_raw), len(cat_unique), cat_dups, total_tokens(cat_unique), cat_weight, len(cat_capped)))

write_jsonl(OUT_PATH, output)

print(f"{'bucket':<20} {'raw':>7} {'unique':>7} {'dups':>6} {'dedup_tokens':>13} {'weight':>8} {'kept_records':>13}")
for row in report:
    print(f"{row[0]:<20} {row[1]:>7} {row[2]:>7} {row[3]:>6} {row[4]:>13} {row[5]:>8.2f} {row[6]:>13}")
print(f"\nTotal output records: {len(output)}  -> {OUT_PATH}")
