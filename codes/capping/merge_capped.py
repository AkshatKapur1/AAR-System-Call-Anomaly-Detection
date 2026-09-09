from capping_common import load_jsonl, write_jsonl

FILES = [
    "/home/aahav/parser/output/capped/adfa_capped.jsonl",
    "/home/aahav/parser/output/capped/liddds_capped.jsonl",
    "/home/aahav/parser/output/capped/plaid_capped.jsonl",
]
OUT_PATH = "/home/aahav/parser/output/capped/combined_training_data.jsonl"

all_records = []
for path in FILES:
    all_records.extend(list(load_jsonl(path)))

write_jsonl(OUT_PATH, all_records)

from collections import Counter
by_source_label = Counter()
tokens_by_source_label = Counter()
for r in all_records:
    key = (r["source"].split("/")[0], r["label"])
    by_source_label[key] += 1
    tokens_by_source_label[key] += r["seq_len"]

print(f"{'source':<10} {'label':<8} {'records':>9} {'raw_tokens':>12}")
for key in sorted(by_source_label):
    print(f"{key[0]:<10} {key[1]:<8} {by_source_label[key]:>9} {tokens_by_source_label[key]:>12}")

print(f"\nTotal records: {len(all_records)}")
print(f"Total raw tokens (pre sample_weight): {sum(r['seq_len'] for r in all_records)}")
print(f"Output: {OUT_PATH}")
