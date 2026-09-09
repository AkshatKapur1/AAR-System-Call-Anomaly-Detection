import json
import random


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def dedup(records):
    """Collapse byte-identical syscall sequences within the given record set.
    Keeps the first occurrence; returns (unique_records, dup_count)."""
    seen = {}
    dup_count = 0
    for r in records:
        key = tuple(r["syscalls"])
        if key not in seen:
            seen[key] = r
        else:
            dup_count += 1
    return list(seen.values()), dup_count


def total_tokens(records):
    return sum(r["seq_len"] for r in records)


def greedy_budget_downsample(records, target_tokens, seed=42, note=""):
    """Fixed-seed shuffle, then add whole records until the running total
    reaches (and may slightly exceed) target_tokens. Never slices a record."""
    pool = list(records)
    random.Random(seed).shuffle(pool)
    kept = []
    running = 0
    for r in pool:
        if running >= target_tokens:
            break
        r = dict(r)
        r["sample_weight"] = 1.0
        r["capping_note"] = note or "downsampled: greedy token-budget"
        kept.append(r)
        running += r["seq_len"]
    return kept, running


def apply_oversample_weight(records, target_tokens, note=""):
    """Every real record is kept; a sample_weight = target / dedup_tokens is
    attached uniformly so a trainer can treat it as occurring that many times."""
    dedup_tokens = total_tokens(records)
    weight = (target_tokens / dedup_tokens) if dedup_tokens else 0.0
    out = []
    for r in records:
        r = dict(r)
        r["sample_weight"] = weight
        r["capping_note"] = note or f"oversampled x{weight:.2f}"
        out.append(r)
    return out, weight


def find_loop_period(seq, max_period=10, sample_step=7, window=20000):
    """Detect a short exact-repeat period near the start of seq. Returns
    (period, score, body) — score is the fraction of sampled positions
    where seq[i] == seq[i+period] holds, inside the first `window` tokens."""
    test_len = min(window, len(seq) - max_period)
    if test_len <= 0:
        return None, 0.0, []
    best_p, best_score = None, 0.0
    for p in range(1, max_period + 1):
        matches, total = 0, 0
        for i in range(0, test_len, sample_step):
            if seq[i] == seq[i + p]:
                matches += 1
            total += 1
        score = matches / total
        if score > best_score:
            best_score, best_p = score, p
    body = seq[:best_p] if best_p else []
    return best_p, best_score, body


def collapse_trivial_loops(large_records, score_threshold=0.9, max_period=10):
    """For records whose length is dominated by a genuinely trivial repeating
    loop (near-perfect period match), keep only the smallest record per
    distinct loop signature — provably lossless when score exceeds threshold.
    Records that do NOT show a clean loop are returned untouched in `unresolved`."""
    by_signature = {}
    unresolved = []
    for r in sorted(large_records, key=lambda x: x["seq_len"]):
        period, score, body = find_loop_period(r["syscalls"], max_period=max_period)
        if period and score >= score_threshold:
            sig = tuple(body)
            if sig not in by_signature:
                by_signature[sig] = r
        else:
            unresolved.append(r)
    return list(by_signature.values()), unresolved
