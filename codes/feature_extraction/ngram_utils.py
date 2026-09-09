def sliding_windows(seq, n):
    """Yield (context, target) for every complete window of length n in seq.
    context is a tuple of n-1 tokens, target is the n-th. Never crosses a
    sequence boundary — call once per record, not across records."""
    for i in range(len(seq) - n + 1):
        window = seq[i:i + n]
        yield tuple(window[:-1]), window[-1]


def load_jsonl(path):
    import json
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
