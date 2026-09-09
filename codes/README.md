# Preprocessing pipeline (ADFA-LD + LID-DS-2021 + PLAID)

Combines all three datasets into one balanced training corpus.

## Stages

1. **`parsers/`** — one parser per dataset (`parse_adfa.py`, `parse_liddds.py`,
   `parse_plaid.py`), each reading that dataset's native raw format and
   emitting a common schema: `source`, `source_split`, `recording_id`,
   `process`, `label`, `attack_type`, `attack_subtype`, `syscalls` (name-only,
   no arguments), `seq_len`, `decode_errors`.

2. **`capping/`** — deduplication + balancing, run independently per source
   (`cap_adfa.py`, `cap_liddds.py`, `cap_plaid.py`, sharing logic from
   `capping_common.py`), then concatenated (`merge_capped.py`):
   - Exact-duplicate removal within each (source, label, category) bucket.
   - Benign data balanced **per source** (~5.4M tokens each).
   - Attack data balanced **per category** (~85K tokens each), with two
     documented exceptions: categories under a 10-unique-record floor are
     excluded from the target rather than inflated, and categories dominated
     by a genuinely trivial repeating syscall loop (e.g. PLAID's `cowroot`)
     keep only one representative per distinct loop signature.

3. **`split/`** (`split_and_tokenize.py`) — benign records split 70/15/15
   into train/val/test, stratified per source. All attack records go to
   test only (semi-supervised setup: train exclusively on benign). Builds a
   syscall vocabulary from train's benign data only, and tokenizes every
   record (`token_ids` field) against it.

4. **`feature_extraction/`** (`extract_features.py`, using `ngram_utils.py`)
   — slides a fixed-size window (currently n=7) across `train.jsonl`'s
   `token_ids`, producing `(context, target)` n-gram windows: the input
   shape either a statistical (HMM-style) model or a sequence model
   (GRU/LSTM) actually trains on. Never crosses a record boundary.

## Requires locally, not committed

The raw ADFA-LD / LID-DS-2021 / PLAID datasets themselves (third-party,
large, not this repo's to redistribute).
