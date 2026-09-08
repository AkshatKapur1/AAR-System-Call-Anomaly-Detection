"""
Phase 3: shared preprocessing pipeline — turns raw syscall recordings into
name-only, integer-encoded sliding-window n-grams using the locked config
in configs/feature_config.py. Every model tier (N-Gram, HMM, LSTM, GRU)
builds on this same windowed representation so Phase 6's benchmark stays
apples-to-apples.
"""
import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'LID-DS'))
sys.path.insert(0, os.path.dirname(__file__))

from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction
from dataloader.data_loader_2021 import RecordingType

from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.syscall_name import SyscallName
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.stream_sum import StreamSum

from feature_config import THREAD_AWARE, NGRAM_LENGTH, WINDOW_LENGTH


def clean_recordings(recordings):
    """
    Cleaning/filtering step: drop IDLE recordings (neither normal behavior
    nor exploit — e.g. container warmup with no traffic) since they carry
    no behavioral signal for training or evaluation. Malformed syscall
    lines are already dropped upstream by Syscall2021.params()/Recording2021.
    """
    kept = []
    for recording in recordings:
        recording_type = get_recording_type(recording)
        if recording_type != RecordingType.IDLE:
            kept.append(recording)
    return kept


def get_recording_type(recording) -> RecordingType:
    from dataloader.data_loader_2021 import get_type_of_recording
    return get_type_of_recording(recording.metadata())


def build_chain():
    """Fresh SyscallName -> IntEmbedding -> Ngram chain (holds per-recording state)."""
    syscall_name = SyscallName()
    int_embedding = IntEmbedding(syscall_name)
    ngram = Ngram([int_embedding], THREAD_AWARE, NGRAM_LENGTH)
    return int_embedding, ngram


def build_scoring_chain(decision_engine):
    """
    Wraps a Phase 4 decision engine (e.g. Stide) with the locked
    WINDOW_LENGTH sliding-window score aggregation (StreamSum), so every
    model tier smooths its per-syscall anomaly score the same way before
    thresholding — matching the roadmap's "7-gram, window length 100"
    Phase 3 default.
    """
    return StreamSum(decision_engine, False, WINDOW_LENGTH, False)


def train_vocabulary(int_embedding, recordings):
    """
    Builds the syscall-name -> integer-ID vocabulary from the given
    recordings (must run BEFORE extract_ngrams, otherwise every syscall
    encodes to 0 = "unknown").
    """
    for recording in recordings:
        for syscall in recording.syscalls():
            int_embedding.train_on(syscall)


def extract_ngrams(recordings, ngram):
    """
    Yields (recording_name, ngram_tuple) for every complete window across
    the given recordings, using an already vocabulary-trained ngram chain.
    Resets the window buffer between recordings so n-grams never span two
    different recordings.
    """
    for recording in recordings:
        ngram.new_recording()
        for syscall in recording.syscalls():
            result = ngram.get_result(syscall)
            if result is not None:
                yield recording.name, result


def export_vocabulary_csv(int_embedding, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['syscall_name', 'integer_id'])
        for name, syscall_id in sorted(int_embedding._syscall_dict.items(), key=lambda kv: kv[1]):
            writer.writerow([name, syscall_id])


def export_ngrams_csv(recordings, ngram, out_path):
    row_count = 0
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['recording'] + [f'syscall_{i+1}' for i in range(NGRAM_LENGTH)])
        for name, ngram_tuple in extract_ngrams(recordings, ngram):
            writer.writerow([name, *ngram_tuple])
            row_count += 1
    return row_count


if __name__ == '__main__':
    scenario_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lid-ds-2021 datasets', 'CVE-2014-0160')
    out_dir = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'phase3_preprocessing')
    scenario_name = os.path.basename(os.path.normpath(scenario_path))

    dataloader = dataloader_factory(scenario_path, direction=Direction.BOTH)
    raw_training = dataloader.training_data()
    training = clean_recordings(raw_training)

    int_embedding, ngram = build_chain()

    print(f"scenario: {scenario_path}")
    print(f"training recordings: {len(raw_training)} raw -> {len(training)} after cleaning "
          f"(dropped {len(raw_training) - len(training)} IDLE)")

    print("building vocabulary (pass 1/2)...")
    train_vocabulary(int_embedding, training)
    print(f"vocabulary size: {len(int_embedding._syscall_dict)} distinct syscalls")

    vocab_path = os.path.join(out_dir, f"{scenario_name}_vocabulary.csv")
    export_vocabulary_csv(int_embedding, vocab_path)
    print(f"wrote vocabulary -> {vocab_path}")

    print("extracting n-grams (pass 2/2)...")
    ngrams_path = os.path.join(out_dir, f"{scenario_name}_ngrams.csv")
    total = export_ngrams_csv(training, ngram, ngrams_path)
    print(f"wrote {total} {NGRAM_LENGTH}-grams (thread_aware={THREAD_AWARE}) -> {ngrams_path}")
    print(f"scoring-chain window length (for Phase 4 decision engines): {WINDOW_LENGTH}")
