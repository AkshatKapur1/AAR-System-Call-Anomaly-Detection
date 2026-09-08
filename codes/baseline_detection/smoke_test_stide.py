"""
Phase 2 smoke test: plain STIDE (N-Gram) baseline on one LID-DS-2021 scenario.
Trimmed version of external/LID-DS/algorithms/example_main.py — drops the
autoencoder branch (avoids a torch dependency) to validate that the
dataloader + feature pipeline + IDS harness work end-to-end on our data
before writing any custom code.
"""
import sys
import os
from pprint import pprint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'LID-DS'))

from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction

from algorithms.ids import IDS
from algorithms.features.impl.max_score_threshold import MaxScoreThreshold
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.syscall_name import SyscallName
from algorithms.features.impl.stream_sum import StreamSum
from algorithms.features.impl.ngram import Ngram

from algorithms.decision_engines.stide import Stide

SCENARIO_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lid-ds-2021 datasets', 'CVE-2014-0160')

THREAD_AWARE = True
NGRAM_LENGTH = 5
STREAM_WINDOW = 500

if __name__ == '__main__':
    dataloader = dataloader_factory(SCENARIO_PATH, direction=Direction.BOTH)

    syscall_name = SyscallName()
    int_embedding = IntEmbedding(syscall_name)
    ngram = Ngram([int_embedding], THREAD_AWARE, NGRAM_LENGTH)
    stide = Stide(ngram)
    stream_sum = StreamSum(stide, False, STREAM_WINDOW, False)
    decider = MaxScoreThreshold(stream_sum)

    ids = IDS(data_loader=dataloader,
              resulting_building_block=decider,
              create_alarms=True,
              plot_switch=False)

    print("at evaluation:")
    results = ids.detect().get_results()
    pprint(results)
