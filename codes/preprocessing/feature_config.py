"""
Phase 3 standard feature-extraction config, shared by every model tier
(N-Gram, HMM, LSTM, GRU) so results stay comparable across Phase 6's
cross-tier benchmark.

Decision (locked): name-only features for the baseline/deep-model cascade —
syscall name sequence only, no argument-level features (fd, res, data, ...).
Argument-rich features are deferred to a later enhancement pass once the
name-only cascade has a working baseline to compare against.
"""

THREAD_AWARE = True
NGRAM_LENGTH = 7        # paper default (roadmap Phase 3)
WINDOW_LENGTH = 100      # paper default (roadmap Phase 3) — StreamSum window
