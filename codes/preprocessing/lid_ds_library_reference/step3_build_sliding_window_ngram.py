"""
REFERENCE COPY — not written by us.

Original file: external/LID-DS/algorithms/features/impl/ngram.py
(part of the upstream LID-DS framework: https://github.com/LID-DS/LID-DS)

Included here, unmodified, purely for transparency/documentation, since
external/LID-DS is excluded from this repo via .gitignore (it's a 220MB+
nested git clone — see README's Environment section). This is the exact
code our preprocessing.py pipeline imports and depends on.

WHAT THIS DOES (step 3 of 3):
Takes the integer-encoded syscalls from step2, one at a time, and groups
them into fixed-length sliding windows (ngram_length=7 in our config —
see feature_config.py). Each new syscall pushes the window forward by
one position (the oldest syscall drops off, the new one is added), so
consecutive windows overlap by ngram_length-1. This is exactly the
"7-gram" in our preprocessed data/*_ngrams.csv.zip files — each row is
one output of this sliding window.

thread_aware=True (our config) keeps a SEPARATE window per thread_id, so
a window never mixes syscalls from two different threads of the same
process — each row is guaranteed to be one thread's real, uninterrupted
execution order.
"""
import typing
from collections import deque
from collections.abc import Iterable
from algorithms import features

from algorithms.building_block import BuildingBlock
from algorithms.features.impl.threadID import ThreadID
from dataloader.syscall import Syscall


class Ngram(BuildingBlock):
    """
    calculate ngram form a stream of system call features
    """

    def __init__(self, feature_list: list, thread_aware: bool, ngram_length: int):
        """
        feature_list: list of features the ngram should use
        thread_aware: True or False
        ngram_length: length of the ngram
        """
        super().__init__()
        self._ngram_buffer = {}
        self._thread_aware = thread_aware
        self._ngram_length = ngram_length
        self._deque_length = None
        self._dependency_list = []
        self._dependency_list.extend(feature_list)
        # Pre-bind get_result methods to avoid repeated attribute lookup
        self._dep_getters = [f.get_result for f in feature_list]
        # Concat strategy (compiled on first valid call)
        self._concat_mask = None

    def depends_on(self):
        return self._dependency_list

    def _calculate(self, syscall: Syscall):
        """
        writes the ngram into dependencies if its complete
        otherwise does not write into dependencies
        """
        results = []
        for getter in self._dep_getters:
            result = getter(syscall)
            if result is None:
                return None
            results.append(result)

        # Compile concat mask on first successful call
        if self._concat_mask is None:
            self._concat_mask = tuple(
                type(v) is not str and hasattr(v, '__iter__') for v in results
            )
            flat_len = sum(len(v) if m else 1 for m, v in zip(self._concat_mask, results))
            self._deque_length = self._ngram_length * flat_len

        # Build flat dependencies using pre-compiled mask
        if not any(self._concat_mask):
            deps = results
        else:
            deps = []
            for is_iter, v in zip(self._concat_mask, results):
                if is_iter:
                    deps.extend(v)
                else:
                    deps.append(v)

        thread_id = syscall.thread_id() if self._thread_aware else 0
        buf = self._ngram_buffer.get(thread_id)
        if buf is None:
            buf = deque(maxlen=self._deque_length)
            self._ngram_buffer[thread_id] = buf

        buf.extend(deps)
        if len(buf) == self._deque_length:
            return tuple(buf)
        return None


    def new_recording(self):
        """
        empty buffer so ngrams consist of same recording only
        """
        self._ngram_buffer = {}
