"""Sort of 64-bit keys per segment, with an optional value moved along each key.

The keys of segment i_s sit at keys[segments_start[i_s]:segments_start[i_s + 1]], and the keys of a segment exceed the
keys of every earlier segment, the segment index leading their bits. Sorting each segment is then the same as sorting
every key at once, which lets the device-wide sort of quadrants serve a set of many keys on GPU, while the sorts per
segment serve small segments and the host backend.
"""

import math
import os
from enum import IntEnum
from typing import NamedTuple

import quadrants as qd
from quadrants.algorithms import sort, sort_scratch_slots

import genesis as gs
import genesis.utils.linalg as lu
from genesis.utils.data import Data, DataKind, Label, Size, StaticConfig, field
from genesis.utils.misc import get_gpu_core_count


class SEGMENT_SORT_KIND(IntEnum):
    """The sort ordering the keys of a segment sort.

    See genesis.utils.sort.
    """

    # The device-wide radix sort of quadrants over every key of the set (GPU, large segments)
    DEVICE_RADIX = 0
    # The rank of each key among the keys of its segment (small segments)
    RANK = 1
    # A radix sort per segment, its chunks in parallel (CPU, large segments)
    SEGMENT_RADIX = 2


class SegmentSortStaticConfig(StaticConfig):
    """The compile-time shape of a sort of 64-bit keys per segment.

    See genesis.utils.sort.
    """

    # The bits a sort must order, a multiple of 16 up to 64
    end_bit: int
    # Scan depth of the device-wide radix sort (see quadrants.algorithms.sort), 1 to 8
    log256_max_n: int
    # SEGMENT_SORT_KIND
    sort_kind: int
    # Whether each key carries a value the sort moves along
    has_values: bool


# Sizes of a sort per segment: its segments, its keys, the slots of the device-wide radix sort and the chunks of the
# radix sort per segment
SORT_SEGMENTS = Size("sort_segments")
SORT_KEYS = Size("sort_keys")
SORT_SLOTS = Size("sort_slots")
SORT_CHUNKS = Size("sort_chunks")
# The values moved along the keys, and the scratch of each sort (see SEGMENT_SORT_KIND)
SORT_VALUES = Label("sort_values")
DEVICE_RADIX_SORT = Label("device_radix_sort")
SEGMENT_RADIX_SORT = Label("segment_radix_sort")


class SegmentSortBuffer(Data, kind=DataKind.SCRATCH, shape=(SORT_KEYS,)):
    """The keys of a sort per segment, their values, the segment bounds and the scratch of the sort.

    See genesis.utils.sort. The first n_keys keys are sorted, segment i_s holding the keys from segments_start[i_s] to
    segments_start[i_s + 1].
    """

    keys: qd.Tensor = field(qd.u64)
    values: qd.Tensor = field(int, labels=SORT_VALUES)
    keys_scratch: qd.Tensor = field(qd.u64)
    values_scratch: qd.Tensor = field(int, labels=SORT_VALUES)
    segments_start: qd.Tensor = field(int, shape=(SORT_SEGMENTS + 1,))
    sort_scratch: qd.Tensor = field(qd.u32, shape=(SORT_SLOTS,), labels=DEVICE_RADIX_SORT)
    sort_hist: qd.Tensor = field(int, shape=(SORT_SEGMENTS, SORT_CHUNKS, 256), labels=SEGMENT_RADIX_SORT)
    n_keys: qd.Tensor = field(int, shape=())


# The rank sort, quadratic per segment, beats the radix sort up to a key count per segment. On the CPU the segments sort
# serially, so the bound holds at every segment count. On the GPU the passes of every segment run together and the
# crossover sits near one key per core for one segment, shrinking with the cube root of the segment count:
# n_segments * n_segment_keys**3 stays under the cube of this fraction of the core count, set under the crossover of
# every device measured.
RANK_SORT_MAX_KEYS_CPU = 256
RANK_SORT_KEYS_PER_CORE = 0.75
# Most chunks of the keys of one segment the host radix sort counts and scatters in parallel (see func_sort_segments).
# The chunks bring parallelism to a set of few segments. A set of many segments is parallel over its segments already,
# and every chunk costs a histogram of 256 bins per pass.
SORT_MAX_CHUNKS = 64


class SegmentSortData(NamedTuple):
    """A sort of keys per segment: the static config of its kernels, and what its buffer is allocated from.

    A struct holding the buffer (see SegmentSortBuffer) takes its sizes under the name of its member, and its unused
    arrays among its own.
    """

    config: SegmentSortStaticConfig
    sizes: dict[str, int]
    unused: dict[Label, bool]


def get_segment_sort_data(
    n_segments: int, n_keys: int, n_segment_keys_max: int, key_bits: int, has_values: bool
) -> SegmentSortData:
    """Return the static config of a sort of keys per segment, and what its buffer is allocated from.

    The sort orders n_keys keys over n_segments segments. A segment holds n_segment_keys_max keys at most, and every
    key, its segment index included, fits in key_bits bits. The sort takes the rank sort for small segments, a radix
    sort per segment on the host backend, and the device-wide radix sort of quadrants otherwise.
    """
    if key_bits > 64:
        gs.raise_exception(f"A segment sort orders 64-bit keys, its keys need {key_bits} bits.")

    # The radix sort orders a positive even multiple of 8 bits
    end_bit = min(64, 16 * math.ceil(key_bits / 16))
    log256_max_n = 1
    while 256**log256_max_n < n_keys:
        log256_max_n += 1
    if gs.backend == gs.cpu:
        is_rank_sort = n_segment_keys_max <= RANK_SORT_MAX_KEYS_CPU
    else:
        max_keys = RANK_SORT_KEYS_PER_CORE * get_gpu_core_count()
        is_rank_sort = n_segments * n_segment_keys_max**3 <= max_keys**3
    # Outside performance mode the radix sort serves every set, with the widest keys and a scan depth reaching any key
    # count a device holds. The choice of the rank sort stays per set, its single pass being much faster on small sets.
    if gs.use_ndarray:
        end_bit = 64
        log256_max_n = max(log256_max_n, 4)
    if is_rank_sort:
        sort_kind = SEGMENT_SORT_KIND.RANK
    elif gs.backend == gs.cpu:
        sort_kind = SEGMENT_SORT_KIND.SEGMENT_RADIX
    else:
        sort_kind = SEGMENT_SORT_KIND.DEVICE_RADIX
    sort_config = SegmentSortStaticConfig(
        end_bit=end_bit, log256_max_n=log256_max_n, sort_kind=sort_kind, has_values=has_values
    )

    n_sort_scratch = sort_scratch_slots(n_keys, log256_max_n) if sort_kind == SEGMENT_SORT_KIND.DEVICE_RADIX else 0
    n_sort_chunks = max(1, min(SORT_MAX_CHUNKS, (os.cpu_count() or 1) // n_segments))
    return SegmentSortData(
        sort_config,
        dict(sort_segments=n_segments, sort_keys=n_keys, sort_slots=n_sort_scratch, sort_chunks=n_sort_chunks),
        {
            SORT_VALUES: not has_values,
            DEVICE_RADIX_SORT: sort_kind != SEGMENT_SORT_KIND.DEVICE_RADIX,
            SEGMENT_RADIX_SORT: sort_kind != SEGMENT_SORT_KIND.SEGMENT_RADIX,
        },
    )


@qd.func(requires_top_level=True)
def func_sort_segments(sort_buffer: SegmentSortBuffer, sort_config: qd.template()):
    """Sort the keys of every segment in increasing order, with their values.

    The caller writes the keys, their values, the segment bounds and the key count first. Equal keys keep their input
    order, so the result is a pure function of the input.
    """
    if qd.static(sort_config.sort_kind == SEGMENT_SORT_KIND.DEVICE_RADIX):
        # Without values, the keys stand in for the placeholder values the sort leaves untouched
        sort(
            sort_buffer.keys,
            sort_buffer.keys_scratch,
            sort_buffer.values if qd.static(sort_config.has_values) else sort_buffer.keys,
            sort_buffer.values_scratch if qd.static(sort_config.has_values) else sort_buffer.keys_scratch,
            sort_buffer.sort_scratch,
            sort_buffer.n_keys,
            key_dtype=qd.u64,
            has_values=qd.static(sort_config.has_values),
            end_bit=qd.static(sort_config.end_bit),
            log256_max_n=qd.static(sort_config.log256_max_n),
        )
    elif qd.static(sort_config.sort_kind == SEGMENT_SORT_KIND.RANK):
        # Each key ranks itself among the keys of its segment, found by a binary search over the segment bounds
        n_segments = sort_buffer.segments_start.shape[0] - 1
        n_search_steps = 32 - qd.cast(qd.math.clz(qd.u32(n_segments)), gs.qd_int)
        for i_k in range(sort_buffer.n_keys[()]):
            i_seg_lo = 0
            i_seg_hi = n_segments
            for _ in range(n_search_steps):
                if i_seg_hi - i_seg_lo > 1:
                    i_seg_mid = (i_seg_lo + i_seg_hi) // 2
                    if sort_buffer.segments_start[i_seg_mid] <= i_k:
                        i_seg_lo = i_seg_mid
                    else:
                        i_seg_hi = i_seg_mid
            i_k_start = sort_buffer.segments_start[i_seg_lo]
            key = sort_buffer.keys[i_k]
            i_k_ = 0
            for i_o in range(i_k_start, sort_buffer.segments_start[i_seg_lo + 1]):
                key_o = sort_buffer.keys[i_o]
                if key_o < key or (key_o == key and i_o < i_k):
                    i_k_ += 1
            sort_buffer.keys_scratch[i_k_start + i_k_] = key
            if qd.static(sort_config.has_values):
                sort_buffer.values_scratch[i_k_start + i_k_] = sort_buffer.values[i_k]
        for i_k in range(sort_buffer.n_keys[()]):
            sort_buffer.keys[i_k] = sort_buffer.keys_scratch[i_k]
            if qd.static(sort_config.has_values):
                sort_buffer.values[i_k] = sort_buffer.values_scratch[i_k]
    else:
        # A least-significant-digit radix sort of eight bits per pass on the host backend, the keys of each segment
        # split over chunks: the chunks count their digits and scatter their keys in parallel, one thread per segment
        # turning the counts into offsets in between. The passes ping-pong between the keys and the scratch and end in
        # the keys. The passes unroll statically: a loop of the kernel is parallel at its top level alone, so a runtime
        # loop over the passes would serialize the chunks.
        n_segments = sort_buffer.segments_start.shape[0] - 1
        n_sort_chunks = sort_buffer.sort_hist.shape[1]
        for i_pass in qd.static(range(qd.static(sort_config.end_bit) // 8)):
            shift = 8 * i_pass
            for i_seg, i_chunk in qd.ndrange(n_segments, n_sort_chunks):
                for i_digit in range(256):
                    sort_buffer.sort_hist[i_seg, i_chunk, i_digit] = 0
                i_k_start = sort_buffer.segments_start[i_seg]
                n_segment_keys = sort_buffer.segments_start[i_seg + 1] - i_k_start
                i_k_start, i_k_end = lu.func_chunk_range(i_k_start, i_chunk, n_segment_keys, n_sort_chunks)
                for i_k in range(i_k_start, i_k_end):
                    key = qd.u64(0)
                    if qd.static(i_pass % 2 == 0):
                        key = sort_buffer.keys[i_k]
                    else:
                        key = sort_buffer.keys_scratch[i_k]
                    digit = qd.i32((key >> shift) & qd.u64(0xFF))
                    sort_buffer.sort_hist[i_seg, i_chunk, digit] += 1
            for i_seg in range(n_segments):
                total = sort_buffer.segments_start[i_seg]
                for i_digit in range(256):
                    for i_chunk in range(n_sort_chunks):
                        count = sort_buffer.sort_hist[i_seg, i_chunk, i_digit]
                        sort_buffer.sort_hist[i_seg, i_chunk, i_digit] = total
                        total += count
            # The keys scattered in order within each chunk, so that the pass is stable
            for i_seg, i_chunk in qd.ndrange(n_segments, n_sort_chunks):
                i_k_start = sort_buffer.segments_start[i_seg]
                n_segment_keys = sort_buffer.segments_start[i_seg + 1] - i_k_start
                i_k_start, i_k_end = lu.func_chunk_range(i_k_start, i_chunk, n_segment_keys, n_sort_chunks)
                for i_k in range(i_k_start, i_k_end):
                    key = qd.u64(0)
                    if qd.static(i_pass % 2 == 0):
                        key = sort_buffer.keys[i_k]
                    else:
                        key = sort_buffer.keys_scratch[i_k]
                    digit = qd.i32((key >> shift) & qd.u64(0xFF))
                    i_dst = sort_buffer.sort_hist[i_seg, i_chunk, digit]
                    sort_buffer.sort_hist[i_seg, i_chunk, digit] = i_dst + 1
                    if qd.static(i_pass % 2 == 0):
                        sort_buffer.keys_scratch[i_dst] = key
                        if qd.static(sort_config.has_values):
                            sort_buffer.values_scratch[i_dst] = sort_buffer.values[i_k]
                    else:
                        sort_buffer.keys[i_dst] = key
                        if qd.static(sort_config.has_values):
                            sort_buffer.values[i_dst] = sort_buffer.values_scratch[i_k]
