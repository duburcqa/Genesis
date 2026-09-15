import quadrants as qd


@qd.func
def qd_block_sum(value):
    """Sum value over the 32 lanes of a block, every lane receiving the bits lane 0 holds."""
    # The butterfly adds the same operands on the two lanes of every pair, but the compiler contracts the multiply that
    # produced a lane's own operand into that add, so the two lanes round differently, and a decision every lane takes
    # on the sum then diverges.
    return qd.simt.subgroup.broadcast(qd.simt.subgroup.reduce_all_add_tiled(value, 5), qd.u32(0))


@qd.func
def qd_block_min(value):
    """Minimum of value over the 32 lanes of a block, every lane receiving the bits lane 0 holds (see qd_block_sum)."""
    return qd.simt.subgroup.broadcast(qd.simt.subgroup.reduce_all_min_tiled(value, 5), qd.u32(0))


@qd.func
def qd_segment_add(tid, i_slot, n_valid, value, i_slot_prev, i_slot_next, sh_acc, i_row):
    """Segmented sum of value over the lanes of a 32-lane chunk sharing a slot, the tail lane of each segment adding
    the segment's total into entry i_row + i_slot of the shared array.

    i_slot_prev / i_slot_next are the slots of the neighboring lanes, -1 past the chunk's n_valid lanes.
    """
    is_head = 1
    if tid > 0 and i_slot_prev == i_slot:
        is_head = 0
    total = qd.simt.subgroup.segmented_reduce_add_tiled(value, is_head, 5)
    if tid < n_valid and i_slot >= 0 and i_slot_next != i_slot:
        sh_acc[i_row + i_slot] = sh_acc[i_row + i_slot] + total


@qd.func
def qd_segment_min(tid, i_slot, n_valid, value, i_slot_prev, i_slot_next, sh_min, i_row):
    """Segmented minimum of value over the lanes of a 32-lane chunk sharing a slot, the tail lane of each segment
    folding the segment's minimum into entry i_row + i_slot of the shared array (see qd_segment_add)."""
    is_head = 1
    if tid > 0 and i_slot_prev == i_slot:
        is_head = 0
    total = qd.simt.subgroup.segmented_reduce_min_tiled(value, is_head, 5)
    if tid < n_valid and i_slot >= 0 and i_slot_next != i_slot:
        sh_min[i_row + i_slot] = qd.min(sh_min[i_row + i_slot], total)
