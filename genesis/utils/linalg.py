import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class


@qd.func
def mat_mul(A: qd.Tensor, B: qd.Tensor, res: qd.Tensor, n, m, l, i_b):
    """
    Performs matrix multiplication between matrices A and B and stores the result in res.

    Args:
        A (qd.field): The first matrix of shape (n, m, B).
        B (qd.field): The second matrix of shape (m, l, B).
        res (qd.field): The result matrix of shape (n, l, B).
        n (int): The number of rows in matrix A and res.
        m (int): The number of columns in matrix A and rows in matrix B.
        l (int): The number of columns in matrix B and res.
        i_b (int): batch index.
    """
    for i in range(n):
        for j in range(l):
            res[i, j, i_b] = 0
            for k in range(m):
                res[i, j, i_b] += A[i, k, i_b] * B[k, j, i_b]


@qd.func
def mat_mul_vec(mat: qd.Tensor, vec: qd.Tensor, res: qd.Tensor, n, m, i_b):
    for i in range(n):
        res[i, i_b] = 0
        for j in range(m):
            res[i, i_b] += mat[i, j, i_b] * vec[j, i_b]
    return res


@qd.func
def mat_inverse(mat: qd.Tensor, L: qd.Tensor, U: qd.Tensor, y: qd.Tensor, res: qd.Tensor, n, i_b):
    """
    Inverse via LU decomposition
    """
    # LU decomposition
    for i in range(n):
        L[i, i, i_b] = 1
        for j in range(i, n):
            tmp = mat[i, j, i_b]
            for k in range(i):
                tmp -= L[i, k, i_b] * U[k, j, i_b]
            U[i, j, i_b] = tmp

        for j in range(i + 1, n):
            tmp = mat[j, i, i_b]
            for k in range(i):
                tmp -= L[j, k, i_b] * U[k, i, i_b]
            L[j, i, i_b] = tmp / U[i, i, i_b]

    # Forward and backward substitution for each column k of the identity matrix
    for k in range(n):
        # Forward substitution
        for i in range(n):
            tmp = gs.qd_float(0.0)
            for j in range(i):
                tmp += L[i, j, i_b] * y[k, j, i_b]
            if i == k:
                y[k, i, i_b] = 1 - tmp
            else:
                y[k, i, i_b] = -tmp

        # Backward substitution
        for i_ in range(n):
            i = n - 1 - i_
            tmp = gs.qd_float(0.0)
            for j in range(i + 1, n):
                tmp += U[i, j, i_b] * res[j, k, i_b]
            res[i, k, i_b] = (y[k, i, i_b] - tmp) / U[i, i, i_b]


@qd.func
def mat_add(A, B, n, m, i_b):
    for i in range(n):
        for k in range(m):
            A[i, k, i_b] += B[i, k, i_b]


@qd.func
def mat_transpose(A: qd.Tensor, B: qd.Tensor, n, m, i_b):
    for i in range(n):
        for k in range(m):
            B[k, i, i_b] = A[i, k, i_b]


@qd.func
def mat_add_eye(A: qd.Tensor, x, n, i_b):
    for i in range(n):
        A[i, i, i_b] += x


@qd.func
def mat_mask(A, mask, n, m, i_b):
    for i in range(n):
        for j in range(m):
            A[i, j, i_b] *= mask[i]


@qd.func
def func_chunk_range(i_start: int, i_chunk: int, n_items: int, n_chunks: int):
    """Return the bounds of one of the contiguous chunks a range of items splits into for a fixed-order sum.

    The items i_start to i_start + n_items split into n_chunks chunks of equal size but the last, chunk i_chunk being
    the items between the returned bounds.
    """
    chunk_size = (n_items + n_chunks - 1) // n_chunks
    return i_start + qd.min(i_chunk * chunk_size, n_items), i_start + qd.min((i_chunk + 1) * chunk_size, n_items)


@qd.func(requires_top_level=True)
def func_reduce_terms(sums: array_class.FixedOrderSums, mask: qd.template()):
    """
    Sum in order the terms of each environment whose mask is set over contiguous chunks of its items.

    See FixedOrderSums in genesis/utils/array_class.py. The chunks are summed in parallel.
    """
    for i_chunk, k, i_b in qd.ndrange(sums.partials.shape[0], sums.terms.shape[1], sums.terms.shape[2]):
        if mask[i_b]:
            value_sum = gs.qd_float(0.0)
            i_start, i_end = func_chunk_range(0, i_chunk, sums.terms.shape[0], sums.partials.shape[0])
            for i in range(i_start, i_end):
                value_sum += sums.terms[i, k, i_b]
            sums.partials[i_chunk, k, i_b] = value_sum


@qd.func
def func_sum_chunks(i_b: int, k: int, sums: array_class.FixedOrderSums):
    """Sum in order the partial sums of the sum `k` of an environment (see FixedOrderSums in array_class.py)."""
    value_sum = gs.qd_float(0.0)
    for i_chunk in range(sums.partials.shape[0]):
        value_sum += sums.partials[i_chunk, k, i_b]
    return value_sum
