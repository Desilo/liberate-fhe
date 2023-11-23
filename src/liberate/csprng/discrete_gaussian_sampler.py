# Sample from discrete gaussian distribution.
import math

import mpmath as mpm
import numpy as np
import torch

#  import discrete_gaussian_cuda
from . import discrete_gaussian_cuda


def build_CDT_binary_search_tree(security_bits=128, sigma=3.2):
    """Currently, ONLY the discrete gaussian sampling at the default input values
    is supported. That is equivalent to the 128 security measure."""
    # Set accuracy to 258 bits = (128 * 2) bits.
    # We will use higher 128 bits for the 128 security model,
    # Hence, retaining the 256 bits are safe for intermediate calculations
    # to carry over all the significant digits.

    mpm.mp.prec = security_bits * 2

    # Truncation boundary.
    # The minimum recommended by
    # Shi Bai, Adeline Langlois, Tancr`ede Lepoint, Damien Stehl ́e,
    # and Ron Ste- infeld. Improved security proofs in lattice-based cryptography:
    # Using the r ́enyi divergence rather than the statistical distance.,
    # as tau = 6 sigma.
    # We want the number tau to be the power of 2 in fact, since it makes the
    # binary tree search constant time. Using a larger tau the the minumum required
    # is no problem in fact and in terms of tree traversing it doesn't score a performance
    # hit because the tree will be balanced as a result.
    # So we calculate the smallest power of 2 bigger than the minimum tau as the number
    # of sampling points.
    sampling_power = math.ceil(math.log2(6 * sigma))
    num_sampling_points = 2**sampling_power
    sampling_points = list(range(num_sampling_points))

    # Calculate probabilities at sampling points.
    # Be careful when converting the python float to mpmath float.
    # No mormalization is done and the mpmath tries to retain the
    # bit pattern of the original float.
    # As a result, when you do mpm.mpf(3.2), you get
    # mpf('3.20000000000000017763568394002504646778106689453125').
    # As a workaround, we can do mpm.mpf('3.2') to get
    # mpf('3.200000000000000000000000000000000000000000000000000000000000000000000000000007')
    mp_sigma = mpm.mpf(str(sigma))
    mp_two = mpm.mpf("2")
    S = mp_sigma * mpm.sqrt(mp_two * mpm.pi)
    discrete_gaussian_prob = (
        lambda x: mpm.exp(-mpm.mpf(str(x)) ** 2 / (mp_two * mp_sigma**2)) / S
    )
    gaussian_prob_at_sampling_points = [
        discrete_gaussian_prob(x) for x in sampling_points
    ]

    # We need to halve the probability at 0.
    # We need to take into account the effect of symmetry, and we have
    # only calculated the probability for the half section.
    gaussian_prob_at_sampling_points[0] /= 2

    # Now, calculate the Cumulative Distribution Table.
    CDT = [0]
    for P in gaussian_prob_at_sampling_points:
        CDT.append(CDT[-1] + P)

    # At this point, we should end up with CDT[-1] which is very close to 0.5.
    # This makes sense because we are calculating the CDT of the half plane.
    # That reduces the effective bits in the CDT to 127, not 128.
    # This again makes sense because we need to reserve 1 bit for sign.

    # We need 128 bits integer representation of the CDT.
    CDT = [int(x * mp_two ** mpm.mpf(str(security_bits))) for x in CDT]

    # Chop the numbers down in a series of 64 bit integers.
    num_chops = security_bits // 64
    chopped_CDT = []
    mask = (1 << 64) - 1
    for chop in range(num_chops):
        chopped_share = [(x >> (64 * chop)) & mask for x in CDT]
        chopped_CDT.append(chopped_share)

    # Now we can put the chopped CDT into a numpy array.
    # All the numbers in the lists are representable by int64s.
    # We transpose the resulting array to make it configured as N x 2.
    CDT_table = np.array(chopped_CDT, dtype=np.uint64).T

    # We want to search through this table efficiently.
    # Build a binary tree.
    # Note that the last leaf is the sampled values.
    # The last leaf index will be calculated in-place at runtime,
    # Thus ommitted.
    tree_depth = sampling_power
    CDT_binary_tree = []
    for depth in range(tree_depth):
        num_nodes = 2**depth
        node_index_step = num_sampling_points // num_nodes
        first_node_index = num_sampling_points // num_nodes // 2
        node_indices = list(
            range(first_node_index, num_sampling_points, node_index_step)
        )
        CDT_binary_tree += node_indices
        # Use 1D expanded binary tree.
        # See https://en.wikipedia.org/wiki/Binary_tree#Arrays.
    btree = CDT_table[CDT_binary_tree]

    # Return the CType pointer together with the array.
    btree_size = btree.shape[0]
    btree_conti = np.ascontiguousarray(btree.T.ravel(), dtype=np.uint64)
    btree_ptr = btree_conti.__array_interface__["data"][0]

    # The returned tree has probably has 31 x 2 dimension.
    # The 31 for the number of nodes, and
    # the 2 for (lower 64 bits, higher 64 bits).
    return btree, btree_ptr, btree_size, tree_depth


def test_discrete_gaussian(N):
    btree, depth = build_CDT_binary_search_tree()

    rand = np.random.randint(0, 2**64, size=(N, 2), dtype=np.uint64)

    GE = lambda x_high, x_low, y_high, y_low: (
        ((x_high) > (y_high)) | (((x_high) == (y_high)) & ((x_low) >= (y_low)))
    )
    result = []
    for r in rand:
        jump = 1
        current = 0
        counter = 0

        sign_bit = int(r[0]) & 1
        r_high = int(r[0]) >> 1
        r_low = r[1]

        for j in range(depth):
            ge_flag = GE(
                r_high,
                r_low,
                btree[counter + current, 1],
                btree[counter + current, 0],
            )
            # ge_flag = (r_high > btree[counter+current, 1])

            # Update current location.
            current = 2 * current + int(ge_flag)

            # Update counter.
            counter += jump

            # Update jump
            jump *= 2

        sample = (sign_bit * 2 - 1) * current
        result.append(sample)

    return result
