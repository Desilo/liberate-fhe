import numpy as np
import torch


# ---------------------------------------------------------------
# Permutation.
# ---------------------------------------------------------------


def circular_shift_permutation(N, shift=1):
    left = np.roll(np.arange(N // 2), shift)
    right = np.roll(np.arange(N // 2), -shift) + N // 2
    return np.concatenate([left, right])


def canon_permutation(N, k=1, verbose=False):
    """
    Permutes the coefficients of the lattice basis that yields correctly the permutation
    of the decoded message.

    The canonical permutation is defined as mu_p(n) = pn mod M where p is coprime with M,
    where p=2*k+1.
    """
    M = 2 * N
    p = int(2 * k + 1)  # Make sure p is an integer.
    n = np.arange(M)  # n starts from 0.
    pn = p * n % M
    if verbose:
        print(f"Canonical permutation for p={p} is\n{pn}")
    return pn


def canon_permutation_torch(N, k=1, device="cuda:0", verbose=False):
    """
    Permutes the coefficients of the lattice basis that yields correctly the permutation
    of the decoded message.

    The canonical permutation is defined as mu_p(n) = pn mod M where p is coprime with M,
    where p=2*k+1.
    """
    M = N * 2
    p = int(2 * k + 1)  # Make sure p is an integer.
    n = torch.arange(N, device=device)  # n starts from 0.
    pn = p * n % M
    if verbose:
        print(f"Canonical permutation for p={p} is\n{pn}")
    return pn


def fold_permutation(N, p, verbose=False):
    """
    In application to crypto, we fold the FFT at Nyquist.

    Inverse FFT results in selection of alternating elements.
    Folding should correct the indices of the permutation according to the
    folding rule.

    For example, 1->0, 3->1, 5->2, and so on.
    """
    fold_p = (p[1::2] - 1) // 2
    if verbose:
        print(f"Folding\n{p}\nresulted in\n{fold_p}.")
    return fold_p


def conjugate_permutation(p, q):
    """
    Conjugate permutations p and q by stacking p on top of q.

    Permutations p and q must share the same cycle structures.
    """
    # Calculate cycles.
    pc = permutation_cycles(p)
    qc = permutation_cycles(q)

    # Check if the cycle structures match.
    cs1 = [len(c) for c in pc]
    cs2 = [len(c) for c in qc]
    assert (
        cs1 == cs2
    ), "Cycle structures of permutations must match for a conjugate to exist!!!"

    # Expand cycles.
    pe = np.array([i for c in pc for i in c])
    qe = np.array([i for c in qc for i in c])

    # Move slots.
    r = np.zeros_like(p)
    r[qe] = pe

    # Return.
    return r


def permutation_cycles(perm):
    """
    Transform a plain permutation into a composition of cycles.
    """
    pi = {i: perm[i] for i in range(len(perm))}
    cycles = []
    while pi:
        elem0 = next(iter(pi))  # arbitrary starting element
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break
        cycles.append(cycle)
    return cycles


def inverse_permutation(p, verbose=False):
    """
    Calculates the inverse permutation.
    """
    N = len(p)
    ind = np.arange(N)
    ip = ind[np.argsort(p)]
    if verbose:
        print(f"The inverse of permutation\n{p}\nis\n{ip}.")
    return ip


# ---------------------------------------------------------------
# Negacyclic fft.
# ---------------------------------------------------------------


def expand2conjugate(m):
    return torch.concat([m, torch.flipud(torch.conj(m))])


def generate_twister(N, device="cuda:0"):
    expr = (
        -1j
        * torch.pi
        * torch.arange(N, device=device, dtype=torch.float64)
        / N
    )
    return torch.exp(expr)


def generate_skewer(N, device="cuda:0"):
    expr = (
        1j * torch.pi * torch.arange(N, device=device, dtype=torch.float64) / N
    )
    skew = torch.exp(expr)
    return skew


def m2poly(m, twister, norm="backward"):
    """
    m is the message and this function turns the message into
    polynomial coefficients.
    The message must be expanded mirrored in conjugacy.
    """

    # Run fft and multiply twister.
    ffted = torch.fft.fft(m, norm=norm)

    # Twist.
    twisted = ffted * twister

    # Return the real part.
    return twisted.real


def poly2m(poly, skewer, norm="backward"):
    """
    poly is the polynomial coefficients and this function turns the coefficients
    into a plain message.
    """

    # Multiply skewer.
    t = poly * skewer

    # Recover.
    recovered = torch.fft.ifft(t, norm=norm)

    # Return the real part.
    return recovered


# ---------------------------------------------------------------
# Utilities.
# ---------------------------------------------------------------

perm_cache = {}
twister_cache = {}
skewer_cache = {}


def prepost_perms(N, device="cuda:0"):
    circ_shift = circular_shift_permutation(N)
    canon_perm = canon_permutation(N)
    fold_perm = fold_permutation(N, canon_perm)
    post_perm = conjugate_permutation(circ_shift, fold_perm)
    pre_perm = inverse_permutation(post_perm)[: N // 2]

    post_perm = torch.from_numpy(post_perm).to(device)
    pre_perm = torch.from_numpy(pre_perm).to(device)
    return pre_perm, post_perm


def pre_permute(m, pre_perm):
    """
    Input m must be a torch tensor.
    """
    N = m.size(-1)
    permed_m = torch.zeros((N * 2,), dtype=m.dtype, device=m.device)
    permed_m[pre_perm] = m
    conj_permed_m = permed_m + permed_m.conj().flip(0)
    return conj_permed_m


def post_permute(m, post_perm):
    """
    Input m must be a torch tensor.
    """
    permed_m = torch.zeros_like(m)
    permed_m[post_perm] = m
    return permed_m


def rotate(m, delta):
    N = m.size(-1)
    C = m.numel() // N

    shift = delta % N
    leap = (3**shift - 1) // 2 % (N * 2)

    if (N, leap, m.device) in perm_cache.keys():
        perm = perm_cache[(N, leap, m.device)]
    else:
        perm = canon_permutation_torch(N, leap, device=m.device)
        perm_cache[(N, leap, m.device)] = perm

    perm_folded = perm % N
    perm_sign = (-1) ** (perm // N)

    # Permute!
    # We want to both be capable of 2D and 1D tensors.
    # Use view.
    rot_m = torch.zeros_like(m)
    rot_m.view(C, N).T[perm_folded] = (perm_sign * m).view(C, N).T

    return rot_m


def conjugate(m):
    N = m.size(-1)
    C = m.numel() // N

    leap = N - 1

    if (N, leap, m.device) in perm_cache.keys():
        perm = perm_cache[(N, leap, m.device)]
    else:
        perm = canon_permutation_torch(N, leap, device=m.device)
        perm_cache[(N, leap, m.device)] = perm

    perm_folded = perm % N
    perm_sign = (-1) ** (perm // N)

    # Permute!
    # We want to both be capable of 2D and 1D tensors.
    # Use view.
    rot_m = torch.zeros_like(m)
    rot_m.view(C, N).T[perm_folded] = (perm_sign * m).view(C, N).T

    return rot_m


def encode(
    m,
    rng=None,
    scale=2**40,
    deviation=1.0,
    device="cuda:0",
    norm="forward",
    return_without_scaling=False,
):
    N = len(m) * 2
    if (N, device) in perm_cache.keys():
        pre_perm, post_perm = perm_cache[(N, device)]
    else:
        pre_perm, post_perm = prepost_perms(N, device=device)
        perm_cache[(N, device)] = (pre_perm, post_perm)

    mm = torch.tensor(m * deviation).to(device)  # check dtype m * deviation
    mm = pre_permute(mm, pre_perm)

    if (N, device) in twister_cache.keys():
        twister = twister_cache[N, device]
    else:
        twister = generate_twister(N, device)
        twister_cache[N, device] = twister

    if return_without_scaling:
        return m2poly(mm, twister, norm)
    else:
        mm = m2poly(mm, twister, norm) * np.float64(scale)
        return rng.randround(mm)


def decode(
    m,
    scale=2**40,
    correction=1.0,
    norm="forward",
    return_without_scaling=False,
):
    N = len(m)
    device = m.device.type + ":" + str(m.device.index)
    if (N, device) in perm_cache.keys():
        pre_perm, post_perm = perm_cache[(N, device)]
    else:
        pre_perm, post_perm = prepost_perms(N, device=device)
        perm_cache[(N, device)] = (pre_perm, post_perm)

    if (N, device) in skewer_cache.keys():
        skewer = skewer_cache[N, device]
    else:
        skewer = generate_skewer(N, device)
        skewer_cache[N, device] = skewer

    if return_without_scaling:
        mm = poly2m(m, skewer, norm=norm)
        mm = post_permute(mm, post_perm)
        return mm
    else:
        mm = poly2m(m, skewer, norm=norm) / scale * correction
        mm = post_permute(mm, post_perm)
        return mm
