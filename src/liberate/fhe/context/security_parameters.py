from scipy.interpolate import InterpolatedUnivariateSpline

# These are bits security levels, i.e., the 1^lambda measure.
security_levels = [128, 192, 256]

# This is the dimension of the cyclotomic moduli, in
# ℤ[X]/Φ𝑚(𝑋), and m = 2^l. Where n is the leading (biggest) power of X in the polynomial Φ𝑚(𝑋).
# Such that, Φ𝑚(𝑋) = X^n + 1, where n = m / 2.
cyclotomic_n = [1024, 2048, 4096, 8192, 16384, 32768]

# The following q is the moduli of a ring ℤq. Note that the numbers are given in log(q) values,
# where log in this context means log base 2.
#
# There are 2 sections in the standard documentation, namely pre- and post- quantum security.
# We separate them in respective dictionaries.
#
# Also, there are 3 different methods of sampling the messages according to respective distributions.
# Those are uniform, error, and (-1, 1) (ternary).
# We differentiate the message distribution by dictionary keys: 'uniform', 'error', and 'ternary'.

# This is the pre-quantum security requirements.
logq_preq = {}
logq_preq["uniform"] = [
    29,
    21,
    16,
    56,
    39,
    31,
    111,
    77,
    60,
    220,
    154,
    120,
    440,
    307,
    239,
    880,
    612,
    478,
]
logq_preq["error"] = [
    29,
    21,
    16,
    56,
    39,
    31,
    111,
    77,
    60,
    220,
    154,
    120,
    440,
    307,
    239,
    883,
    613,
    478,
]
logq_preq["ternary"] = [
    27,
    19,
    14,
    54,
    37,
    29,
    109,
    75,
    58,
    218,
    152,
    118,
    438,
    305,
    237,
    881,
    611,
    476,
]

# This is the post-quantum security requirements.
logq_postq = {}
logq_postq["uniform"] = [
    27,
    19,
    15,
    53,
    37,
    29,
    103,
    72,
    56,
    206,
    143,
    111,
    413,
    286,
    222,
    829,
    573,
    445,
]
logq_postq["error"] = [
    27,
    19,
    15,
    53,
    37,
    29,
    103,
    72,
    56,
    206,
    143,
    111,
    413,
    286,
    222,
    829,
    573,
    445,
]
logq_postq["ternary"] = [
    25,
    17,
    13,
    51,
    35,
    27,
    101,
    70,
    54,
    202,
    141,
    109,
    411,
    284,
    220,
    827,
    571,
    443,
]


# Partition q's by security levels.
def partitq(q):
    qlen = len(q)
    levlen = len(security_levels)
    grouped = [
        [q[i] for i in range(0 + lev, qlen, levlen)] for lev in range(levlen)
    ]
    by_sec_lev = {lev: grouped[l] for l, lev in enumerate(security_levels)}
    return by_sec_lev


# Gather up.
logq = {}
distributions = ["uniform", "error", "ternary"]
logq["pre_quantum"] = {
    distributions[disti]: partitq(logq_preq[dist])
    for disti, dist in enumerate(distributions)
}
logq["post_quantum"] = {
    distributions[disti]: partitq(logq_postq[dist])
    for disti, dist in enumerate(distributions)
}


def minimum_cyclotomic_order(
    q_bits, security_bits=128, quantum="post_quantum", distribution="uniform"
):
    assert quantum in [
        "pre_quantum",
        "post_quantum",
    ], "Wrong quantum security model!!!"
    assert distribution in ["uniform", "error", "ternary"]
    assert security_bits in [128, 192, 256]

    x = logq[quantum][distribution][security_bits]
    y = cyclotomic_n
    s = InterpolatedUnivariateSpline(x, y, k=1)
    return s(q_bits)


def maximum_qbits(
    L, security_bits=128, quantum="post_quantum", distribution="uniform"
):
    assert quantum in [
        "pre_quantum",
        "post_quantum",
    ], "Wrong quantum security model!!!"
    assert distribution in ["uniform", "error", "ternary"]
    assert security_bits in [128, 192, 256]

    x = cyclotomic_n
    y = logq[quantum][distribution][security_bits]
    s = InterpolatedUnivariateSpline(x, y, k=1)
    return s(L)
