import numpy as np
from matplotlib import pyplot as plt


def random_complex_array(
    n: int = 2**8,
    amin: int = -(2**20),
    amax: int = 2**20,
    decimal_exponent: int = 10,
):
    base = 10**decimal_exponent
    a = np.random.randint(amin * base, amax * base, n) / base
    b = np.random.randint(amin * base, amax * base, n) / base
    ret = a + b * 1j
    return ret


def check_errors(test_message, test_message_dec, idx=10, title="errors"):
    errs = test_message_dec - test_message
    plt.figure(figsize=(16, 9))
    plt.plot(errs)
    plt.grid()
    plt.title(title)
    plt.show()

    print("============================================================")
    for x, y in zip(test_message[:idx], test_message_dec[:idx]):
        print(f"{x.real:19.10f} | {y.real:19.10f} | {(y - x).real:14.10f}")
    print("============================================================")
    print(f"mean\t=\t{errs.mean():10.15f}")
    print(f"std\t=\t{errs.std():10.15f}")
    print(f"max err\t=\t{abs(errs).max().real:10.15f}")
    print(f"min err\t=\t{abs(errs).min().real:10.15f}")


def absmax_error(x, y):
    if type(x[0]) is np.complex128 and type(y[0]) is np.complex128:
        r = np.abs(x.real - y.real).max() + np.abs(x.imag - y.imag).max() * 1j
    else:
        r = np.abs(np.array(x) - np.array(y)).max()
    return r
