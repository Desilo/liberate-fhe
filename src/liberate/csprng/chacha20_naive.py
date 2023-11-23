import binascii
import math
import os

import torch

torch.backends.cudnn.benchmark = True


@torch.jit.script
def roll(x: torch.Tensor, s: int) -> None:
    """
    x's dtype must be torch.int64.
    We are kind of forced to do this because
    1. pytorch doesn't support unit32, and
    2. >> doesn't move the sign bit.
    """
    mask = 0xFFFFFFFF
    right_shift = 32 - s
    down = (x & mask) >> right_shift
    # x <<=  s
    # x |= down
    x.__ilshift__(s).bitwise_or_(down).bitwise_and_(mask)


@torch.jit.script
def roll16(x: torch.Tensor) -> None:
    mask = 0xFFFFFFFF
    down = x >> 16
    x.__ilshift__(16).bitwise_or_(down).bitwise_and_(mask)


@torch.jit.script
def roll12(x: torch.Tensor) -> None:
    mask = 0xFFFFFFFF
    down = x >> 20
    x.__ilshift__(12).bitwise_or_(down).bitwise_and_(mask)


@torch.jit.script
def roll8(x: torch.Tensor) -> None:
    mask = 0xFFFFFFFF
    down = x >> 24
    x.__ilshift__(8).bitwise_or_(down).bitwise_and_(mask)


@torch.jit.script
def roll7(x: torch.Tensor) -> None:
    mask = 0xFFFFFFFF
    down = x >> 25
    x.__ilshift__(7).bitwise_or_(down).bitwise_and_(mask)


@torch.jit.script
def QR(x: torch.Tensor, a: int, b: int, c: int, d: int) -> None:
    """
    The CHACHA quarter round.
    """
    mask = 0xFFFFFFFF

    x[a].add_(x[b])
    x[a].bitwise_and_(mask)
    x[d].bitwise_xor_(x[a])
    roll16(x[d])

    x[c].add_(x[d])
    x[c].bitwise_and_(mask)
    x[b].bitwise_xor_(x[c])
    roll12(x[b])

    x[a].add_(x[b])
    x[a].bitwise_and_(mask)
    x[d].bitwise_xor_(x[a])
    roll8(x[d])

    x[c].add_(x[d])
    x[c].bitwise_and_(mask)
    x[b].bitwise_xor_(x[c])
    roll7(x[b])


@torch.jit.script
def one_round(x: torch.Tensor) -> None:
    # Odd round.
    QR(x, 0, 4, 8, 12)
    QR(x, 1, 5, 9, 13)
    QR(x, 2, 6, 10, 14)
    QR(x, 3, 7, 11, 15)
    # Even round.
    QR(x, 0, 5, 10, 15)
    QR(x, 1, 6, 11, 12)
    QR(x, 2, 7, 8, 13)
    QR(x, 3, 4, 9, 14)


@torch.jit.script
def increment_counter(state: torch.Tensor, inc: int) -> None:
    state[12] += inc
    state[13] += state[12] >> 32
    state[12] = state[12] & 0xFFFFFFFF


@torch.jit.script
def chacha20(state: torch.Tensor) -> torch.Tensor:
    x = state.clone()

    for _ in range(10):
        one_round(x)

    # Return the random bytes.
    return (x + state) & 0xFFFFFFFF


class chacha20_naive:
    def __init__(
        self, size, seed=None, nonce=None, count_step=1, device="cuda:0"
    ):
        self.size = size
        self.device = device
        self.count_step = count_step

        # expand 32-byte k.
        # This is 1634760805, 857760878, 2036477234, 1797285236.
        str2ord = lambda s: sum([2 ** (i * 8) * c for i, c in enumerate(s)])
        self.nothing_up_my_sleeve = torch.tensor(
            [
                str2ord(b"expa"),
                str2ord(b"nd 3"),
                str2ord(b"2-by"),
                str2ord(b"te k"),
            ],
            device=device,
            dtype=torch.int64,
        )

        # Prepare the state tensor.
        self.state_size = (*self.size, 16)
        self.state_buffer = torch.zeros(
            16, (math.prod(self.size)), dtype=torch.int64, device=self.device
        )

        # The ind counter.
        self.ind = torch.arange(
            0, math.prod(self.size), self.count_step, device=device
        )

        # Increment is the number of indices.
        self.inc = math.prod(self.size)

        self.initialize_state(seed, nonce)

        # Capture computation graph.
        self.capture()

    def initialize_state(self, seed=None, nonce=None):
        # Generate seed if necessary.
        self.key(seed)

        # Generate nonce if necessary.
        self.generate_nonce(nonce)

        # Zero out the state.
        self.state_buffer.zero_()

        # Set the counter.
        self.state_buffer[12, :] = self.ind

        # Set the expand 32-bye k
        self.state_buffer[0:4, :] = self.nothing_up_my_sleeve[:, None]

        # Set the seed.
        self.state_buffer[4:12, :] = self.seed[:, None]

        # Fill in nonce.
        self.state_buffer[14:, :] = self.nonce[:, None]

    def key(self, seed=None):
        # 256bits seed as a key.
        if seed is None:
            # 256bits key as a seed,
            nbytes = 32
            part_bytes = 4
            n_keys = nbytes // part_bytes
            hex2int = lambda x, nbytes: int(binascii.hexlify(x), 16)
            self.seed = torch.tensor(
                [
                    hex2int(os.urandom(part_bytes), part_bytes)
                    for _ in range(n_keys)
                ],
                device=self.device,
                dtype=torch.int64,
            )
        else:
            self.seed = torch.tensor(
                seed, device=self.device, dtype=torch.int64
            )

    def generate_nonce(self, nonce):
        # nonce is 64bits.
        if nonce is None:
            # 256bits key as a seed,
            nbytes = 8
            part_bytes = 4
            n_keys = nbytes // part_bytes
            hex2int = lambda x, nbytes: int(binascii.hexlify(x), 16)
            self.nonce = torch.tensor(
                [
                    hex2int(os.urandom(part_bytes), part_bytes)
                    for _ in range(n_keys)
                ],
                device=self.device,
                dtype=torch.int64,
            )
        else:
            self.nonce = torch.tensor(
                nonce, device=self.device, dtype=torch.int64
            )

    def capture(self, warmup_periods=3, fuser="fuser1"):
        with torch.cuda.device(self.device):
            # Reserve ample amount of excution cache.
            torch.jit.set_fusion_strategy([("STATIC", 100)])

            # Output buffer.
            self.out = torch.zeros_like(self.state_buffer)

            # Warm up.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.jit.fuser(fuser):
                for _ in range(warmup_periods):
                    self.out = chacha20(self.state_buffer)
            torch.cuda.current_stream().wait_stream(s)

            # Capture.
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self.out = chacha20(self.state_buffer)

    def step(self):
        increment_counter(self.state_buffer, self.inc)

    def randbyte(self):
        self.g.replay()
        self.step()
        return self.out.clone()
