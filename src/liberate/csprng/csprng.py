import binascii
import os

import numpy as np
import torch

from . import (
    chacha20_cuda,
    discrete_gaussian_cuda,
    randint_cuda,
    randround_cuda,
)
from .discrete_gaussian_sampler import build_CDT_binary_search_tree

torch.backends.cudnn.benchmark = True


class Csprng:
    def __init__(
        self,
        num_coefs=2**15,
        num_channels=[8],
        num_repeating_channels=2,
        sigma=3.2,
        devices=None,
        seed=None,
        nonce=None,
    ):
        """N is the length of the polynomial, and C is the number of RNS channels.
        procure the maximum (at level zero, special multiplication) at initialization.
        """

        # This CSPRNG class generates
        # 1. num_coefs x (num_channels + num_repeating_channels) uniform distributed
        #    random numbers at max. num_channels can be reduced down according
        #    to the input q at the time of generation.
        #    The numbers generated ri are 0 <= ri < qi.
        #    Seeds in the repeated channels are the same, and hence in those
        #    channels, the generated numbers are the same across GPUs.
        #    Generation of the repeated random numbers is optional.
        #    The same function can be used to generate ranged random integers
        #    in a fixed range. Again, generation of the repeated numbers is optional.
        # 2. Generation of Discrete Gaussian random numbers. The numbers can be generated
        #    in the non-repeating channels (with the maximum number of channels num_channels),
        #    or in the repeating channels (with the maximum number of
        #    channels num_repeating_channels, where in the most typical scenario is same as 1).

        self.num_coefs = num_coefs
        self.num_channels = num_channels
        self.num_repeating_channels = num_repeating_channels
        self.sigma = sigma

        # Set up GPUs.
        # By default, use all the available GPUs on the system.
        if devices is None:
            gpu_count = torch.cuda.device_count()
            self.devices = [f"cuda:{i}" for i in range(gpu_count)]
        else:
            self.devices = devices

        self.num_devices = len(self.devices)

        # Compute shares of channels per a GPU.
        if len(self.num_channels) == 1:
            # Allocate the same number of channels per every GPU.
            self.shares = [self.num_channels[0]] * self.num_devices
        elif len(self.num_channels) == self.num_devices:
            self.shares = self.num_channels
        else:
            # User input was contradicting.
            raise Exception(
                "There was a contradicting mismatch between "
                "num_channels, and devices."
            )

        # How many channels in total?
        self.total_num_channels = sum(self.shares)

        # We generate random bytes 4x4 = 16 per an array and hence,
        # internally only need to procure N // 4 length arrays.
        # Out of the 16, we generate discrete gaussian or uniform
        # samples 4 at a time.
        self.L = self.num_coefs // 4

        # We build binary search tree for discrete gaussian here.
        (
            self.btree,
            self.btree_ptr,
            self.btree_size,
            self.tree_depth,
        ) = build_CDT_binary_search_tree(security_bits=128, sigma=sigma)

        # Counter range at each GPU.
        # Note that the counters only include the non-repeating channels.
        # We can add later counters at the end that start from the
        # self.total_num_channels * self.L
        self.start_ind = [0] + [s * self.L for s in self.shares[:-1]]
        self.ind_increments = [s * self.L for s in self.shares]
        self.end_ind = [
            s + e for s, e in zip(self.start_ind, self.ind_increments)
        ]

        # Total increment to add to counters after each random bytes generation.
        self.inc = (
            self.total_num_channels + self.num_repeating_channels
        ) * self.L
        self.repeating_start = self.total_num_channels * self.L

        # expand 32-byte k.
        # This is 1634760805, 857760878, 2036477234, 1797285236.
        str2ord = lambda s: sum([2 ** (i * 8) * c for i, c in enumerate(s)])
        self.nothing_up_my_sleeve = []
        for device in self.devices:
            str_constant = torch.tensor(
                [
                    str2ord(b"expa"),
                    str2ord(b"nd 3"),
                    str2ord(b"2-by"),
                    str2ord(b"te k"),
                ],
                device=device,
                dtype=torch.int64,
            )
            self.nothing_up_my_sleeve.append(str_constant)

        # Prepare the state tensors.
        self.states = []
        for dev_id in range(self.num_devices):
            state_size = (
                (self.shares[dev_id] + self.num_repeating_channels) * self.L,
                16,
            )
            state = torch.zeros(
                state_size, dtype=torch.int64, device=self.devices[dev_id]
            )
            self.states.append(state)

        # Prepare a channeled views.
        self.channeled_states = [
            self.states[i].view(
                self.shares[i] + self.num_repeating_channels, self.L, -1
            )
            for i in range(self.num_devices)
        ]

        # The counter.
        self.counters = []
        repeating_counter = list(range(self.repeating_start, self.inc))
        for dev_id in range(self.num_devices):
            counter = (
                list(range(self.start_ind[dev_id], self.end_ind[dev_id]))
                + repeating_counter
            )

            counter_tensor = torch.tensor(
                counter, dtype=torch.int64, device=self.devices[dev_id]
            )
            self.counters.append(counter_tensor)

        self.refresh(seed, nonce)

    def refresh(self, seed=None, nonce=None):
        # Generate seed if necessary.
        self.key = self.generate_key(seed)

        # Generate nonce if necessary.
        self.nonce = self.generate_nonce(nonce)

        # Iterate over all devices.
        for dev_id in range(self.num_devices):
            self.initialize_states(dev_id, seed, nonce)

    def initialize_states(self, dev_id, seed=None, nonce=None):
        state = self.states[dev_id]
        state.zero_()

        # Set the counter.
        # It is hardly unlikely we will use CxL > 2**32.
        # Just fill in the 12th element
        # (The lower bytes of the counter).
        state[:, 12] = self.counters[dev_id][None, :]

        # Set the expand 32-bye k
        state[:, 0:4] = self.nothing_up_my_sleeve[dev_id][None, :]

        # Set the seed.
        state[:, 4:12] = self.key[dev_id][None, :]

        # Fill in nonce.
        state[:, 14:] = self.nonce[dev_id][None, :]

    def generate_initial_bytes(self, nbytes, part_bytes=4, seed=None):
        seeds = []
        if seed is None:
            n_keys = nbytes // part_bytes
            hex2int = lambda x, nbytes: int(binascii.hexlify(x), 16)
            seed0 = [
                hex2int(os.urandom(part_bytes), part_bytes)
                for _ in range(n_keys)
            ]
            for dev_id in range(self.num_devices):
                cuda_seed = torch.tensor(
                    seed0, dtype=torch.int64, device=self.devices[dev_id]
                )
                seeds.append(cuda_seed)
        else:
            seed0 = seed
            for dev_id in range(self.num_devices):
                cuda_seed = torch.tensor(
                    seed0, dtype=torch.int64, device=self.devices[dev_id]
                )
                seeds.append(cuda_seed)
        return seeds

    def generate_key(self, seed):
        # 256bits seed as a key.
        # We generate the same key seed for every GPU.
        # Randomity is produced by counters, not the key.
        return self.generate_initial_bytes(32, seed=None)

    def generate_nonce(self, seed):
        # nonce is 64bits.
        return self.generate_initial_bytes(8, seed=None)

    def randbytes(self, shares=None, repeats=0, reshape=False):
        # Generates (shares_i + repeats) X length random bytes.
        if shares is None:
            shares = self.shares

        # Set the target states.
        target_states = []
        for devi in range(self.num_devices):
            start_channel = self.shares[devi] - shares[devi]
            end_channel = self.shares[devi] + repeats
            device_states = self.channeled_states[devi][
                start_channel:end_channel, :, :
            ]
            target_states.append(device_states.view(-1, 16))

        # Derive random bytes.
        random_bytes = chacha20_cuda.chacha20(target_states, self.inc)

        # If not reshape, flatten.
        if reshape:
            random_bytes = [rb.view(-1, self.L, 16) for rb in random_bytes]

        return random_bytes

    def randint(self, amax=3, shift=0, repeats=0):
        # The default values are for generating the same uniform ternary
        # arrays in all GPUs.

        if not isinstance(amax, (list, tuple)):
            amax = [[amax] for share in self.shares]

        # Calculate shares.
        # If repeats are greater than 0, those channels are
        # subtracted from shares.
        shares = [len(am) - repeats for am in amax]

        # Convert the amax list to contiguous numpy array pointers.
        q_conti = [np.ascontiguousarray(q, dtype=np.uint64) for q in amax]
        q_ptr = [q.__array_interface__["data"][0] for q in q_conti]

        # Set the target states.
        target_states = []
        for devi in range(self.num_devices):
            start_channel = self.shares[devi] - shares[devi]
            end_channel = self.shares[devi] + repeats
            device_states = self.channeled_states[devi][
                start_channel:end_channel, :, :
            ]
            target_states.append(device_states)

        # Generate the randint.
        rand_int = randint_cuda.randint_fast(
            target_states, q_ptr, shift, self.inc
        )

        return rand_int

    def discrete_gaussian(self, non_repeats=0, repeats=1):
        if not isinstance(non_repeats, (list, tuple)):
            shares = [non_repeats] * self.num_devices
        else:
            shares = non_repeats

        # Set the target states.
        target_states = []
        for devi in range(self.num_devices):
            start_channel = self.shares[devi] - shares[devi]
            end_channel = self.shares[devi] + repeats
            device_states = self.channeled_states[devi][
                start_channel:end_channel, :, :
            ]
            target_states.append(device_states.view(-1, 16))

        # Generate the randint.
        rand_int = discrete_gaussian_cuda.discrete_gaussian_fast(
            target_states,
            self.btree_ptr,
            self.btree_size,
            self.tree_depth,
            self.inc,
        )
        # Reformat the rand_int.
        rand_int = [ri.view(-1, self.num_coefs) for ri in rand_int]

        return rand_int

    def randround(self, coef):
        """Randomly round coef. Coef must be a double tensor.
        coef must reside in the fist GPU in the GPUs list"""

        # The following slicing is OK, since we're using only the first
        # contiguous stream of states.
        # It will not make the target state strided.
        L = self.num_coefs // 16
        rand_bytes = chacha20_cuda.chacha20((self.states[0][:L],), self.inc)[
            0
        ].ravel()
        randround_cuda.randround([coef], [rand_bytes])
        return rand_bytes
