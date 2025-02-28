import datetime
import math
import pickle
from hashlib import sha256
from pathlib import Path
import warnings

from . import errors
import numpy as np
import torch
from loguru import logger
from liberate.utils.mvc import initonly, strictype
from liberate.fhe.context.ckks_context import CkksContext
from liberate.fhe.typing import *
from liberate.fhe.encdec import (
    decode as codec_decode,
    encode as codec_encode,
    rotate as codec_rotate,
    conjugate as codec_conjugate,
)
from liberate.ntt import NTTContext
from liberate.ntt import ntt_cuda
from liberate.csprng import Csprng


class CkksEngine:
    def __init__(
        self,
        devices: List[int] = None,
        verbose: bool = False,
        bias_guard: bool = True,
        norm: str = "forward",
        **ctx_params,
    ):
        self.bias_guard = bias_guard
        self.norm = norm
        self.ctx = CkksContext(**ctx_params)
        self.ntt = NTTContext(self.ctx, devices=devices, verbose=verbose)
        self.rng = Csprng(
            num_coefs=self.ntt.ckksCtx.N,
            num_channels=[len(di) for di in self.ntt.p.d],
            num_repeating_channels=max(self.ntt.num_special_primes, 2),
            devices=self.ntt.devices,
        )

        self.make_adjustments_and_corrections()
        self.mont_PR = self.make_mont_PR()
        self.create_ksk_rescales()
        self.alloc_parts()
        self.leveled_devices()
        self.rescale_scales = self.create_rescale_scales()

    @property
    def num_slots(self) -> int:
        return self.ctx.N // 2

    @property
    def num_levels(self) -> int:
        return self.ntt.num_levels - 1

    @property
    def int_scale(self) -> int:
        return 2**self.ctx.scale_bits

    @property
    def scale(self) -> float:
        return np.float64(2**self.ctx.scale_bits)

    @property
    def device0(self) -> int:
        return self.ntt.devices[0]

    @property
    def hash(self) -> str:
        qstr = ",".join([str(qi) for qi in self.ctx.q])
        hashstr = (self.ctx.generation_string + "_" + qstr).encode("utf-8")
        return sha256(bytes(hashstr)).hexdigest()

    # -------------------------------------------------------------------------------------------
    # Various pre-calculations.
    # -------------------------------------------------------------------------------------------
    @initonly
    def create_rescale_scales(self):
        rescale_scales = []
        for level in range(self.num_levels):
            rescale_scales.append([])

            for device_id in range(self.ntt.num_devices):
                dest_level = self.ntt.p.destination_arrays[level]

                if device_id < len(dest_level):
                    dest = dest_level[device_id]
                    rescaler_device_id = self.ntt.p.rescaler_loc[level]
                    m0 = self.ctx.q[level]

                    if rescaler_device_id == device_id:
                        m = [self.ctx.q[i] for i in dest[1:]]
                    else:
                        m = [self.ctx.q[i] for i in dest]

                    scales = [(pow(m0, -1, mi) * self.ctx.R) % mi for mi in m]

                    scales = torch.tensor(
                        scales,
                        dtype=self.ctx.torch_dtype,
                        device=self.ntt.devices[device_id],
                    )
                    rescale_scales[level].append(scales)

        return rescale_scales

    @initonly
    def leveled_devices(self):
        self.len_devices = []
        for level in range(self.num_levels):
            self.len_devices.append(
                len([[a] for a in self.ntt.p.p[level] if len(a) > 0])
            )

        self.neighbor_devices = []
        for level in range(self.num_levels):
            self.neighbor_devices.append([])
            len_devices_at = self.len_devices[level]
            available_devices_ids = range(len_devices_at)
            for src_device_id in available_devices_ids:
                neighbor_devices_at = [
                    device_id
                    for device_id in available_devices_ids
                    if device_id != src_device_id
                ]
                self.neighbor_devices[level].append(neighbor_devices_at)

    @initonly
    def alloc_parts(self):
        self.parts_alloc = []
        for level in range(self.num_levels):
            num_parts = [len(parts) for parts in self.ntt.p.p[level]]
            parts_alloc = [
                alloc[-num_parts[di] - 1 : -1]
                for di, alloc in enumerate(self.ntt.p.part_allocations)
            ]
            self.parts_alloc.append(parts_alloc)

        self.stor_ids = []
        for level in range(self.num_levels):
            self.stor_ids.append([])
            alloc = self.parts_alloc[level]
            min_id = min([min(a) for a in alloc if len(a) > 0])
            for device_id in range(self.ntt.num_devices):
                global_ids = self.parts_alloc[level][device_id]
                new_ids = [i - min_id for i in global_ids]
                self.stor_ids[level].append(new_ids)

    @initonly
    def create_ksk_rescales(self):
        # reserve the buffers.
        self.ksk_buffers = []
        for device_id in range(self.ntt.num_devices):
            self.ksk_buffers.append([])
            for part_id in range(len(self.ntt.p.p[0][device_id])):
                buffer = torch.empty(
                    [self.ntt.num_special_primes, self.ctx.N],
                    dtype=self.ctx.torch_dtype,
                ).pin_memory()
                self.ksk_buffers[device_id].append(buffer)

        # Create the buffers.
        R = self.ctx.R
        P = self.ctx.q[-self.ntt.num_special_primes :][::-1]
        m = self.ctx.q
        PiR = [
            [(pow(Pj, -1, mi) * R) % mi for mi in m[: -P_ind - 1]]
            for P_ind, Pj in enumerate(P)
        ]

        self.PiRs = []

        level = 0
        self.PiRs.append([])

        for P_ind in range(self.ntt.num_special_primes):
            self.PiRs[level].append([])

            for device_id in range(self.ntt.num_devices):
                dest = self.ntt.p.destination_arrays_with_special[level][
                    device_id
                ]
                PiRi = [PiR[P_ind][i] for i in dest[: -P_ind - 1]]
                PiRi = torch.tensor(
                    PiRi,
                    device=self.ntt.devices[device_id],
                    dtype=self.ctx.torch_dtype,
                )
                self.PiRs[level][P_ind].append(PiRi)

        for level in range(1, self.num_levels):
            self.PiRs.append([])

            for P_ind in range(self.ntt.num_special_primes):

                self.PiRs[level].append([])

                for device_id in range(self.ntt.num_devices):
                    start = self.ntt.starts[level][device_id]
                    PiRi = self.PiRs[0][P_ind][device_id][start:]

                    self.PiRs[level][P_ind].append(PiRi)

    @initonly
    def make_mont_PR(self):
        P = math.prod(self.ntt.ckksCtx.q[-self.ntt.num_special_primes :])
        R = self.ctx.R
        PR = P * R
        mont_PR = []
        for device_id in range(self.ntt.num_devices):
            dest = self.ntt.p.destination_arrays[0][device_id]
            m = [self.ctx.q[i] for i in dest]
            PRm = [PR % mi for mi in m]
            PRm = torch.tensor(
                PRm,
                device=self.ntt.devices[device_id],
                dtype=self.ctx.torch_dtype,
            )
            mont_PR.append(PRm)
        return mont_PR

    @initonly
    def make_adjustments_and_corrections(self):

        self.alpha = [
            (self.scale / np.float64(q)) ** 2
            for q in self.ctx.q[: self.ctx.num_scales]
        ]
        self.deviations = [1]
        for al in self.alpha:
            self.deviations.append(self.deviations[-1] ** 2 * al)

        self.final_q_ind = [
            da[0][0] for da in self.ntt.p.destination_arrays[:-1]
        ]
        self.final_q = [self.ctx.q[ind] for ind in self.final_q_ind]
        self.final_alpha = [(self.scale / np.float64(q)) for q in self.final_q]
        self.corrections = [
            1 / (d * fa) for d, fa in zip(self.deviations, self.final_alpha)
        ]

        self.base_prime = self.ctx.q[self.ntt.p.base_prime_idx]

        self.final_scalar = []
        for qi, q in zip(self.final_q_ind, self.final_q):
            scalar = (
                pow(q, -1, self.base_prime) * self.ctx.R
            ) % self.base_prime
            scalar = torch.tensor(
                [scalar],
                device=self.ntt.devices[0],
                dtype=self.ctx.torch_dtype,
            )
            self.final_scalar.append(scalar)

    # -------------------------------------------------------------------------------------------
    # Encode/Decode
    # -------------------------------------------------------------------------------------------

    def padding(self, m: Union[list, np.ndarray, torch.Tensor]):
        # todo how about length > num_slots
        if isinstance(m, torch.Tensor):
            assert (
                len(m.shape) == 1
            ), f"Input tensor should be 1D, but got {len(m.shape)}D."
        if isinstance(m, torch.Tensor):
            padding_result = torch.cat(
                (m, torch.zeros(self.num_slots - m.shape[0], device=m.device))
            )
        else:
            try:
                m_len = len(m)
                padding_result = np.pad(
                    m, (0, self.num_slots - m_len), constant_values=(0, 0)
                )
            except TypeError as e:
                m_len = len([m])
                padding_result = np.pad(
                    [m], (0, self.num_slots - m_len), constant_values=(0, 0)
                )

        return padding_result

    def encode(self, m, level: int = 0, padding=True) -> list[torch.Tensor]:
        """
        Encode a plain message m.
        Note that the encoded plain text is pre-permuted to yield cyclic rotation.
        """
        deviation = self.deviations[level]
        if padding:
            m = self.padding(m)
        encoded = [
            codec_encode(
                m,
                scale=self.scale,
                rng=self.rng,
                device=self.device0,
                deviation=deviation,
                norm=self.norm,
            )
        ]

        pt_buffer = self.ksk_buffers[0][0][0]
        pt_buffer.copy_(encoded[-1])
        for dev_id in range(1, self.ntt.num_devices):
            encoded.append(pt_buffer.cuda(self.ntt.devices[dev_id]))
        return encoded

    def decode(self, m, level=0, is_real: bool = False) -> list:
        """
        Base prime is located at -1 of the RNS channels in GPU0.
        Assuming this is an orginary RNS deinclude_special.
        """
        correction = self.corrections[level]
        decoded = codec_decode(
            m[0].squeeze(),
            scale=self.scale,
            correction=correction,
            norm=self.norm,
        )
        m = decoded[: self.ctx.N // 2].cpu().numpy()
        if is_real:
            m = m.real
        return m

    # -------------------------------------------------------------------------------------------
    # secret key/public key generation.
    # -------------------------------------------------------------------------------------------

    def create_secret_key(self, include_special: bool = True) -> SecretKey:
        uniform_ternary = self.rng.randint(amax=3, shift=-1, repeats=1)

        mult_type = -2 if include_special else -1
        unsigned_ternary = self.ntt.tile_unsigned(
            uniform_ternary, lvl=0, mult_type=mult_type
        )
        self.ntt.enter_ntt(unsigned_ternary, 0, mult_type)

        return SecretKey(
            data=unsigned_ternary,
            include_special=include_special,
            montgomery_state=True,
            ntt_state=True,
            level=0,
        )

    @strictype
    def create_public_key(
        self,
        sk: SecretKey,
        include_special: bool = False,
        a: List[torch.Tensor] = None,
    ) -> PublicKey:
        """
        Generates a public key against the secret key sk.
        pk = -a * sk + e = e - a * sk
        """
        # if sk.origin != origin_names["sk"]:
        #     raise errors.NotMatchType(origin=sk.origin, to=origin_names["sk"])
        if include_special and not sk.include_special:
            raise errors.SecretKeyNotIncludeSpecialPrime()

        # Set the mult_type
        mult_type = -2 if include_special else -1

        # Generate errors for the ordinary case.
        level = 0
        e = self.rng.discrete_gaussian(repeats=1)
        e = self.ntt.tile_unsigned(e, level, mult_type)

        self.ntt.enter_ntt(e, level, mult_type)
        repeats = self.ctx.num_special_primes if sk.include_special else 0

        # Applying mont_mult in the order of 'a', sk will
        if a is None:
            a = self.rng.randint(
                self.ntt.q_prepack[mult_type][level][0], repeats=repeats
            )

        sa = self.ntt.mont_mult(a, sk.data, 0, mult_type)
        pk0 = self.ntt.mont_sub(e, sa, 0, mult_type)

        return PublicKey(
            data=[pk0, a],
            include_special=include_special,
            ntt_state=True,
            montgomery_state=True,
            level=0,
        )

    # -------------------------------------------------------------------------------------------
    # Encrypt/Decrypt
    # -------------------------------------------------------------------------------------------

    @strictype
    def encrypt(
        self, pt: List[torch.Tensor], pk: PublicKey, level: int = 0
    ) -> Ciphertext:
        """
        We again, multiply pt by the scale.
        Since pt is already multiplied by the scale,
        the multiplied pt no longer can be stored
        in a single RNS channel.
        That means we are forced to do the multiplication
        in full RNS domain.
        Note that we allow encryption at
        levels other than 0, and that will take care of multiplying
        the deviation factors.
        """
        # if pk.origin != origin_names["pk"]:
        #     raise errors.NotMatchType(origin=pk.origin, to=origin_names["pk"])

        mult_type = -2 if pk.include_special else -1

        e0e1 = self.rng.discrete_gaussian(repeats=2)

        e0 = [e[0] for e in e0e1]
        e1 = [e[1] for e in e0e1]

        e0_tiled = self.ntt.tile_unsigned(e0, level, mult_type)
        e1_tiled = self.ntt.tile_unsigned(e1, level, mult_type)

        pt_tiled = self.ntt.tile_unsigned(pt, level, mult_type)
        self.ntt.mont_enter_scale(pt_tiled, level, mult_type)
        self.ntt.mont_redc(pt_tiled, level, mult_type)
        pte0 = self.ntt.mont_add(pt_tiled, e0_tiled, level, mult_type)

        start = self.ntt.starts[level]
        pk0 = [
            pk.data[0][di][start[di] :] for di in range(self.ntt.num_devices)
        ]
        pk1 = [
            pk.data[1][di][start[di] :] for di in range(self.ntt.num_devices)
        ]

        v = self.rng.randint(amax=2, shift=0, repeats=1)

        v = self.ntt.tile_unsigned(v, level, mult_type)
        self.ntt.enter_ntt(v, level, mult_type)

        vpk0 = self.ntt.mont_mult(v, pk0, level, mult_type)
        vpk1 = self.ntt.mont_mult(v, pk1, level, mult_type)

        self.ntt.intt_exit(vpk0, level, mult_type)
        self.ntt.intt_exit(vpk1, level, mult_type)

        ct0 = self.ntt.mont_add(vpk0, pte0, level, mult_type)
        ct1 = self.ntt.mont_add(vpk1, e1_tiled, level, mult_type)

        self.ntt.reduce_2q(ct0, level, mult_type)
        self.ntt.reduce_2q(ct1, level, mult_type)

        ct = Ciphertext(
            data=[ct0, ct1],
            include_special=mult_type == -2,
            ntt_state=False,
            montgomery_state=False,
            level=level,
        )

        return ct

    @strictype
    def decrypt_triplet(
        self, ct_mult: CiphertextTriplet, sk: SecretKey, final_round=True
    ) -> list[torch.Tensor]:

        if not ct_mult.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not ct_mult.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)
        if not sk.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not sk.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)

        level = ct_mult.level
        d0 = [ct_mult.data[0][0].clone()]
        d1 = [ct_mult.data[1][0]]
        d2 = [ct_mult.data[2][0]]

        self.ntt.intt_exit_reduce(d0, level)

        sk_data = [sk.data[0][self.ntt.starts[level][0] :]]

        d1_s = self.ntt.mont_mult(d1, sk_data, level)

        s2 = self.ntt.mont_mult(sk_data, sk_data, level)
        d2_s2 = self.ntt.mont_mult(d2, s2, level)

        self.ntt.intt_exit(d1_s, level)
        self.ntt.intt_exit(d2_s2, level)

        pt = self.ntt.mont_add(d0, d1_s, level)
        pt = self.ntt.mont_add(pt, d2_s2, level)
        self.ntt.reduce_2q(pt, level)

        base_at = (
            -self.ctx.num_special_primes - 1 if ct_mult.include_special else -1
        )

        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        final_scalar = self.final_scalar[level]
        scaled = self.ntt.mont_sub([base], [scaler], -1)
        self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
        self.ntt.reduce_2q(scaled, -1)
        self.ntt.make_signed(scaled, -1)

        # Round?
        if final_round:
            # The scaler and the base channels are guaranteed to be in the
            # device 0.
            rounding_prime = self.ntt.qlists[0][
                -self.ctx.num_special_primes - 2
            ]
            rounder = (scaler[0] > (rounding_prime // 2)) * 1
            scaled[0] += rounder

        return scaled

    @strictype
    def decrypt_double(
        self, ct: Ciphertext, sk: SecretKey, final_round=True
    ) -> list[torch.Tensor]:

        if ct.ntt_state:
            raise errors.NTTStateError(expected=False)
        if ct.montgomery_state:
            raise errors.MontgomeryStateError(expected=False)
        if not sk.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not sk.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)

        ct0 = ct.data[0][0]
        level = ct.level
        sk_data = sk.data[0][self.ntt.starts[level][0] :]
        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)
        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        pt = self.ntt.mont_add([ct0], sa, level)
        self.ntt.reduce_2q(pt, level)

        base_at = (
            -self.ctx.num_special_primes - 1 if ct.include_special else -1
        )

        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        final_scalar = self.final_scalar[level]
        scaled = self.ntt.mont_sub([base], [scaler], -1)
        self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
        self.ntt.reduce_2q(scaled, -1)
        self.ntt.make_signed(scaled, -1)

        # Round?
        if final_round:
            # The scaler and the base channels are guaranteed to be in the
            # device 0.
            rounding_prime = self.ntt.qlists[0][
                -self.ctx.num_special_primes - 2
            ]
            rounder = (scaler[0] > (rounding_prime // 2)) * 1
            scaled[0] += rounder

        return scaled

    # @restrict_type
    def decrypt(
        self,
        ct: Union[Ciphertext, CiphertextTriplet],
        sk: SecretKey,
        final_round=True,
    ) -> list[torch.Tensor]:
        """
        Decrypt the cipher text ct using the secret key sk.
        Note that the final rescaling must precede the actual decryption process.
        """

        # if sk.origin != origin_names["sk"]:
        #     raise errors.NotMatchType(origin=sk.origin, to=origin_names["sk"])

        if isinstance(ct, CiphertextTriplet):
            pt = self.decrypt_triplet(
                ct_mult=ct, sk=sk, final_round=final_round
            )
        else:
            # raise errors.NotMatchType(origin=ct.origin, to=f"{origin_names['ct']} or {origin_names['ctt']}")
            # try decode as ct
            if not isinstance(ct, Ciphertext):
                logger.warning(
                    f"ct type is mismatched: excepted {Ciphertext} or {CiphertextTriplet}, got {type(ct)}, will try to decode as ct anyway."
                )
            pt = self.decrypt_double(ct=ct, sk=sk, final_round=final_round)

        return pt

    # -------------------------------------------------------------------------------------------
    # Key switching.
    # -------------------------------------------------------------------------------------------

    @strictype
    def create_key_switching_key(
        self, sk_from: SecretKey, sk_to: SecretKey, a=None
    ) -> KeySwitchKey:
        """
        Creates a key to switch the key for sk_src to sk_dst.
        """

        if not sk_from.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not sk_from.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)
        if not sk_to.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not sk_to.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)

        level = 0

        stops = self.ntt.stops[-1]
        Psk_src = [
            sk_from.data[di][: stops[di]].clone()
            for di in range(self.ntt.num_devices)
        ]

        self.ntt.mont_enter_scalar(Psk_src, self.mont_PR, level)

        ksk = [[] for _ in range(self.ntt.p.num_partitions + 1)]

        for device_id in range(self.ntt.num_devices):
            for part_id, part in enumerate(self.ntt.p.p[level][device_id]):
                global_part_id = self.ntt.p.part_allocations[device_id][
                    part_id
                ]

                crs = a[global_part_id] if a else None
                pk = self.create_public_key(sk_to, include_special=True, a=crs)

                key = tuple(part)
                astart = part[0]
                astop = part[-1] + 1
                shard = Psk_src[device_id][astart:astop]
                pk_data = pk.data[0][device_id][astart:astop]

                _2q = self.ntt.parts_pack[device_id][key]["_2q"]
                update_part = ntt_cuda.mont_add([pk_data], [shard], _2q)[0]
                pk_data.copy_(update_part, non_blocking=True)

                pk.description = f"key switch key part index {global_part_id}"  # todo allow this and rename to description

                ksk[global_part_id] = pk

        return KeySwitchKey(
            data=ksk,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            level=level,
        )

    def pre_extend(self, a, device_id, level, part_id, exit_ntt=False):
        # param_parts contain only the ordinary parts.
        # Hence, loop around it.
        # text_parts contain special primes.
        text_part = self.ntt.p.parts[level][device_id][part_id]
        param_part = self.ntt.p.p[level][device_id][part_id]

        # Carve out the partition.
        alpha = len(text_part)
        a_part = a[device_id][text_part[0] : text_part[-1] + 1]

        # Release ntt.
        if exit_ntt:
            self.ntt.intt_exit_reduce([a_part], level, device_id, part_id)

        # Prepare a state.
        # Initially, it is x[0] % m[i].
        # However, m[i] is a monotonically increasing
        # sequence, i.e., repeating the row would suffice
        # to do the job.

        # 2023-10-16, Juwhan Kim, In fact, m[i] is NOT monotonically increasing.

        state = a_part[0].repeat(alpha, 1)

        key = tuple(param_part)
        for i in range(alpha - 1):
            mont_pack = self.ntt.parts_pack[device_id][param_part[i + 1],][
                "mont_pack"
            ]
            _2q = self.ntt.parts_pack[device_id][param_part[i + 1],]["_2q"]
            Y_scalar = self.ntt.parts_pack[device_id][key]["Y_scalar"][i][None]

            Y = (a_part[i + 1] - state[i + 1])[None, :]

            # mont_enter will take care of signedness.
            # ntt_cuda.make_unsigned([Y], _2q)
            ntt_cuda.mont_enter([Y], [Y_scalar], *mont_pack)
            # ntt_cuda.reduce_2q([Y], _2q)

            state[i + 1] = Y

            if i + 2 < alpha:
                state_key = tuple(param_part[i + 2 :])
                state_mont_pack = self.ntt.parts_pack[device_id][state_key][
                    "mont_pack"
                ]
                state_2q = self.ntt.parts_pack[device_id][state_key]["_2q"]
                L_scalar = self.ntt.parts_pack[device_id][key]["L_scalar"][i]
                new_state_len = alpha - (i + 2)
                new_state = Y.repeat(new_state_len, 1)
                ntt_cuda.mont_enter([new_state], [L_scalar], *state_mont_pack)
                state[i + 2 :] += new_state

        # Returned state is in plain integer format.
        return state

    def extend(self, state, device_id, level, part_id, target_device_id=None):
        # Note that device_id, level, and part_id is from
        # where the state has been originally calculated at.
        # The state can reside in a different GPU than
        # the original one.

        if target_device_id is None:
            target_device_id = device_id

        rns_len = len(
            self.ntt.p.destination_arrays_with_special[level][target_device_id]
        )
        alpha = len(state)

        # Initialize the output
        extended = state[0].repeat(rns_len, 1)
        self.ntt.mont_enter([extended], level, target_device_id, -2)

        # Generate the search key to find the L_enter.
        part = self.ntt.p.p[level][device_id][part_id]
        key = tuple(part)

        # Extract the L_enter in the target device.
        L_enter = self.ntt.parts_pack[device_id][key]["L_enter"][
            target_device_id
        ]

        # L_enter covers the whole rns range.
        # Start from the leveled start.
        start = self.ntt.starts[level][target_device_id]

        # Loop to generate.
        for i in range(alpha - 1):
            Y = state[i + 1].repeat(rns_len, 1)

            self.ntt.mont_enter_scalar(
                [Y], [L_enter[i][start:]], level, target_device_id, -2
            )
            extended = self.ntt.mont_add(
                [extended], [Y], level, target_device_id, -2
            )[0]

        # Returned extended is in the Montgomery format.
        return extended

    def create_switcher(
        self, a: List[torch.Tensor], ksk: KeySwitchKey, level, exit_ntt=False
    ) -> tuple:
        # ksk parts allocation.
        ksk_alloc = self.parts_alloc[level]

        # Device lens and neighbor devices.
        len_devices = self.len_devices[level]
        neighbor_devices = self.neighbor_devices[level]

        # Iterate over source device ids, and then part ids.
        num_parts = sum([len(alloc) for alloc in ksk_alloc])
        part_results = [
            [
                [[] for _ in range(len_devices)],
                [[] for _ in range(len_devices)],
            ]
            for _ in range(num_parts)
        ]

        # 1. Generate states.
        states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for part_id in range(len(self.ntt.p.p[level][src_device_id])):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                state = self.pre_extend(
                    a, src_device_id, level, part_id, exit_ntt
                )
                states[storage_id] = state

        # 2. Copy to CPU. # todo if only one device, maybe can skip copying to CPU.
        CPU_states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for part_id, part in enumerate(self.ntt.p.p[level][src_device_id]):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                alpha = len(part)
                CPU_state = self.ksk_buffers[src_device_id][part_id][:alpha]
                CPU_state.copy_(states[storage_id], non_blocking=True)
                CPU_states[storage_id] = CPU_state

        # 3. Continue on with the follow ups on source devices.
        for src_device_id in range(len_devices):
            for part_id in range(len(self.ntt.p.p[level][src_device_id])):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                state = states[storage_id]
                d0, d1 = self.switcher_later_part(
                    state, ksk, src_device_id, src_device_id, level, part_id
                )

                part_results[storage_id][0][src_device_id] = d0
                part_results[storage_id][1][src_device_id] = d1

        # 4. Copy onto neighbor GPUs the states.
        CUDA_states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for j, dst_device_id in enumerate(neighbor_devices[src_device_id]):
                for part_id, part in enumerate(
                    self.ntt.p.p[level][src_device_id]
                ):
                    storage_id = self.stor_ids[level][src_device_id][part_id]
                    CPU_state = CPU_states[storage_id]
                    CUDA_states[storage_id] = CPU_state.cuda(
                        self.ntt.devices[dst_device_id], non_blocking=True
                    )

        # 5. Synchronize.
        # torch.cuda.synchronize()

        # 6. Do follow ups on neighbors.
        for src_device_id in range(len_devices):
            for j, dst_device_id in enumerate(neighbor_devices[src_device_id]):
                for part_id, part in enumerate(
                    self.ntt.p.p[level][src_device_id]
                ):
                    storage_id = self.stor_ids[level][src_device_id][part_id]
                    CUDA_state = CUDA_states[storage_id]
                    d0, d1 = self.switcher_later_part(
                        CUDA_state,
                        ksk,
                        src_device_id,
                        dst_device_id,
                        level,
                        part_id,
                    )
                    part_results[storage_id][0][dst_device_id] = d0
                    part_results[storage_id][1][dst_device_id] = d1

        # 7. Sum up.
        summed0 = part_results[0][0]
        summed1 = part_results[0][1]

        for i in range(1, len(part_results)):
            summed0 = self.ntt.mont_add(summed0, part_results[i][0], level, -2)
            summed1 = self.ntt.mont_add(summed1, part_results[i][1], level, -2)

        # Rename summed's.
        d0 = summed0
        d1 = summed1

        # intt to prepare for division by P.
        self.ntt.intt_exit_reduce(d0, level, -2)
        self.ntt.intt_exit_reduce(d1, level, -2)

        # 6. Divide by P.
        # This is actually done in successive order.
        # Rescale from the most outer prime channel.
        # Start from the special len and drop channels one by one.

        # Pre-montgomery enter the ordinary part.
        # Note that special prime channels remain intact.
        c0 = [d[: -self.ntt.num_special_primes] for d in d0]
        c1 = [d[: -self.ntt.num_special_primes] for d in d1]

        self.ntt.mont_enter(c0, level, -1)
        self.ntt.mont_enter(c1, level, -1)

        current_len = [
            len(d) for d in self.ntt.p.destination_arrays_with_special[level]
        ]

        for P_ind in range(self.ntt.num_special_primes):
            PiRi = self.PiRs[level][P_ind]

            # Tile.
            P0 = [
                d[-1 - P_ind].repeat(current_len[di], 1)
                for di, d in enumerate(d0)
            ]
            P1 = [
                d[-1 - P_ind].repeat(current_len[di], 1)
                for di, d in enumerate(d1)
            ]

            # mont enter only the ordinary part.
            Q0 = [d[: -self.ntt.num_special_primes] for d in P0]
            Q1 = [d[: -self.ntt.num_special_primes] for d in P1]

            self.ntt.mont_enter(Q0, level, -1)
            self.ntt.mont_enter(Q1, level, -1)

            # subtract P0 and P1.
            # Note that by the consequence of the above mont_enter
            # ordinary parts will be in montgomery form,
            # while the special part remains plain.
            d0 = self.ntt.mont_sub(d0, P0, level, -2)
            d1 = self.ntt.mont_sub(d1, P1, level, -2)

            self.ntt.mont_enter_scalar(d0, PiRi, level, -2)
            self.ntt.mont_enter_scalar(d1, PiRi, level, -2)

        # Carve out again, since d0 and d1 are fresh new.
        c0 = [d[: -self.ntt.num_special_primes] for d in d0]
        c1 = [d[: -self.ntt.num_special_primes] for d in d1]

        # Exit the montgomery.
        self.ntt.mont_redc(c0, level, -1)
        self.ntt.mont_redc(c1, level, -1)

        self.ntt.reduce_2q(c0, level, -1)
        self.ntt.reduce_2q(c1, level, -1)

        # 7. Return
        return c0, c1

    def switcher_later_part(
        self,
        state,
        ksk: KeySwitchKey,
        src_device_id,
        dst_device_id,
        level,
        part_id,
    ):

        # Extend basis.
        extended = self.extend(
            state, src_device_id, level, part_id, dst_device_id
        )

        # ntt extended to prepare polynomial multiplication.
        # extended is in the Montgomery format already.
        self.ntt.ntt([extended], level, dst_device_id, -2)

        # Extract the ksk.
        ksk_loc = self.parts_alloc[level][src_device_id][part_id]
        ksk_part_data = ksk.data[ksk_loc].data

        start = self.ntt.starts[level][dst_device_id]
        ksk0_data = ksk_part_data[0][dst_device_id][start:]
        ksk1_data = ksk_part_data[1][dst_device_id][start:]

        # Multiply.
        d0 = self.ntt.mont_mult(
            [extended], [ksk0_data], level, dst_device_id, -2
        )
        d1 = self.ntt.mont_mult(
            [extended], [ksk1_data], level, dst_device_id, -2
        )

        # When returning, un-list the results by taking the 0th element.
        return d0[0], d1[0]

    @strictype
    def switch_key(self, ct: Ciphertext, ksk: KeySwitchKey) -> Ciphertext:
        include_special = ct.include_special
        ntt_state = ct.ntt_state
        montgomery_state = ct.montgomery_state
        # if ct.origin != origin_names["ct"]:
        #     raise errors.NotMatchType(origin=ct.origin, to=origin_names["ct"])

        level = ct.level
        a = ct.data[1]
        d0, d1 = self.create_switcher(a, ksk, level, exit_ntt=ct.ntt_state)

        new_ct0 = self.ntt.mont_add(ct.data[0], d0, level, -1)
        self.ntt.reduce_2q(new_ct0, level, -1)

        return Ciphertext(
            data=[new_ct0, d1],
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            level=level,
        )

    # -------------------------------------------------------------------------------------------
    # Multiplication.
    # -------------------------------------------------------------------------------------------

    @strictype
    def rescale(self, ct: Ciphertext, exact_rounding=True) -> Ciphertext:

        level = ct.level
        next_level = level + 1

        if next_level >= self.num_levels:
            raise errors.MaximumLevelError(
                level=ct.level, level_max=self.num_levels
            )

        rescaler_device_id = self.ntt.p.rescaler_loc[level]
        neighbor_devices_before = self.neighbor_devices[level]
        neighbor_devices_after = self.neighbor_devices[next_level]
        len_devices_after = len(neighbor_devices_after)
        len_devices_before = len(neighbor_devices_before)

        rescaling_scales = self.rescale_scales[level]
        data0 = [[] for _ in range(len_devices_after)]
        data1 = [[] for _ in range(len_devices_after)]

        rescaler0 = [[] for _ in range(len_devices_before)]
        rescaler1 = [[] for _ in range(len_devices_before)]

        rescaler0_at = ct.data[0][rescaler_device_id][0]
        rescaler0[rescaler_device_id] = rescaler0_at

        rescaler1_at = ct.data[1][rescaler_device_id][0]
        rescaler1[rescaler_device_id] = rescaler1_at

        if rescaler_device_id < len_devices_after:
            data0[rescaler_device_id] = ct.data[0][rescaler_device_id][1:]
            data1[rescaler_device_id] = ct.data[1][rescaler_device_id][1:]

        CPU_rescaler0 = self.ksk_buffers[0][0][0]
        CPU_rescaler1 = self.ksk_buffers[0][1][0]

        CPU_rescaler0.copy_(rescaler0_at, non_blocking=True)
        CPU_rescaler1.copy_(rescaler1_at, non_blocking=True)

        for device_id in neighbor_devices_before[rescaler_device_id]:
            device = self.ntt.devices[device_id]
            CUDA_rescaler0 = CPU_rescaler0.cuda(device)
            CUDA_rescaler1 = CPU_rescaler1.cuda(device)

            rescaler0[device_id] = CUDA_rescaler0
            rescaler1[device_id] = CUDA_rescaler1

            if device_id < len_devices_after:
                data0[device_id] = ct.data[0][device_id]
                data1[device_id] = ct.data[1][device_id]

        if exact_rounding:
            rescale_channel_prime_id = self.ntt.p.destination_arrays[level][
                rescaler_device_id
            ][0]

            round_at = self.ctx.q[rescale_channel_prime_id] // 2

            rounder0 = [[] for _ in range(len_devices_before)]
            rounder1 = [[] for _ in range(len_devices_before)]

            for device_id in range(len_devices_after):
                rounder0[device_id] = torch.where(
                    rescaler0[device_id] > round_at, 1, 0
                )
                rounder1[device_id] = torch.where(
                    rescaler1[device_id] > round_at, 1, 0
                )

        data0 = [(d - s) for d, s in zip(data0, rescaler0)]
        data1 = [(d - s) for d, s in zip(data1, rescaler1)]

        self.ntt.mont_enter_scalar(
            data0, self.rescale_scales[level], next_level
        )

        self.ntt.mont_enter_scalar(
            data1, self.rescale_scales[level], next_level
        )

        if exact_rounding:
            data0 = [(d + r) for d, r in zip(data0, rounder0)]
            data1 = [(d + r) for d, r in zip(data1, rounder1)]

        self.ntt.reduce_2q(data0, next_level)
        self.ntt.reduce_2q(data1, next_level)

        return Ciphertext(
            data=[data0, data1],
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            level=next_level,
        )

    @strictype
    def create_evk(self, sk: SecretKey) -> EvaluationKey:

        sk2_data = self.ntt.mont_mult(sk.data, sk.data, 0, -2)
        sk2 = EvaluationKey(
            data=sk2_data,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            level=sk.level,
        )
        return EvaluationKey.wrap(self.create_key_switching_key(sk2, sk))

    @strictype
    def cc_mult(
        self,
        a: Ciphertext,
        b: Ciphertext,
        evk: EvaluationKey,
        pre_rescale=True,
        post_relin=True,
    ) -> Union[Ciphertext, CiphertextTriplet]:

        if pre_rescale:
            x = self.rescale(a)
            y = self.rescale(b)
        else:
            x, y = a, b

        level = x.level

        # Multiply.
        x0 = x.data[0]
        x1 = x.data[1]

        y0 = y.data[0]
        y1 = y.data[1]

        self.ntt.enter_ntt(x0, level)
        self.ntt.enter_ntt(x1, level)
        self.ntt.enter_ntt(y0, level)
        self.ntt.enter_ntt(y1, level)

        d0 = self.ntt.mont_mult(x0, y0, level)

        x0y1 = self.ntt.mont_mult(x0, y1, level)
        x1y0 = self.ntt.mont_mult(x1, y0, level)
        d1 = self.ntt.mont_add(x0y1, x1y0, level)

        d2 = self.ntt.mont_mult(x1, y1, level)

        ct_mult = CiphertextTriplet(
            data=[d0, d1, d2],
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            level=level,
        )
        if post_relin:
            ct_mult = self.relinearize(ct_triplet=ct_mult, evk=evk)

        return ct_mult

    @strictype
    def relinearize(
        self, ct_triplet: CiphertextTriplet, evk: EvaluationKey
    ) -> Ciphertext:

        if not ct_triplet.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not ct_triplet.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)

        d0, d1, d2 = ct_triplet.data
        level = ct_triplet.level

        # intt.
        self.ntt.intt_exit_reduce(d0, level)
        self.ntt.intt_exit_reduce(d1, level)
        self.ntt.intt_exit_reduce(d2, level)

        # Key switch the x1y1.
        d2_0, d2_1 = self.create_switcher(d2, evk, level)

        # Add the switcher to d0, d1.
        d0 = [p + q for p, q in zip(d0, d2_0)]
        d1 = [p + q for p, q in zip(d1, d2_1)]

        # Final reduction.
        self.ntt.reduce_2q(d0, level)
        self.ntt.reduce_2q(d1, level)

        # Compose and return.
        return Ciphertext(
            data=[d0, d1],
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            level=level,
        )

    # -------------------------------------------------------------------------------------------
    # Rotation.
    # -------------------------------------------------------------------------------------------

    @strictype
    def create_rotation_key(
        self, sk: SecretKey, delta: int, a: List[torch.Tensor] = None
    ) -> RotationKey:

        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data)
        sk_new_data = [codec_rotate(s, delta) for s in sk_new_data]
        self.ntt.ntt(sk_new_data)
        sk_rotated = SecretKey(
            data=sk_new_data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            level=0,
        )

        rotk = RotationKey.wrap(
            self.create_key_switching_key(sk_rotated, sk, a=a)
        )

        rotk.delta = delta
        return rotk

    @strictype
    def rotate_single(self, ct: Ciphertext, rotk: RotationKey) -> Ciphertext:

        level = ct.level
        include_special = ct.include_special
        ntt_state = ct.ntt_state
        montgomery_state = ct.montgomery_state
        # origin = rotk.origin
        # delta = int(origin.split(":")[-1])

        rotated_ct_data = [
            [codec_rotate(d, rotk.delta) for d in ct_data]
            for ct_data in ct.data
        ]

        # Rotated ct may contain negative numbers.
        mult_type = -2 if include_special else -1
        for ct_data in rotated_ct_data:
            self.ntt.make_unsigned(ct_data, level, mult_type)
            self.ntt.reduce_2q(ct_data, level, mult_type)

        rotated_ct_rotated_sk = Ciphertext(
            data=rotated_ct_data,
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            level=level,
        )

        rotated_ct = self.switch_key(rotated_ct_rotated_sk, rotk)
        return rotated_ct

    @strictype
    def create_galois_key(self, sk: SecretKey) -> GaloisKey:
        galois_deltas = [2**i for i in range(self.ctx.logN - 1)]
        galois_key_parts = [
            self.create_rotation_key(sk, delta) for delta in galois_deltas
        ]

        galois_key = GaloisKey(
            data=galois_key_parts,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            level=0,
        )
        return galois_key

    @strictype
    def rotate_galois(
        self, ct: Ciphertext, gk: GaloisKey, delta: int, return_circuit=False
    ) -> Ciphertext:

        current_delta = delta % (self.ctx.N // 2)
        galois_circuit = []
        galois_deltas = [2**i for i in range(self.ctx.logN - 1)]
        while current_delta:
            galois_ind = int(math.log2(current_delta))
            galois_delta = galois_deltas[galois_ind]
            galois_circuit.append(galois_ind)
            current_delta -= galois_delta

        if len(galois_circuit) > 0:
            rotated_ct = self.rotate_single(ct, gk.data[galois_circuit[0]])

            for delta_ind in galois_circuit[1:]:
                rotated_ct = self.rotate_single(rotated_ct, gk.data[delta_ind])
        elif len(galois_circuit) == 0:
            rotated_ct = ct
        else:
            pass

        if return_circuit:
            return rotated_ct, galois_circuit
        else:
            return rotated_ct

    # -------------------------------------------------------------------------------------------
    # Add/sub.
    # -------------------------------------------------------------------------------------------
    @strictype
    def cc_add_double(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:

        if a.ntt_state:
            raise errors.NTTStateError(expected=False)
        if a.montgomery_state:
            raise errors.MontgomeryStateError(expected=False)
        if b.ntt_state:
            raise errors.NTTStateError(expected=False)
        if b.montgomery_state:
            raise errors.MontgomeryStateError(expected=False)

        level = a.level
        data = []
        c0 = self.ntt.mont_add(a.data[0], b.data[0], level)
        c1 = self.ntt.mont_add(a.data[1], b.data[1], level)
        self.ntt.reduce_2q(c0, level)
        self.ntt.reduce_2q(c1, level)
        data.extend([c0, c1])

        return Ciphertext(
            data=data,
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            level=level,
        )

    @strictype
    def cc_add_triplet(
        self, a: CiphertextTriplet, b: CiphertextTriplet
    ) -> CiphertextTriplet:

        if not a.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not a.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)
        if not b.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not b.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)

        level = a.level
        data = []
        c0 = self.ntt.mont_add(a.data[0], b.data[0], level)
        c1 = self.ntt.mont_add(a.data[1], b.data[1], level)
        self.ntt.reduce_2q(c0, level)
        self.ntt.reduce_2q(c1, level)
        data.extend([c0, c1])
        c2 = self.ntt.mont_add(a.data[2], b.data[2], level)
        self.ntt.reduce_2q(c2, level)
        data.append(c2)

        return CiphertextTriplet(
            data=data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            level=level,
        )

    @strictype
    def cc_add(
        self,
        a: Union[Ciphertext, CiphertextTriplet],
        b: Union[Ciphertext, CiphertextTriplet],
    ) -> Union[Ciphertext, CiphertextTriplet]:

        # if a.origin == origin_names["ct"] and b.origin == origin_names["ct"]:
        if isinstance(a, Ciphertext) and isinstance(b, Ciphertext):
            result = self.cc_add_double(a, b)
        # elif (
        #     a.origin == origin_names["ctt"] and b.origin == origin_names["ctt"]
        # ):
        elif isinstance(a, CiphertextTriplet) and isinstance(
            b, CiphertextTriplet
        ):
            result = self.cc_add_triplet(a, b)
        else:
            raise errors.DifferentTypeError(a=type(a), b=type(b))

        return result

    @strictype
    def cc_sub_double(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:

        if a.ntt_state:
            raise errors.NTTStateError(expected=False)
        if a.montgomery_state:
            raise errors.MontgomeryStateError(expected=False)
        if b.ntt_state:
            raise errors.NTTStateError(expected=False)
        if b.montgomery_state:
            raise errors.MontgomeryStateError(expected=False)

        level = a.level
        data = []

        c0 = self.ntt.mont_sub(a.data[0], b.data[0], level)
        c1 = self.ntt.mont_sub(a.data[1], b.data[1], level)
        self.ntt.reduce_2q(c0, level)
        self.ntt.reduce_2q(c1, level)
        data.extend([c0, c1])

        return Ciphertext(
            data=data,
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            level=level,
        )

    @strictype
    def cc_sub_triplet(
        self, a: CiphertextTriplet, b: CiphertextTriplet
    ) -> CiphertextTriplet:

        if not a.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not a.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)
        if not b.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not b.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)

        level = a.level
        data = []
        c0 = self.ntt.mont_sub(a.data[0], b.data[0], level)
        c1 = self.ntt.mont_sub(a.data[1], b.data[1], level)
        c2 = self.ntt.mont_sub(a.data[2], b.data[2], level)
        self.ntt.reduce_2q(c0, level)
        self.ntt.reduce_2q(c1, level)
        self.ntt.reduce_2q(c2, level)
        data.extend([c0, c1, c2])

        return CiphertextTriplet(
            data=data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            level=level,
        )

    @strictype
    def cc_sub(
        self,
        a: Union[Ciphertext, CiphertextTriplet],
        b: Union[Ciphertext, CiphertextTriplet],
    ) -> Union[Ciphertext, CiphertextTriplet]:

        if isinstance(a, Ciphertext) and isinstance(b, Ciphertext):
            ct_sub = self.cc_sub_double(a, b)
        elif isinstance(a, CiphertextTriplet) and isinstance(
            b, CiphertextTriplet
        ):
            ct_sub = self.cc_sub_triplet(a, b)
        else:
            raise errors.DifferentTypeError(a=type(a), b=type(b))
        return ct_sub

    # -------------------------------------------------------------------------------------------
    # Level up.
    # -------------------------------------------------------------------------------------------
    @strictype
    def level_up(self, ct: Ciphertext, dst_level: int) -> Ciphertext:

        current_level = ct.level

        new_ct = self.rescale(ct)

        src_level = current_level + 1

        dst_len_devices = len(self.ntt.p.destination_arrays[dst_level])

        diff_deviation = self.deviations[dst_level] / np.sqrt(
            self.deviations[src_level]
        )

        deviated_delta = round(self.scale * diff_deviation)

        if dst_level - src_level > 0:
            src_rns_lens = [
                len(d) for d in self.ntt.p.destination_arrays[src_level]
            ]
            dst_rns_lens = [
                len(d) for d in self.ntt.p.destination_arrays[dst_level]
            ]

            diff_rns_lens = [y - x for x, y in zip(dst_rns_lens, src_rns_lens)]

            new_ct_data0 = []
            new_ct_data1 = []

            for device_id in range(dst_len_devices):
                new_ct_data0.append(
                    new_ct.data[0][device_id][diff_rns_lens[device_id] :]
                )
                new_ct_data1.append(
                    new_ct.data[1][device_id][diff_rns_lens[device_id] :]
                )
        else:
            new_ct_data0, new_ct_data1 = new_ct.data

        multipliers = []
        for device_id in range(dst_len_devices):
            dest = self.ntt.p.destination_arrays[dst_level][device_id]
            q = [self.ctx.q[i] for i in dest]

            multiplier = [(deviated_delta * self.ctx.R) % qi for qi in q]
            multiplier = torch.tensor(
                multiplier,
                dtype=self.ctx.torch_dtype,
                device=self.ntt.devices[device_id],
            )
            multipliers.append(multiplier)

        self.ntt.mont_enter_scalar(new_ct_data0, multipliers, dst_level)
        self.ntt.mont_enter_scalar(new_ct_data1, multipliers, dst_level)

        self.ntt.reduce_2q(new_ct_data0, dst_level)
        self.ntt.reduce_2q(new_ct_data1, dst_level)

        new_ct = Ciphertext(
            data=[new_ct_data0, new_ct_data1],
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            level=dst_level,
        )

        return new_ct

    # -------------------------------------------------------------------------------------------
    # Fused enc/dec.
    # -------------------------------------------------------------------------------------------

    @strictype
    def encodecrypt(
        self, m, pk: PublicKey, level: int = 0, padding=True
    ) -> Ciphertext:

        if padding:
            m = self.padding(m=m)

        deviation = self.deviations[level]
        pt = codec_encode(
            m,
            scale=self.scale,
            device=self.device0,
            norm=self.norm,
            deviation=deviation,
            rng=self.rng,
            return_without_scaling=self.bias_guard,
        )

        if self.bias_guard:
            dc_integral = pt[0].item() // 1
            pt[0] -= dc_integral

            dc_scale = int(dc_integral) * int(self.scale)
            dc_rns = []
            for device_id, dest in enumerate(
                self.ntt.p.destination_arrays[level]
            ):
                dci = [dc_scale % self.ctx.q[i] for i in dest]
                dci = torch.tensor(
                    dci,
                    dtype=self.ctx.torch_dtype,
                    device=self.ntt.devices[device_id],
                )
                dc_rns.append(dci)

            pt *= np.float64(self.scale)
            pt = self.rng.randround(pt)

        encoded = [pt]

        pt_buffer = self.ksk_buffers[0][0][0]
        pt_buffer.copy_(encoded[-1])
        for dev_id in range(1, self.ntt.num_devices):
            encoded.append(pt_buffer.cuda(self.ntt.devices[dev_id]))

        mult_type = -2 if pk.include_special else -1

        e0e1 = self.rng.discrete_gaussian(repeats=2)

        e0 = [e[0] for e in e0e1]
        e1 = [e[1] for e in e0e1]

        e0_tiled = self.ntt.tile_unsigned(e0, level, mult_type)
        e1_tiled = self.ntt.tile_unsigned(e1, level, mult_type)

        pt_tiled = self.ntt.tile_unsigned(encoded, level, mult_type)

        if self.bias_guard:
            for device_id, pti in enumerate(pt_tiled):
                pti[:, 0] += dc_rns[device_id]

        self.ntt.mont_enter_scale(pt_tiled, level, mult_type)
        self.ntt.mont_redc(pt_tiled, level, mult_type)
        pte0 = self.ntt.mont_add(pt_tiled, e0_tiled, level, mult_type)

        start = self.ntt.starts[level]
        pk0 = [
            pk.data[0][di][start[di] :] for di in range(self.ntt.num_devices)
        ]
        pk1 = [
            pk.data[1][di][start[di] :] for di in range(self.ntt.num_devices)
        ]

        v = self.rng.randint(amax=2, shift=0, repeats=1)

        v = self.ntt.tile_unsigned(v, level, mult_type)
        self.ntt.enter_ntt(v, level, mult_type)

        vpk0 = self.ntt.mont_mult(v, pk0, level, mult_type)
        vpk1 = self.ntt.mont_mult(v, pk1, level, mult_type)

        self.ntt.intt_exit(vpk0, level, mult_type)
        self.ntt.intt_exit(vpk1, level, mult_type)

        ct0 = self.ntt.mont_add(vpk0, pte0, level, mult_type)
        ct1 = self.ntt.mont_add(vpk1, e1_tiled, level, mult_type)

        self.ntt.reduce_2q(ct0, level, mult_type)
        self.ntt.reduce_2q(ct1, level, mult_type)

        ct = Ciphertext(
            data=[ct0, ct1],
            include_special=mult_type == -2,
            ntt_state=False,
            montgomery_state=False,
            level=level,
        )

        return ct

    def decryptcode(
        self,
        ct: Union[Ciphertext, CiphertextTriplet],
        sk: SecretKey,
        is_real=False,
        final_round=True,
    ):  # todo keep on GPU or not convert back to numpy

        if not sk.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not sk.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)

        level = ct.level
        sk_data = sk.data[0][self.ntt.starts[level][0] :]

        if isinstance(ct, CiphertextTriplet):
            if not ct.ntt_state:
                raise errors.NTTStateError(expected=True)
            if not ct.montgomery_state:
                raise errors.MontgomeryStateError(expected=True)

            d0 = [ct.data[0][0].clone()]
            d1 = [ct.data[1][0]]
            d2 = [ct.data[2][0]]

            self.ntt.intt_exit_reduce(d0, level)

            sk_data = [sk.data[0][self.ntt.starts[level][0] :]]

            d1_s = self.ntt.mont_mult(d1, sk_data, level)

            s2 = self.ntt.mont_mult(sk_data, sk_data, level)
            d2_s2 = self.ntt.mont_mult(d2, s2, level)

            self.ntt.intt_exit(d1_s, level)
            self.ntt.intt_exit(d2_s2, level)

            pt = self.ntt.mont_add(d0, d1_s, level)
            pt = self.ntt.mont_add(pt, d2_s2, level)
            self.ntt.reduce_2q(pt, level)
        else:
            if not isinstance(ct, Ciphertext):
                logger.warning(
                    f"ct type is mismatched: excepted {Ciphertext} or {CiphertextTriplet}, got {type(ct)}, will try to decode as ct anyway."
                )
            if ct.ntt_state:
                raise errors.NTTStateError(expected=False)
            if ct.montgomery_state:
                raise errors.MontgomeryStateError(expected=False)

            ct0 = ct.data[0][0]
            a = ct.data[1][0].clone()

            self.ntt.enter_ntt([a], level)

            sa = self.ntt.mont_mult([a], [sk_data], level)
            self.ntt.intt_exit(sa, level)

            pt = self.ntt.mont_add([ct0], sa, level)
            self.ntt.reduce_2q(pt, level)

        base_at = (
            -self.ctx.num_special_primes - 1 if ct.include_special else -1
        )
        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        len_left = len(self.ntt.p.destination_arrays[level][0])

        if (len_left >= 3) and self.bias_guard:
            dc0 = base[0][0].item()
            dc1 = scaler[0][0].item()
            dc2 = pt[0][1][0].item()

            base[0][0] = 0
            scaler[0][0] = 0

            q0_ind = self.ntt.p.destination_arrays[level][0][base_at]
            q1_ind = self.ntt.p.destination_arrays[level][0][0]
            q2_ind = self.ntt.p.destination_arrays[level][0][1]

            q0 = self.ctx.q[q0_ind]
            q1 = self.ctx.q[q1_ind]
            q2 = self.ctx.q[q2_ind]

            Q = q0 * q1 * q2
            Q0 = q1 * q2
            Q1 = q0 * q2
            Q2 = q0 * q1

            Qi0 = pow(Q0, -1, q0)
            Qi1 = pow(Q1, -1, q1)
            Qi2 = pow(Q2, -1, q2)

            dc = (dc0 * Qi0 * Q0 + dc1 * Qi1 * Q1 + dc2 * Qi2 * Q2) % Q

            half_Q = Q // 2
            dc = dc if dc <= half_Q else dc - Q

            dc = (dc + (q1 - 1)) // q1

        final_scalar = self.final_scalar[level]
        scaled = self.ntt.mont_sub([base], [scaler], -1)
        self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
        self.ntt.reduce_2q(scaled, -1)
        self.ntt.make_signed(scaled, -1)

        # Round?
        if final_round:
            # The scaler and the base channels are guaranteed to be in the
            # device 0.
            rounding_prime = self.ntt.qlists[0][
                -self.ctx.num_special_primes - 2
            ]
            rounder = (scaler[0] > (rounding_prime // 2)) * 1
            scaled[0] += rounder

        # Decoding.
        correction = self.corrections[level]
        decoded = codec_decode(
            scaled[0][-1],
            scale=self.scale,
            correction=correction,
            norm=self.norm,
            return_without_scaling=self.bias_guard,
        )
        decoded = decoded[: self.ctx.N // 2].cpu().numpy()
        ##

        decoded = decoded / self.scale * correction

        # Bias guard.
        if (len_left >= 3) and self.bias_guard:
            decoded += dc / self.scale * correction
        if is_real:
            decoded = decoded.real
        return decoded

    # -------------------------------------------------------------------------------------------
    # Conjugation
    # -------------------------------------------------------------------------------------------

    @strictype
    def create_conjugation_key(self, sk: SecretKey) -> ConjugationKey:

        if not sk.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not sk.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)

        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data)
        sk_new_data = [codec_conjugate(s) for s in sk_new_data]
        self.ntt.ntt(sk_new_data)
        sk_rotated = SecretKey(
            data=sk_new_data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            level=0,
        )
        rotk = ConjugationKey.wrap(
            self.create_key_switching_key(sk_rotated, sk)
        )
        # rotk = rotk.origin = origin_names["conjk"]
        return rotk

    @strictype
    def conjugate(self, ct: Ciphertext, conjk: ConjugationKey) -> Ciphertext:
        level = ct.level
        conj_ct_data = [
            [codec_conjugate(d) for d in ct_data] for ct_data in ct.data
        ]

        conj_ct_sk = Ciphertext(
            data=conj_ct_data,
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            level=level,
        )

        conj_ct = self.switch_key(conj_ct_sk, conjk)
        return conj_ct

    # -------------------------------------------------------------------------------------------
    # Move data back and forth from GPUs to the CPU.
    # -------------------------------------------------------------------------------------------

    # todo rewrite movetos

    def download_to_cpu(self, gpu_data, level, include_special):
        # Prepare destination arrays.
        if include_special:
            dest = self.ntt.p.destination_arrays_with_special[level]
        else:
            dest = self.ntt.p.destination_arrays[level]

        # dest contain indices that are the absolute order of primes.
        # Convert them to tensor channel indices at this level.
        # That is, force them to start from zero.
        min_ind = min([min(d) for d in dest])
        dest = [[di - min_ind for di in d] for d in dest]

        # Tensor size parameters.
        num_rows = sum([len(d) for d in dest])
        num_cols = self.ctx.N
        cpu_size = (num_rows, num_cols)

        # Make a cpu tensor to aggregate the data in GPUs.
        cpu_tensor = torch.empty(
            cpu_size, dtype=self.ctx.torch_dtype, device="cpu"
        )

        for ten, dest_i in zip(gpu_data, dest):
            # Check if the tensor is in the gpu.
            if ten.device.type != "cuda":
                raise Exception(
                    "To download data to the CPU, it must already be in a GPU!!!"
                )

            # Copy in the data.
            cpu_tensor[dest_i] = ten.cpu()

        # To avoid confusion, make a list with a single element (only one device, that is the CPU),
        # and return it.
        return [cpu_tensor]

    def upload_to_gpu(self, cpu_data, level, include_special):
        # There's only one device data in the cpu data.
        cpu_tensor = cpu_data[0]

        # Check if the tensor is in the cpu.
        if cpu_tensor.device.type != "cpu":
            raise Exception(
                "To upload data to GPUs, it must already be in the CPU!!!"
            )

        # Prepare destination arrays.
        if include_special:
            dest = self.ntt.p.destination_arrays_with_special[level]
        else:
            dest = self.ntt.p.destination_arrays[level]

        # dest contain indices that are the absolute order of primes.
        # Convert them to tensor channel indices at this level.
        # That is, force them to start from zero.
        min_ind = min([min(d) for d in dest])
        dest = [[di - min_ind for di in d] for d in dest]

        gpu_data = []
        for device_id in range(len(dest)):
            # Copy in the data.
            dest_device = dest[device_id]
            device = self.ntt.devices[device_id]
            gpu_tensor = cpu_tensor[dest_device].to(device=device)

            # Append to the gpu_data list.
            gpu_data.append(gpu_tensor)

        return gpu_data

    def move_tensors(self, data, level, include_special, direction):
        func = {
            "gpu2cpu": self.download_to_cpu,
            "cpu2gpu": self.upload_to_gpu,
        }[direction]

        # Some data has 1 depth.
        if not isinstance(data[0], list):
            moved = func(data, level, include_special)
            new_data = moved
        else:
            new_data = []
            for part in data:
                moved = func(part, level, include_special)
                new_data.append(moved)
        return new_data

    def move_to(self, text, direction="gpu2cpu"):
        if not isinstance(text.data[0], DataStruct):
            level = text.level
            include_special = text.include_special

            # data are tensors.
            data = self.move_tensors(
                text.data, level, include_special, direction
            )

            wrapper = DataStruct(
                data,
                text.include_special,
                text.ntt_state,
                text.montgomery_state,
                text.origin,
                text.level,
                text.hash,
                text.version,
            )

        else:
            wrapper = DataStruct(
                [],
                text.include_special,
                text.ntt_state,
                text.montgomery_state,
                text.origin,
                text.level,
                text.hash,
                text.version,
            )

            for d in text.data:
                moved = self.move_to(d, direction)
                wrapper.data.append(moved)

        return wrapper

        # Shortcuts

    def cpu(self, ct):
        return self.move_to(ct, "gpu2cpu")

    def cuda(self, ct):
        return self.move_to(ct, "cpu2gpu")

    # -------------------------------------------------------------------------------------------
    # Save and load.
    # -------------------------------------------------------------------------------------------

    def auto_generate_filename(self, fmt_str="%Y%m%d%H%M%s%f"):
        return datetime.datetime.now().strftime(fmt_str) + ".pkl"

    def save(self, text, filename=None):
        if filename is None:
            filename = self.auto_generate_filename()

        savepath = Path(filename)

        # Check if the text is in the CPU.
        # If not, move to CPU.
        if self.device(text) != "cpu":
            cpu_text = self.cpu(text)
        else:
            cpu_text = text

        with savepath.open("wb") as f:
            pickle.dump(cpu_text, f)

    def load(self, filename, move_to_gpu=True):
        savepath = Path(filename)
        with savepath.open("rb") as f:
            # gc.disable()
            cpu_text = pickle.load(f)
            # gc.enable()

        if move_to_gpu:
            text = self.cuda(cpu_text)
        else:
            text = cpu_text

        return text

    # -------------------------------------------------------------------------------------------
    # Negate.
    # -------------------------------------------------------------------------------------------

    @strictype
    def negate(self, ct: Ciphertext) -> Ciphertext:
        neg_ct = ct.clone()

        for part in neg_ct.data:
            for d in part:
                d *= -1
            self.ntt.make_signed(part, ct.level)

        return neg_ct

    # -------------------------------------------------------------------------------------------
    # scalar ops.
    # -------------------------------------------------------------------------------------------

    @strictype
    def pc_add(
        self,
        pt: Union[list, Plaintext],
        ct: Ciphertext,
        inplace=False,
    ):
        # process cache
        
        if not isinstance(pt, Plaintext):  # legacy pt
            # deprecated warning
            warnings.warn(
                "The use of a list as a plaintext is deprecated. Please use the Plaintext class instead.",
                DeprecationWarning,
            )  # todo remove all legacy usage
            pt_ = self.ntt.tile_unsigned(pt, ct.level)
            self.ntt.mont_enter_scale(pt_, ct.level)
        else:  # its Plaintext
            if not str(self.pc_add) in pt.cache[ct.level]:
                m = pt.src * math.sqrt(self.deviations[ct.level + 1])
                pt_ = self.encode(m, ct.level)
                pt_ = self.ntt.tile_unsigned(pt_, ct.level)
                self.ntt.mont_enter_scale(pt_, ct.level)
                pt.cache[ct.level][str(self.pc_add)] = pt_
            pt_ = pt.cache[ct.level][
                str(self.pc_add)
            ]  # todo does rewrite to auto trace impact performance?

        # process ct
        
        new_ct = ct.clone() if not inplace else ct

        self.ntt.mont_enter(new_ct.data[0], ct.level)
        new_d0 = self.ntt.mont_add(pt_, new_ct.data[0], ct.level)
        self.ntt.mont_redc(new_d0, ct.level)
        self.ntt.reduce_2q(new_d0, ct.level)
        new_ct.data[0] = new_d0
        return new_ct

    @strictype
    def pc_mult(
        self,
        pt: Union[list, Plaintext],
        ct: Ciphertext,
        inplace=False,
        post_rescale=True,
    ):
        # process cache
        
        if not isinstance(pt, Plaintext):  # legacy pt
            # deprecated warning
            warnings.warn(
                "The use of a list as a plaintext is deprecated. Please use the Plaintext class instead.",
                DeprecationWarning,
            )  # todo remove all legacy usage
            pt_ = self.ntt.tile_unsigned(pt, ct.level)
            self.ntt.enter_ntt(pt_, ct.level)
        else:  # its Plaintext
            if not str(self.pc_mult) in pt.cache[ct.level]:
                m = pt.src * math.sqrt(self.deviations[ct.level + 1])
                pt_ = self.encode(m, ct.level)
                pt_ = self.ntt.tile_unsigned(pt_, ct.level)
                self.ntt.enter_ntt(pt_, ct.level)
                pt.cache[ct.level][str(self.pc_mult)] = pt_
            pt_ = pt.cache[ct.level][
                str(self.pc_mult)
            ]  # todo does rewrite to auto trace impact performance?

        # process ct
        
        new_ct = ct.clone() if not inplace else ct

        self.ntt.enter_ntt(new_ct.data[0], ct.level)
        self.ntt.enter_ntt(new_ct.data[1], ct.level)

        new_d0 = self.ntt.mont_mult(pt_, new_ct.data[0], ct.level)
        new_d1 = self.ntt.mont_mult(pt_, new_ct.data[1], ct.level)

        self.ntt.intt_exit_reduce(new_d0, ct.level)
        self.ntt.intt_exit_reduce(new_d1, ct.level)

        new_ct.data[0] = new_d0
        new_ct.data[1] = new_d1

        if post_rescale:
            new_ct = self.rescale(new_ct)
        return new_ct

    @strictype
    def mult_int_scalar(self, ct: Ciphertext, scalar) -> Ciphertext:

        device_len = len(ct.data[0])

        int_scalar = int(scalar)
        mont_scalar = [(int_scalar * self.ctx.R) % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[ct.level]

        partitioned_mont_scalar = [
            [mont_scalar[i] for i in desti] for desti in dest
        ]
        tensorized_scalar = []
        for device_id in range(device_len):
            scal_tensor = torch.tensor(
                partitioned_mont_scalar[device_id],
                dtype=self.ctx.torch_dtype,
                device=self.ntt.devices[device_id],
            )
            tensorized_scalar.append(scal_tensor)

        new_ct = ct.clone()
        new_data = new_ct.data
        for i in [0, 1]:
            self.ntt.mont_enter_scalar(
                new_data[i], tensorized_scalar, ct.level
            )
            self.ntt.reduce_2q(new_data[i], ct.level)

        return new_ct

    @strictype
    def mult_scalar(self, ct: Ciphertext, scalar) -> Ciphertext:

        device_len = len(ct.data[0])

        scaled_scalar = int(
            scalar * self.scale * np.sqrt(self.deviations[ct.level + 1]) + 0.5
        )

        mont_scalar = [(scaled_scalar * self.ctx.R) % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[ct.level]

        partitioned_mont_scalar = [
            [mont_scalar[i] for i in dest_i] for dest_i in dest
        ]
        tensorized_scalar = []
        for device_id in range(device_len):
            scal_tensor = torch.tensor(
                partitioned_mont_scalar[device_id],
                dtype=self.ctx.torch_dtype,
                device=self.ntt.devices[device_id],
            )
            tensorized_scalar.append(scal_tensor)

        # todo encode scalar should be done in encode function and produce a Plaintext

        new_ct = ct.clone()
        new_data = new_ct.data

        for i in [0, 1]:
            self.ntt.mont_enter_scalar(
                new_data[i], tensorized_scalar, ct.level
            )
            self.ntt.reduce_2q(new_data[i], ct.level)

        return self.rescale(new_ct)

    @strictype
    def add_scalar(self, ct: Ciphertext, scalar) -> Ciphertext:

        device_len = len(ct.data[0])

        scaled_scalar = int(
            scalar * self.scale * self.deviations[ct.level] + 0.5
        )

        if self.norm == "backward":
            scaled_scalar *= self.ctx.N

        scaled_scalar *= self.int_scale

        scaled_scalar = [scaled_scalar % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[ct.level]

        partitioned_mont_scalar = [
            [scaled_scalar[i] for i in desti] for desti in dest
        ]
        tensorized_scalar = []
        for device_id in range(device_len):
            scal_tensor = torch.tensor(
                partitioned_mont_scalar[device_id],
                dtype=self.ctx.torch_dtype,
                device=self.ntt.devices[device_id],
            )
            tensorized_scalar.append(scal_tensor)

        new_ct = ct.clone()
        new_data = new_ct.data

        dc = [d[:, 0] for d in new_data[0]]
        for device_id in range(device_len):
            dc[device_id] += tensorized_scalar[device_id]

        self.ntt.reduce_2q(new_data[0], ct.level)

        return new_ct

    # todo try to rewrite ops into the data structure as override methods

    # def sub_scalar(self, ct, scalar):
    #     return self.add_scalar(ct, -scalar)

    # def int_scalar_mult(self, scalar, ct, evk=None, relin=True):
    #     return self.mult_int_scalar(ct, scalar)

    # def scalar_mult(self, scalar, ct, evk=None, relin=True):
    #     return self.mult_scalar(ct, scalar)

    # def scalar_add(self, scalar, ct):
    #     return self.add_scalar(ct, scalar)

    # def scalar_sub(self, scalar, ct):
    #     neg_ct = self.negate(ct)
    #     return self.add_scalar(neg_ct, scalar)

    # -------------------------------------------------------------------------------------------
    # message ops.
    # -------------------------------------------------------------------------------------------

    @strictype
    def mc_mult(
        self, m, ct: Ciphertext, inplace=False, post_rescale=True
    ) -> Ciphertext:
        return self.pc_mult(
            pt=Plaintext(m),
            ct=ct,
            inplace=inplace,
            post_rescale=post_rescale,
        )

    @strictype
    def mc_add(self, m, ct: Ciphertext, inplace=False) -> Ciphertext:
        return self.pc_add(pt=Plaintext(m), ct=ct, inplace=inplace)

    # todo overwrite ops in the data structure, in a new file with @patch

    # def mc_sub(self, m, ct):
    #     neg_ct = self.negate(ct)
    #     return self.mc_add(m, neg_ct)

    # def cm_mult(self, ct, m, evk=None, relin=True):
    #     return self.mc_mult(m, ct)

    # def cm_add(self, ct, m):
    #     return self.mc_add(m, ct)

    # def cm_sub(self, ct, m):
    #     return self.mc_add(-np.array(m), ct)

    # -------------------------------------------------------------------------------------------
    # Misc.
    # -------------------------------------------------------------------------------------------

    def align_level(self, ct0, ct1):
        level_diff = ct0.level - ct1.level
        if level_diff < 0:
            new_ct0 = self.level_up(ct0, ct1.level)
            return new_ct0, ct1
        elif level_diff > 0:
            new_ct1 = self.level_up(ct1, ct0.level)
            return ct0, new_ct1
        else:
            return ct0, ct1

    def refresh(self):
        # Refreshes the rng state.
        self.rng.refresh()

    def reduce_error(self, ct):
        # Reduce the accumulated error in the cipher text.
        return self.mult_scalar(ct, 1.0)

    @strictype
    def sum(self, ct: Ciphertext, gk: GaloisKey) -> Ciphertext:
        new_ct = ct.clone()
        for roti in range(self.ctx.logN - 1):
            rotk = gk.data[roti]
            rot_ct = self.rotate_single(new_ct, rotk)
            new_ct = self.cc_add(rot_ct, new_ct)
        return new_ct

    @strictype
    def mean(self, ct: Ciphertext, gk: GaloisKey, alpha=1):
        # Divide by num_slots.
        # The cipher text is refreshed here, and hence
        # doesn't beed to be refreshed at roti=0 in the loop.
        new_ct = self.mc_mult(m=1 / self.num_slots / alpha, ct=ct)
        for roti in range(self.ctx.logN - 1):
            rotk = gk.data[roti]
            rot_ct = self.rotate_single(new_ct, rotk)
            new_ct = self.cc_add(rot_ct, new_ct)
        return new_ct

    @strictype
    def cov(
        self,
        ct_a: Ciphertext,
        ct_b: Ciphertext,
        evk: EvaluationKey,
        gk: GaloisKey,
    ) -> Ciphertext:
        cta_mean = self.mean(ct_a, gk)
        ctb_mean = self.mean(ct_b, gk)

        cta_dev = self.cc_sub(ct_a, cta_mean)
        ctb_dev = self.cc_sub(ct_b, ctb_mean)

        ct_cov = self.mc_mult(
            ct=self.cc_mult(cta_dev, ctb_dev, evk), m=1 / (self.num_slots - 1)
        )
        return ct_cov

    @strictype
    def pow(
        self, ct: Ciphertext, power: int, evk: EvaluationKey
    ) -> Ciphertext:
        current_exponent = 2
        pow_list = [ct]
        while current_exponent <= power:
            current_ct = pow_list[-1]
            new_ct = self.cc_mult(current_ct, current_ct, evk)
            pow_list.append(new_ct)
            current_exponent *= 2

        remaining_exponent = power - current_exponent // 2
        new_ct = pow_list[-1]

        while remaining_exponent > 0:
            pow_ind = math.floor(math.log2(remaining_exponent))
            pow_term = pow_list[pow_ind]
            # align level
            new_ct, pow_term = self.align_level(new_ct, pow_term)
            new_ct = self.cc_mult(new_ct, pow_term, evk)
            remaining_exponent -= 2**pow_ind

        return new_ct

    @strictype
    def square(
        self, ct: Ciphertext, evk: EvaluationKey, relin=True
    ) -> Union[CiphertextTriplet, Ciphertext]:
        x = self.rescale(ct)

        level = x.level

        # Multiply.
        x0, x1 = x.data

        self.ntt.enter_ntt(x0, level)
        self.ntt.enter_ntt(x1, level)

        d0 = self.ntt.mont_mult(x0, x0, level)
        x0y1 = self.ntt.mont_mult(x0, x1, level)
        d2 = self.ntt.mont_mult(x1, x1, level)

        d1 = self.ntt.mont_add(x0y1, x0y1, level)

        ct_mult = CiphertextTriplet(
            data=[d0, d1, d2],
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            level=level,
        )
        if relin:
            ct_mult = self.relinearize(ct_triplet=ct_mult, evk=evk)

        return ct_mult

    #### -------------------------------------------------------------------------------------------
    ####  Statistics
    #### -------------------------------------------------------------------------------------------

    @strictype
    def sqrt(
        self, ct: Ciphertext, evk: EvaluationKey, e=0.0001, alpha=0.0001
    ) -> Ciphertext:
        a = ct.clone()
        b = ct.clone()

        while e <= 1 - alpha:
            k = float(np.roots([1 - e**3, -6 + 6 * e**2, 9 - 9 * e])[1])
            t = self.mult_scalar(a, k, evk)
            b0 = self.sub_scalar(t, 3)
            b1 = self.mult_scalar(b, (k**0.5) / 2, evk)
            b = self.cc_mult(b0, b1, evk)

            a0 = self.mult_scalar(a, (k**3) / 4)
            t = self.sub_scalar(a, 3 / k)
            a1 = self.square(t, evk)
            a = self.cc_mult(a0, a1, evk)
            e = k * (3 - k) ** 2 / 4

        return b

    @strictype
    def var(
        self, ct: Ciphertext, evk: EvaluationKey, gk: GaloisKey, relin=False
    ) -> Ciphertext:
        ct_mean = self.mean(ct=ct, gk=gk)
        dev = self.sub(ct, ct_mean)
        dev = self.square(ct=dev, evk=evk, relin=relin)
        if not relin:
            dev = self.relinearize(ct_triplet=dev, evk=evk)
        ct_var = self.mean(ct=dev, gk=gk)
        return ct_var

    def std(
        self, ct: Ciphertext, evk: EvaluationKey, gk: GaloisKey, relin=False
    ) -> Ciphertext:
        ct_var = self.var(ct=ct, evk=evk, gk=gk, relin=relin)
        ct_std = self.sqrt(ct=ct_var, evk=evk)
        return ct_std
