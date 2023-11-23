import datetime
import gc
import math
import pickle
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

#  from context.ckks_context import ckks_context
from .context.ckks_context import ckks_context
from .data_struct import data_struct
from .encdec import decode, encode, rotate, conjugate
from .version import VERSION
from .presets import types, errors
from liberate.ntt import ntt_context
from liberate.ntt import ntt_cuda
from liberate.csprng import Csprng


class ckks_engine:
    @errors.log_error
    def __init__(self, devices: list[int] = None, verbose: bool = False,
                 bias_guard: bool = True, norm: str = 'forward', **ctx_params):
        """
            buffer_bit_length=62,
            scale_bits=40,
            logN=15,
            num_scales=None,
            num_special_primes=2,
            sigma=3.2,
            uniform_tenary_secret=True,
            cache_folder='cache/',
            security_bits=128,
            quantum='post_quantum',
            distribution='uniform',
            read_cache=True,
            save_cache=True,
            verbose=False
        """

        self.bias_guard = bias_guard

        self.norm = norm

        self.version = VERSION

        self.ctx = ckks_context(**ctx_params)
        self.ntt = ntt_context(self.ctx, devices=devices, verbose=verbose)

        if self.bias_guard:
            if self.ctx.num_special_primes < 2:
                raise errors.NotEnoughPrimesForBiasGuard(
                    bias_guard=self.bias_guard,
                    num_special_primes=self.ctx.num_special_primes)

        self.num_levels = self.ntt.num_levels - 1

        self.num_slots = self.ctx.N // 2

        rng_repeats = max(self.ntt.num_special_primes, 2)
        self.rng = Csprng(self.ntt.ctx.N, [len(di) for di in self.ntt.p.d], rng_repeats, devices=self.ntt.devices)

        self.int_scale = 2 ** self.ctx.scale_bits
        self.scale = np.float64(self.int_scale)

        qstr = ','.join([str(qi) for qi in self.ctx.q])
        hashstr = (self.ctx.generation_string + "_" + qstr).encode("utf-8")
        self.hash = sha256(bytes(hashstr)).hexdigest()

        self.make_adjustments_and_corrections()

        self.device0 = self.ntt.devices[0]

        self.make_mont_PR()

        self.reserve_ksk_buffers()

        self.create_ksk_rescales()

        self.alloc_parts()

        self.leveled_devices()

        self.create_rescale_scales()

        self.galois_deltas = [2 ** i for i in range(self.ctx.logN - 1)]

        self.mult_dispatch_dict = {
            (data_struct, data_struct): self.auto_cc_mult,
            (list, data_struct): self.mc_mult,
            (np.ndarray, data_struct): self.mc_mult,
            (data_struct, np.ndarray): self.cm_mult,
            (data_struct, list): self.cm_mult,
            (float, data_struct): self.scalar_mult,
            (data_struct, float): self.mult_scalar,
            (int, data_struct): self.int_scalar_mult,
            (data_struct, int): self.mult_int_scalar
        }

        self.add_dispatch_dict = {
            (data_struct, data_struct): self.auto_cc_add,
            (list, data_struct): self.mc_add,
            (np.ndarray, data_struct): self.mc_add,
            (data_struct, np.ndarray): self.cm_add,
            (data_struct, list): self.cm_add,
            (float, data_struct): self.scalar_add,
            (data_struct, float): self.add_scalar,
            (int, data_struct): self.scalar_add,
            (data_struct, int): self.add_scalar
        }

        self.sub_dispatch_dict = {
            (data_struct, data_struct): self.auto_cc_sub,
            (list, data_struct): self.mc_sub,
            (np.ndarray, data_struct): self.mc_sub,
            (data_struct, np.ndarray): self.cm_sub,
            (data_struct, list): self.cm_sub,
            (float, data_struct): self.scalar_sub,
            (data_struct, float): self.sub_scalar,
            (int, data_struct): self.scalar_sub,
            (data_struct, int): self.sub_scalar
        }

    # -------------------------------------------------------------------------------------------
    # Various pre-calculations.
    # -------------------------------------------------------------------------------------------
    def create_rescale_scales(self):
        self.rescale_scales = []
        for level in range(self.num_levels):
            self.rescale_scales.append([])

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

                    scales = torch.tensor(scales,
                                          dtype=self.ctx.torch_dtype,
                                          device=self.ntt.devices[device_id])
                    self.rescale_scales[level].append(scales)

    def leveled_devices(self):
        self.len_devices = []
        for level in range(self.num_levels):
            self.len_devices.append(len([[a] for a in self.ntt.p.p[level] if len(a) > 0]))

        self.neighbor_devices = []
        for level in range(self.num_levels):
            self.neighbor_devices.append([])
            len_devices_at = self.len_devices[level]
            available_devices_ids = range(len_devices_at)
            for src_device_id in available_devices_ids:
                neighbor_devices_at = [
                    device_id for device_id in available_devices_ids if device_id != src_device_id
                ]
                self.neighbor_devices[level].append(neighbor_devices_at)

    def alloc_parts(self):
        self.parts_alloc = []
        for level in range(self.num_levels):
            num_parts = [len(parts) for parts in self.ntt.p.p[level]]
            parts_alloc = [
                alloc[-num_parts[di] - 1:-1] for di, alloc in enumerate(self.ntt.p.part_allocations)
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

    def create_ksk_rescales(self):
        R = self.ctx.R
        P = self.ctx.q[-self.ntt.num_special_primes:][::-1]
        m = self.ctx.q
        PiR = [[(pow(Pj, -1, mi) * R) % mi for mi in m[:-P_ind - 1]] for P_ind, Pj in enumerate(P)]

        self.PiRs = []

        level = 0
        self.PiRs.append([])

        for P_ind in range(self.ntt.num_special_primes):
            self.PiRs[level].append([])

            for device_id in range(self.ntt.num_devices):
                dest = self.ntt.p.destination_arrays_with_special[level][device_id]
                PiRi = [PiR[P_ind][i] for i in dest[:-P_ind - 1]]
                PiRi = torch.tensor(PiRi,
                                    device=self.ntt.devices[device_id],
                                    dtype=self.ctx.torch_dtype)
                self.PiRs[level][P_ind].append(PiRi)

        for level in range(1, self.num_levels):
            self.PiRs.append([])

            for P_ind in range(self.ntt.num_special_primes):

                self.PiRs[level].append([])

                for device_id in range(self.ntt.num_devices):
                    start = self.ntt.starts[level][device_id]
                    PiRi = self.PiRs[0][P_ind][device_id][start:]

                    self.PiRs[level][P_ind].append(PiRi)

    def reserve_ksk_buffers(self):
        self.ksk_buffers = []
        for device_id in range(self.ntt.num_devices):
            self.ksk_buffers.append([])
            for part_id in range(len(self.ntt.p.p[0][device_id])):
                buffer = torch.empty(
                    [self.ntt.num_special_primes, self.ctx.N],
                    dtype=self.ctx.torch_dtype
                ).pin_memory()
                self.ksk_buffers[device_id].append(buffer)

    def make_mont_PR(self):
        P = math.prod(self.ntt.ctx.q[-self.ntt.num_special_primes:])
        R = self.ctx.R
        PR = P * R
        self.mont_PR = []
        for device_id in range(self.ntt.num_devices):
            dest = self.ntt.p.destination_arrays[0][device_id]
            m = [self.ctx.q[i] for i in dest]
            PRm = [PR % mi for mi in m]
            PRm = torch.tensor(PRm,
                               device=self.ntt.devices[device_id],
                               dtype=self.ctx.torch_dtype)
            self.mont_PR.append(PRm)

    def make_adjustments_and_corrections(self):

        self.alpha = [(self.scale / np.float64(q)) ** 2 for q in self.ctx.q[:self.ctx.num_scales]]
        self.deviations = [1]
        for al in self.alpha:
            self.deviations.append(self.deviations[-1] ** 2 * al)

        self.final_q_ind = [da[0][0] for da in self.ntt.p.destination_arrays[:-1]]
        self.final_q = [self.ctx.q[ind] for ind in self.final_q_ind]
        self.final_alpha = [(self.scale / np.float64(q)) for q in self.final_q]
        self.corrections = [1 / (d * fa) for d, fa in zip(self.deviations, self.final_alpha)]

        self.base_prime = self.ctx.q[self.ntt.p.base_prime_idx]

        self.final_scalar = []
        for qi, q in zip(self.final_q_ind, self.final_q):
            scalar = (pow(q, -1, self.base_prime) * self.ctx.R) % self.base_prime
            scalar = torch.tensor([scalar],
                                  device=self.ntt.devices[0],
                                  dtype=self.ctx.torch_dtype)
            self.final_scalar.append(scalar)

    # -------------------------------------------------------------------------------------------
    # Example generation.
    # -------------------------------------------------------------------------------------------

    def absmax_error(self, x, y):
        if type(x[0]) == np.complex128 and type(y[0]) == np.complex128:
            r = np.abs(x.real - y.real).max() + np.abs(x.imag - y.imag).max() * 1j
        else:
            r = np.abs(np.array(x) - np.array(y)).max()
        return r

    def integral_bits_available(self):
        base_prime = self.base_prime
        max_bits = math.floor(math.log2(base_prime))
        integral_bits = max_bits - self.ctx.scale_bits
        return integral_bits

    @errors.log_error
    def example(self, amin=None, amax=None, decimal_places: int = 10) -> np.array:
        if amin is None:
            amin = -(2 ** self.integral_bits_available())

        if amax is None:
            amax = 2 ** self.integral_bits_available()

        base = 10 ** decimal_places
        a = np.random.randint(amin * base, amax * base, self.ctx.N // 2) / base
        b = np.random.randint(amin * base, amax * base, self.ctx.N // 2) / base

        sample = a + b * 1j

        return sample

    # -------------------------------------------------------------------------------------------
    # Encode/Decode
    # -------------------------------------------------------------------------------------------

    def padding(self, m):
        # m = m[:self.num_slots]
        try:
            m_len = len(m)
            padding_result = np.pad(m, (0, self.num_slots - m_len), constant_values=(0, 0))
        except TypeError as e:
            m_len = len([m])
            padding_result = np.pad([m], (0, self.num_slots - m_len), constant_values=(0, 0))
        except Exception as e:
            raise Exception("[Error] encoding Padding Error.")
        return padding_result

    @errors.log_error
    def encode(self, m, level: int = 0, padding=True) -> list[torch.Tensor]:
        """
            Encode a plain message m, using an encoding function.
            Note that the encoded plain text is pre-permuted to yield cyclic rotation.
        """
        deviation = self.deviations[level]
        if padding:
            m = self.padding(m)
        encoded = [encode(m, scale=self.scale, rng=self.rng,
                          device=self.device0,
                          deviation=deviation, norm=self.norm)]

        pt_buffer = self.ksk_buffers[0][0][0]
        pt_buffer.copy_(encoded[-1])
        for dev_id in range(1, self.ntt.num_devices):
            encoded.append(pt_buffer.cuda(self.ntt.devices[dev_id]))
        return encoded

    @errors.log_error
    def decode(self, m, level=0, is_real: bool = False) -> list:
        """
            Base prime is located at -1 of the RNS channels in GPU0.
            Assuming this is an orginary RNS deinclude_special.
        """
        correction = self.corrections[level]
        decoded = decode(m[0].squeeze(), scale=self.scale, correction=correction, norm=self.norm)
        m = decoded[:self.ctx.N // 2].cpu().numpy()
        if is_real:
            m = m.real
        return m

    # -------------------------------------------------------------------------------------------
    # secret key/public key generation.
    # -------------------------------------------------------------------------------------------

    @errors.log_error
    def create_secret_key(self, include_special: bool = True) -> data_struct:
        uniform_ternary = self.rng.randint(amax=3, shift=-1, repeats=1)

        mult_type = -2 if include_special else -1
        unsigned_ternary = self.ntt.tile_unsigned(uniform_ternary, lvl=0, mult_type=mult_type)
        self.ntt.enter_ntt(unsigned_ternary, 0, mult_type)

        return data_struct(
            data=unsigned_ternary,
            include_special=include_special,
            montgomery_state=True,
            ntt_state=True,
            origin=types.origins["sk"],
            level=0,
            hash=self.hash,
            version=self.version
        )

    @errors.log_error
    def create_public_key(self, sk: data_struct, include_special: bool = False,
                          a: list[torch.Tensor] = None) -> data_struct:
        """
            Generates a public key against the secret key sk.
            pk = -a * sk + e = e - a * sk
        """
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
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
                self.ntt.q_prepack[mult_type][level][0],
                repeats=repeats
            )

        sa = self.ntt.mont_mult(a, sk.data, 0, mult_type)
        pk0 = self.ntt.mont_sub(e, sa, 0, mult_type)

        return data_struct(
            data=(pk0, a),
            include_special=include_special,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["pk"],
            level=0,
            hash=self.hash,
            version=self.version
        )

    # -------------------------------------------------------------------------------------------
    # Encrypt/Decrypt
    # -------------------------------------------------------------------------------------------

    @errors.log_error
    def encrypt(self, pt: list[torch.Tensor], pk: data_struct, level: int = 0) -> data_struct:
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
        if pk.origin != types.origins["pk"]:
            raise errors.NotMatchType(origin=pk.origin, to=types.origins["pk"])

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
        pk0 = [pk.data[0][di][start[di]:] for di in range(self.ntt.num_devices)]
        pk1 = [pk.data[1][di][start[di]:] for di in range(self.ntt.num_devices)]

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

        ct = data_struct(
            data=(ct0, ct1),
            include_special=mult_type == -2,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level=level,
            hash=self.hash,
            version=self.version
        )

        return ct

    def decrypt_triplet(self, ct_mult: data_struct, sk: data_struct) -> list[torch.Tensor]:
        if ct_mult.origin != types.origins["ctt"]:
            raise errors.NotMatchType(origin=ct_mult.origin, to=types.origins["ctt"])
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        if not ct_mult.ntt_state or not ct_mult.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct_mult.origin)
        if (not sk.ntt_state) or (not sk.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk.origin)

        level = ct_mult.level
        d0 = [ct_mult.data[0][0].clone()]
        d1 = [ct_mult.data[1][0]]
        d2 = [ct_mult.data[2][0]]

        self.ntt.intt_exit_reduce(d0, level)

        sk_data = [sk.data[0][self.ntt.starts[level][0]:]]

        d1_s = self.ntt.mont_mult(d1, sk_data, level)

        s2 = self.ntt.mont_mult(sk_data, sk_data, level)
        d2_s2 = self.ntt.mont_mult(d2, s2, level)

        self.ntt.intt_exit(d1_s, level)
        self.ntt.intt_exit(d2_s2, level)

        pt = self.ntt.mont_add(d0, d1_s, level)
        pt = self.ntt.mont_add(pt, d2_s2, level)
        self.ntt.reduce_2q(pt, level)

        base_at = -self.ctx.num_special_primes - 1 if ct_mult.include_special else -1

        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        final_scalar = self.final_scalar[level]
        scaled = self.ntt.mont_sub([base], [scaler], -1)
        self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
        self.ntt.reduce_2q(scaled, -1)
        self.ntt.make_signed(scaled, -1)
        return scaled

    def decrypt_double(self, ct: data_struct, sk: data_struct) -> list[torch.Tensor]:
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if ct.ntt_state or ct.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct.origin)
        if not sk.ntt_state or not sk.montgomery_state:
            raise errors.NotMatchDataStructState(origin=sk.origin)

        ct0 = ct.data[0][0]
        level = ct.level
        sk_data = sk.data[0][self.ntt.starts[level][0]:]
        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)
        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        pt = self.ntt.mont_add([ct0], sa, level)
        self.ntt.reduce_2q(pt, level)

        base_at = -self.ctx.num_special_primes - 1 if ct.include_special else -1

        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]
        #############################################################################

        final_scalar = self.final_scalar[level]
        scaled = self.ntt.mont_sub([base], [scaler], -1)
        self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
        self.ntt.reduce_2q(scaled, -1)
        self.ntt.make_signed(scaled, -1)
        return scaled

    def decrypt(self, ct: data_struct, sk: data_struct) -> list[torch.Tensor]:
        """
            Decrypt the cipher text ct using the secret key sk.
            Note that the final rescaling must precede the actual decryption process.
        """

        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        if ct.origin == types.origins["ctt"]:
            pt = self.decrypt_triplet(ct_mult=ct, sk=sk)
        elif ct.origin == types.origins["ct"]:
            pt = self.decrypt_double(ct=ct, sk=sk)
        else:
            raise errors.NotMatchType(origin=ct.origin, to=f"{types.origins['ct']} or {types.origins['ctt']}")

        return pt

    # -------------------------------------------------------------------------------------------
    # Key switching.
    # -------------------------------------------------------------------------------------------

    def create_key_switching_key(self, sk_from: data_struct, sk_to: data_struct, a=None) -> data_struct:
        """
            Creates a key to switch the key for sk_src to sk_dst.
        """

        if sk_from.origin != types.origins["sk"] or sk_from.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin="not a secret key", to=types.origins["sk"])
        if (not sk_from.ntt_state) or (not sk_from.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk_from.origin)
        if (not sk_to.ntt_state) or (not sk_to.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk_to.origin)

        level = 0

        stops = self.ntt.stops[-1]
        Psk_src = [sk_from.data[di][:stops[di]].clone() for di in range(self.ntt.num_devices)]

        self.ntt.mont_enter_scalar(Psk_src, self.mont_PR, level)

        ksk = [[] for _ in range(self.ntt.p.num_partitions + 1)]

        for device_id in range(self.ntt.num_devices):
            for part_id, part in enumerate(self.ntt.p.p[level][device_id]):
                global_part_id = self.ntt.p.part_allocations[device_id][part_id]

                crs = a[global_part_id] if a else None
                pk = self.create_public_key(sk_to, include_special=True, a=crs)

                key = tuple(part)
                astart = part[0]
                astop = part[-1] + 1
                shard = Psk_src[device_id][astart:astop]
                pk_data = pk.data[0][device_id][astart:astop]

                _2q = self.ntt.parts_pack[device_id][key]['_2q']
                update_part = ntt_cuda.mont_add([pk_data], [shard], _2q)[0]
                pk_data.copy_(update_part, non_blocking=True)

                pk_name = f'key switch key part index {global_part_id}'
                pk = pk._replace(origin=pk_name)

                ksk[global_part_id] = pk

        return data_struct(
            data=ksk,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ksk"],
            level=level,
            hash=self.hash,
            version=self.version)

    def pre_extend(self, a, device_id, level, part_id, exit_ntt=False):
        text_part = self.ntt.p.parts[level][device_id][part_id]
        param_part = self.ntt.p.p[level][device_id][part_id]

        alpha = len(text_part)
        a_part = a[device_id][text_part[0]:text_part[-1] + 1]

        if exit_ntt:
            self.ntt.intt_exit_reduce([a_part], level, device_id, part_id)

        state = a_part[0].repeat(alpha, 1)
        key = tuple(param_part)
        for i in range(alpha - 1):
            mont_pack = self.ntt.parts_pack[device_id][param_part[i + 1],]['mont_pack']
            _2q = self.ntt.parts_pack[device_id][param_part[i + 1],]['_2q']
            Y_scalar = self.ntt.parts_pack[device_id][key]['Y_scalar'][i][None]

            Y = (a_part[i + 1] - state[i + 1])[None, :]

            ntt_cuda.mont_enter([Y], [Y_scalar], *mont_pack)
            ntt_cuda.reduce_2q([Y], _2q)

            state[i + 1] = Y

            if i + 2 < alpha:
                state_key = tuple(param_part[i + 2:])
                state_mont_pack = self.ntt.parts_pack[device_id][state_key]['mont_pack']
                state_2q = self.ntt.parts_pack[device_id][state_key]['_2q']
                L_scalar = self.ntt.parts_pack[device_id][key]['L_scalar'][i]
                new_state_len = alpha - (i + 2)
                new_state = Y.repeat(new_state_len, 1)
                ntt_cuda.mont_enter([new_state], [L_scalar], *state_mont_pack)
                ntt_cuda.reduce_2q([new_state], state_2q)
                state[i + 2:] += new_state
                ntt_cuda.reduce_2q([state[i + 2:]], state_2q)

        return state

    def extend(self, state, device_id, level, part_id, target_device_id=None):

        if target_device_id is None:
            target_device_id = device_id

        rns_len = len(
            self.ntt.p.destination_arrays_with_special[level][target_device_id])
        alpha = len(state)

        extended = state[0].repeat(rns_len, 1)
        self.ntt.mont_enter([extended], level, target_device_id, -2)

        part = self.ntt.p.p[level][device_id][part_id]
        key = tuple(part)

        L_enter = self.ntt.parts_pack[device_id][key]['L_enter'][target_device_id]

        start = self.ntt.starts[level][target_device_id]

        for i in range(alpha - 1):
            Y = state[i + 1].repeat(rns_len, 1)

            self.ntt.mont_enter_scalar([Y], [L_enter[i][start:]], level, target_device_id, -2)
            extended = self.ntt.mont_add([extended], [Y], level, target_device_id, -2)[0]

        return extended

    def create_switcher(self, a: list[torch.Tensor], ksk: data_struct, level, exit_ntt=False) -> tuple:
        ksk_alloc = self.parts_alloc[level]

        len_devices = self.len_devices[level]
        neighbor_devices = self.neighbor_devices[level]

        num_parts = sum([len(alloc) for alloc in ksk_alloc])
        part_results = [[[[] for _ in range(len_devices)], [[] for _ in range(len_devices)]] for _ in range(num_parts)]

        states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for part_id in range(len(self.ntt.p.p[level][src_device_id])):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                state = self.pre_extend(a,
                                        src_device_id,
                                        level,
                                        part_id,
                                        exit_ntt
                                        )
                states[storage_id] = state

        CPU_states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for part_id, part in enumerate(self.ntt.p.p[level][src_device_id]):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                alpha = len(part)
                CPU_state = self.ksk_buffers[src_device_id][part_id][:alpha]
                CPU_state.copy_(states[storage_id], non_blocking=True)
                CPU_states[storage_id] = CPU_state

        for src_device_id in range(len_devices):
            for part_id in range(len(self.ntt.p.p[level][src_device_id])):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                state = states[storage_id]
                d0, d1 = self.switcher_later_part(state, ksk,
                                                  src_device_id,
                                                  src_device_id,
                                                  level, part_id)

                part_results[storage_id][0][src_device_id] = d0
                part_results[storage_id][1][src_device_id] = d1

        CUDA_states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for j, dst_device_id in enumerate(
                    neighbor_devices[src_device_id]):
                for part_id, part in enumerate(self.ntt.p.p[level][src_device_id]):
                    storage_id = self.stor_ids[level][src_device_id][part_id]
                    CPU_state = CPU_states[storage_id]
                    CUDA_states[storage_id] = CPU_state.cuda(self.ntt.devices[dst_device_id], non_blocking=True)

        torch.cuda.synchronize()

        for src_device_id in range(len_devices):
            for j, dst_device_id in enumerate(
                    neighbor_devices[src_device_id]):
                for part_id, part in enumerate(self.ntt.p.p[level][src_device_id]):
                    storage_id = self.stor_ids[level][src_device_id][part_id]
                    CUDA_state = CUDA_states[storage_id]
                    d0, d1 = self.switcher_later_part(CUDA_state,
                                                      ksk,
                                                      src_device_id,
                                                      dst_device_id,
                                                      level,
                                                      part_id)
                    part_results[storage_id][0][dst_device_id] = d0
                    part_results[storage_id][1][dst_device_id] = d1

        summed0 = part_results[0][0]
        summed1 = part_results[0][1]

        for i in range(1, len(part_results)):
            summed0 = self.ntt.mont_add(summed0, part_results[i][0], level, -2)
            summed1 = self.ntt.mont_add(summed1, part_results[i][1], level, -2)

        d0 = summed0
        d1 = summed1

        current_len = [len(d) for d in self.ntt.p.destination_arrays_with_special[level]]

        for P_ind in range(self.ntt.num_special_primes):
            current_len = [c - 1 for c in current_len]

            PiRi = self.PiRs[level][P_ind]

            P0 = [d[-1].repeat(current_len[di], 1) for di, d in enumerate(d0)]
            P1 = [d[-1].repeat(current_len[di], 1) for di, d in enumerate(d1)]

            d0 = [d0[i][:current_len[i]] - P0[i] for i in range(len_devices)]
            d1 = [d1[i][:current_len[i]] - P1[i] for i in range(len_devices)]

            self.ntt.mont_enter_scalar(d0, PiRi, level, -2)
            self.ntt.mont_enter_scalar(d1, PiRi, level, -2)

        self.ntt.reduce_2q(d0, level, -1)
        self.ntt.reduce_2q(d1, level, -1)

        return d0, d1

    def switcher_later_part(self,
                            state, ksk,
                            src_device_id,
                            dst_device_id,
                            level, part_id):

        extended = self.extend(state, src_device_id, level, part_id, dst_device_id)

        self.ntt.ntt([extended], level, dst_device_id, -2)

        ksk_loc = self.parts_alloc[level][src_device_id][part_id]
        ksk_part_data = ksk.data[ksk_loc].data

        start = self.ntt.starts[level][dst_device_id]
        ksk0_data = ksk_part_data[0][dst_device_id][start:]
        ksk1_data = ksk_part_data[1][dst_device_id][start:]

        d0 = self.ntt.mont_mult([extended], [ksk0_data], level, dst_device_id, -2)
        d1 = self.ntt.mont_mult([extended], [ksk1_data], level, dst_device_id, -2)

        self.ntt.intt_exit_reduce(d0, level, dst_device_id, -2)
        self.ntt.intt_exit_reduce(d1, level, dst_device_id, -2)

        return d0[0], d1[0]

    def switch_key(self, ct: data_struct, ksk: data_struct) -> data_struct:
        include_special = ct.include_special
        ntt_state = ct.ntt_state
        montgomery_state = ct.montgomery_state
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])

        level = ct.level
        a = ct.data[1]
        d0, d1 = self.create_switcher(a, ksk, level, exit_ntt=ct.ntt_state)

        new_ct0 = self.ntt.mont_add(ct.data[0], d0, level, -1)

        return data_struct(
            data=(new_ct0, d1),
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            origin=types.origins["ct"],
            level=level,
            hash=self.hash
        )

    # -------------------------------------------------------------------------------------------
    # Multiplication.
    # -------------------------------------------------------------------------------------------

    def rescale(self, ct: data_struct, exact_rounding=True) -> data_struct:
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        level = ct.level
        next_level = level + 1

        if next_level >= self.num_levels:
            raise errors.MaximumLevelError(level=ct.level, level_max=self.num_levels)

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

        CPU_rescaler0.copy_(rescaler0_at, non_blocking=False)
        CPU_rescaler1.copy_(rescaler1_at, non_blocking=False)

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
            rescale_channel_prime_id = self.ntt.p.destination_arrays[level][rescaler_device_id][0]

            round_at = self.ctx.q[rescale_channel_prime_id] // 2

            rounder0 = [[] for _ in range(len_devices_before)]
            rounder1 = [[] for _ in range(len_devices_before)]

            for device_id in range(len_devices_after):
                rounder0[device_id] = torch.where(rescaler0[device_id] > round_at, 1, 0)
                rounder1[device_id] = torch.where(rescaler1[device_id] > round_at, 1, 0)

        data0 = [(d - s) for d, s in zip(data0, rescaler0)]
        data1 = [(d - s) for d, s in zip(data1, rescaler1)]

        self.ntt.mont_enter_scalar(data0, self.rescale_scales[level], next_level)

        self.ntt.mont_enter_scalar(data1, self.rescale_scales[level], next_level)

        if exact_rounding:
            data0 = [(d + r) for d, r in zip(data0, rounder0)]
            data1 = [(d + r) for d, r in zip(data1, rounder1)]

        self.ntt.reduce_2q(data0, next_level)
        self.ntt.reduce_2q(data1, next_level)

        return data_struct(
            data=(data0, data1),
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level=next_level,
            hash=self.hash,
            version=self.version
        )

    def create_evk(self, sk: data_struct) -> data_struct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        sk2_data = self.ntt.mont_mult(sk.data, sk.data, 0, -2)
        sk2 = data_struct(
            data=sk2_data,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level=sk.level,
            hash=self.hash,
            version=self.version
        )

        return self.create_key_switching_key(sk2, sk)

    def cc_mult(self, a: data_struct, b: data_struct, evk: data_struct, relin=True) -> data_struct:
        if a.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=a.origin, to=types.origins["sk"])
        if b.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=b.origin, to=types.origins["sk"])
        # Rescale.
        x = self.rescale(a)
        y = self.rescale(b)

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

        ct_mult = data_struct(
            data=(d0, d1, d2),
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ctt"],
            level=level,
            hash=self.hash
        )
        if relin:
            ct_mult = self.relinearize(ct_triplet=ct_mult, evk=evk)

        return ct_mult

    def relinearize(self, ct_triplet: data_struct, evk: data_struct) -> data_struct:
        if ct_triplet.origin != types.origins["ctt"]:
            raise errors.NotMatchType(origin=ct_triplet.origin, to=types.origins["ctt"])
        if not ct_triplet.ntt_state or not ct_triplet.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct_triplet.origin)

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
        return data_struct(
            data=(d0, d1),
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level=level,
            hash=self.hash
        )

    # -------------------------------------------------------------------------------------------
    # Rotation.
    # -------------------------------------------------------------------------------------------

    def create_rotation_key(self, sk: data_struct, delta: int, a: list[torch.Tensor] = None) -> data_struct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data)
        sk_new_data = [rotate(s, delta) for s in sk_new_data]
        self.ntt.ntt(sk_new_data)
        sk_rotated = data_struct(
            data=sk_new_data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level=0,
            hash=self.hash,
            version=self.version
        )

        rotk = self.create_key_switching_key(sk_rotated, sk, a=a)
        rotk = rotk._replace(origin=types.origins["rotk"] + f"{delta}")
        return rotk

    def rotate_single(self, ct: data_struct, rotk: data_struct) -> data_struct:

        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if types.origins["rotk"] not in rotk.origin:
            raise errors.NotMatchType(origin=rotk.origin, to=types.origins["rotk"])

        level = ct.level
        include_special = ct.include_special
        ntt_state = ct.ntt_state
        montgomery_state = ct.montgomery_state
        origin = rotk.origin
        delta = int(origin.split(':')[-1])

        rotated_ct_data = [[rotate(d, delta) for d in ct_data] for ct_data in ct.data]

        rotated_ct_rotated_sk = data_struct(
            data=rotated_ct_data,
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            origin=types.origins["ct"],
            level=level,
            hash=self.hash,
            version=self.version
        )

        rotated_ct = self.switch_key(rotated_ct_rotated_sk, rotk)
        return rotated_ct

    def create_galois_key(self, sk: data_struct) -> data_struct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        galois_key_parts = [self.create_rotation_key(sk, delta) for delta in self.galois_deltas]

        galois_key = data_struct(
            data=galois_key_parts,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["galk"],
            level=0,
            hash=self.hash,
            version=self.version
        )
        return galois_key

    def rotate_galois(self, ct: data_struct, gk: data_struct, delta: int, return_circuit=False) -> data_struct:

        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if gk.origin != types.origins["galk"]:
            raise errors.NotMatchType(origin=gk.origin, to=types.origins["galk"])

        current_delta = delta % (self.ctx.N // 2)
        galois_circuit = []

        while current_delta:
            galois_ind = int(math.log2(current_delta))
            galois_delta = self.galois_deltas[galois_ind]
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
    def cc_add_double(self, a: data_struct, b: data_struct) -> data_struct:
        if a.origin != types.origins["ct"] or b.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=f"{a.origin} and {b.origin}", to=types.origins["ct"])
        if a.ntt_state or a.montgomery_state:
            raise errors.NotMatchDataStructState(origin=a.origin)
        if b.ntt_state or b.montgomery_state:
            raise errors.NotMatchDataStructState(origin=b.origin)

        level = a.level
        data = []
        c0 = self.ntt.mont_add(a.data[0], b.data[0], level)
        c1 = self.ntt.mont_add(a.data[1], b.data[1], level)
        self.ntt.reduce_2q(c0, level)
        self.ntt.reduce_2q(c1, level)
        data.extend([c0, c1])

        return data_struct(
            data=data,
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level=level,
            hash=self.hash
        )

    def cc_add_triplet(self, a: data_struct, b: data_struct) -> data_struct:
        if a.origin != types.origins["ctt"] or b.origin != types.origins["ctt"]:
            raise errors.NotMatchType(origin=f"{a.origin} and {b.origin}", to=types.origins["ctt"])
        if not a.ntt_state or not a.montgomery_state:
            raise errors.NotMatchDataStructState(origin=a.origin)
        if not b.ntt_state or not b.montgomery_state:
            raise errors.NotMatchDataStructState(origin=b.origin)

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

        return data_struct(
            data=data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ctt"],
            level=level,
            hash=self.hash,
            version=self.version
        )

    def cc_add(self, a: data_struct, b: data_struct) -> data_struct:

        if a.origin == types.origins["ct"] and b.origin == types.origins["ct"]:
            ct_add = self.cc_add_double(a, b)
        elif a.origin == types.origins["ctt"] and b.origin == types.origins["ctt"]:
            ct_add = self.cc_add_triplet(a, b)
        else:
            raise errors.DifferentTypeError(a=a.origin, b=b.origin)

        return ct_add

    def cc_sub_double(self, a: data_struct, b: data_struct) -> data_struct:
        if a.origin != types.origins["ct"] or b.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=f"{a.origin} and {b.origin}", to=types.origins["ct"])
        if a.ntt_state or a.montgomery_state:
            raise errors.NotMatchDataStructState(origin=a.origin)
        if b.ntt_state or b.montgomery_state:
            raise errors.NotMatchDataStructState(origin=b.origin)

        level = a.level
        data = []

        c0 = self.ntt.mont_sub(a.data[0], b.data[0], level)
        c1 = self.ntt.mont_sub(a.data[1], b.data[1], level)
        self.ntt.reduce_2q(c0, level)
        self.ntt.reduce_2q(c1, level)
        data.extend([c0, c1])

        return data_struct(
            data=data,
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level=level,
            hash=self.hash,
            version=self.version
        )

    def cc_sub_triplet(self, a: data_struct, b: data_struct) -> data_struct:
        if a.origin != types.origins["ctt"] or b.origin != types.origins["ctt"]:
            raise errors.NotMatchType(origin=f"{a.origin} and {b.origin}", to=types.origins["ctt"])
        if not a.ntt_state or not a.montgomery_state:
            raise errors.NotMatchDataStructState(origin=a.origin)
        if not b.ntt_state or not b.montgomery_state:
            raise errors.NotMatchDataStructState(origin=b.origin)

        level = a.level
        data = []
        c0 = self.ntt.mont_sub(a.data[0], b.data[0], level)
        c1 = self.ntt.mont_sub(a.data[1], b.data[1], level)
        c2 = self.ntt.mont_sub(a.data[2], b.data[2], level)
        self.ntt.reduce_2q(c0, level)
        self.ntt.reduce_2q(c1, level)
        self.ntt.reduce_2q(c2, level)
        data.extend([c0, c1, c2])

        return data_struct(
            data=data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ctt"],
            level=level,
            hash=self.hash,
            version=self.version
        )

    def cc_sub(self, a: data_struct, b: data_struct) -> data_struct:
        if a.origin != b.origin:
            raise Exception(f"[Error] triplet error")

        if types.origins["ct"] == a.origin and types.origins["ct"] == b.origin:
            ct_sub = self.cc_sub_double(a, b)
        elif a.origin == types.origins["ctt"] and b.origin == types.origins["ctt"]:
            ct_sub = self.cc_sub_triplet(a, b)
        else:
            raise errors.DifferentTypeError(a=a.origin, b=b.origin)
        return ct_sub

    def cc_subtract(self, a, b):
        return self.cc_sub(a, b)

    # -------------------------------------------------------------------------------------------
    # Level up.
    # -------------------------------------------------------------------------------------------
    def level_up(self, ct: data_struct, dst_level: int):
        if types.origins["ct"] != ct.origin:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])

        current_level = ct.level

        new_ct = self.rescale(ct)

        src_level = current_level + 1

        dst_len_devices = len(self.ntt.p.destination_arrays[dst_level])

        diff_deviation = self.deviations[dst_level] / np.sqrt(self.deviations[src_level])

        deviated_delta = round(self.scale * diff_deviation)

        if dst_level - src_level > 0:
            src_rns_lens = [len(d) for d in self.ntt.p.destination_arrays[src_level]]
            dst_rns_lens = [len(d) for d in self.ntt.p.destination_arrays[dst_level]]

            diff_rns_lens = [y - x for x, y in zip(dst_rns_lens, src_rns_lens)]

            new_ct_data0 = []
            new_ct_data1 = []

            for device_id in range(dst_len_devices):
                new_ct_data0.append(new_ct.data[0][device_id][diff_rns_lens[device_id]:])
                new_ct_data1.append(new_ct.data[1][device_id][diff_rns_lens[device_id]:])
        else:
            new_ct_data0, new_ct_data1 = new_ct.data

        multipliers = []
        for device_id in range(dst_len_devices):
            dest = self.ntt.p.destination_arrays[dst_level][device_id]
            q = [self.ctx.q[i] for i in dest]

            multiplier = [(deviated_delta * self.ctx.R) % qi for qi in q]
            multiplier = torch.tensor(multiplier, dtype=self.ctx.torch_dtype, device=self.ntt.devices[device_id])
            multipliers.append(multiplier)

        self.ntt.mont_enter_scalar(new_ct_data0, multipliers, dst_level)
        self.ntt.mont_enter_scalar(new_ct_data1, multipliers, dst_level)

        self.ntt.reduce_2q(new_ct_data0, dst_level)
        self.ntt.reduce_2q(new_ct_data1, dst_level)

        new_ct = data_struct(
            data=(new_ct_data0, new_ct_data1),
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level=dst_level,
            hash=self.hash,
            version=self.version
        )

        return new_ct

    # -------------------------------------------------------------------------------------------
    # Fused enc/dec.
    # -------------------------------------------------------------------------------------------
    def encodecrypt(self, m, pk: data_struct, level: int = 0, padding=True) -> data_struct:
        if pk.origin != types.origins["pk"]:
            raise errors.NotMatchType(origin=pk.origin, to=types.origins["pk"])

        if padding:
            m = self.padding(m=m)

        deviation = self.deviations[level]
        pt = encode(m, scale=self.scale,
                    device=self.device0, norm=self.norm,
                    deviation=deviation, rng=self.rng,
                    return_without_scaling=self.bias_guard)

        if self.bias_guard:
            dc_integral = pt[0].item() // 1
            pt[0] -= dc_integral

            dc_scale = int(dc_integral) * int(self.scale)
            dc_rns = []
            for device_id, dest in enumerate(self.ntt.p.destination_arrays[level]):
                dci = [dc_scale % self.ctx.q[i] for i in dest]
                dci = torch.tensor(dci,
                                   dtype=self.ctx.torch_dtype,
                                   device=self.ntt.devices[device_id])
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
        pk0 = [pk.data[0][di][start[di]:] for di in range(self.ntt.num_devices)]
        pk1 = [pk.data[1][di][start[di]:] for di in range(self.ntt.num_devices)]

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

        ct = data_struct(
            data=(ct0, ct1),
            include_special=mult_type == -2,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level=level,
            hash=self.hash,
            version=self.version
        )

        return ct

    def decryptcode(self, ct: data_struct, sk: data_struct, is_real=False) -> data_struct:
        if (not sk.ntt_state) or (not sk.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk.origin)

        level = ct.level
        sk_data = sk.data[0][self.ntt.starts[level][0]:]

        if ct.origin == types.origins["ct"]:
            if ct.ntt_state or ct.montgomery_state:
                raise errors.NotMatchDataStructState(origin=ct.origin)

            ct0 = ct.data[0][0]
            a = ct.data[1][0].clone()

            self.ntt.enter_ntt([a], level)

            sa = self.ntt.mont_mult([a], [sk_data], level)
            self.ntt.intt_exit(sa, level)

            pt = self.ntt.mont_add([ct0], sa, level)
            self.ntt.reduce_2q(pt, level)

        elif ct.origin == types.origins["ctt"]:
            if not ct.ntt_state or not ct.montgomery_state:
                raise errors.NotMatchDataStructState(origin=ct.origin)

            d0 = [ct.data[0][0].clone()]
            d1 = [ct.data[1][0]]
            d2 = [ct.data[2][0]]

            self.ntt.intt_exit_reduce(d0, level)

            sk_data = [sk.data[0][self.ntt.starts[level][0]:]]

            d1_s = self.ntt.mont_mult(d1, sk_data, level)

            s2 = self.ntt.mont_mult(sk_data, sk_data, level)
            d2_s2 = self.ntt.mont_mult(d2, s2, level)

            self.ntt.intt_exit(d1_s, level)
            self.ntt.intt_exit(d2_s2, level)

            pt = self.ntt.mont_add(d0, d1_s, level)
            pt = self.ntt.mont_add(pt, d2_s2, level)
            self.ntt.reduce_2q(pt, level)
        else:
            raise errors.NotMatchType(origin=ct.origin, to=f"{types.origins['ct']} or {types.origins['ctt']}")

        base_at = -self.ctx.num_special_primes - 1 if ct.include_special else -1
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

        # Decoding.
        correction = self.corrections[level]
        decoded = decode(
            scaled[0][-1],
            scale=self.scale,
            correction=correction,
            norm=self.norm,
            return_without_scaling=self.bias_guard
        )
        decoded = decoded[:self.ctx.N // 2].cpu().numpy()
        ##

        decoded = decoded / self.scale * correction

        # Bias guard.
        if (len_left >= 3) and self.bias_guard:
            decoded += dc / self.scale * correction
        if is_real:
            decoded = decoded.real
        return decoded

    # Shortcuts.
    def encorypt(self, m, pk: data_struct, level: int = 0, padding=True):
        return self.encodecrypt(m, pk=pk, level=level, padding=padding)

    def decrode(self, ct: data_struct, sk: data_struct, is_real=False):
        return self.decryptcode(ct=ct, sk=sk, is_real=is_real)

    # -------------------------------------------------------------------------------------------
    # Conjugation
    # -------------------------------------------------------------------------------------------

    def create_conjugation_key(self, sk: data_struct) -> data_struct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if (not sk.ntt_state) or (not sk.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk.origin)

        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data)
        sk_new_data = [conjugate(s) for s in sk_new_data]
        self.ntt.ntt(sk_new_data)
        sk_rotated = data_struct(
            data=sk_new_data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level=0,
            hash=self.hash,
            version=self.version
        )
        rotk = self.create_key_switching_key(sk_rotated, sk)
        rotk = rotk._replace(origin=types.origins["conjk"])
        return rotk

    def conjugate(self, ct: data_struct, conjk: data_struct):
        level = ct.level
        conj_ct_data = [[conjugate(d) for d in ct_data] for ct_data in ct.data]

        conj_ct_sk = data_struct(
            data=conj_ct_data,
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level=level,
            hash=self.hash,
            version=self.version
        )

        conj_ct = self.switch_key(conj_ct_sk, conjk)
        return conj_ct

        # -------------------------------------------------------------------------------------------
        # Clone.
        # -------------------------------------------------------------------------------------------

    def clone_tensors(self, data: data_struct) -> data_struct:
        new_data = []
        # Some data has 1 depth.
        if not isinstance(data[0], list):
            for device_data in data:
                new_data.append(device_data.clone())
        else:
            for part in data:
                new_data.append([])
                for device_data in part:
                    new_data[-1].append(device_data.clone())
        return new_data

    def clone(self, text):
        if not isinstance(text.data[0], data_struct):
            # data are tensors.
            data = self.clone_tensors(text.data)

            wrapper = data_struct(
                data=data,
                include_special=text.include_special,
                ntt_state=text.ntt_state,
                montgomery_state=text.montgomery_state,
                origin=text.origin,
                level=text.level,
                hash=text.hash,
                version=text.version
            )

        else:
            wrapper = data_struct(
                data=[],
                include_special=text.include_special,
                ntt_state=text.ntt_state,
                montgomery_state=text.montgomery_state,
                origin=text.origin,
                level=text.level,
                hash=text.hash,
                version=text.version
            )

            for d in text.data:
                wrapper.data.append(self.clone(d))

        return wrapper

        # -------------------------------------------------------------------------------------------
        # Move data back and forth from GPUs to the CPU.
        # -------------------------------------------------------------------------------------------

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
        cpu_tensor = torch.empty(cpu_size, dtype=self.ctx.torch_dtype, device='cpu')

        for ten, dest_i in zip(gpu_data, dest):
            # Check if the tensor is in the gpu.
            if ten.device.type != 'cuda':
                raise Exception("To download data to the CPU, it must already be in a GPU!!!")

            # Copy in the data.
            cpu_tensor[dest_i] = ten.cpu()

        # To avoid confusion, make a list with a single element (only one device, that is the CPU),
        # and return it.
        return [cpu_tensor]

    def upload_to_gpu(self, cpu_data, level, include_special):
        # There's only one device data in the cpu data.
        cpu_tensor = cpu_data[0]

        # Check if the tensor is in the cpu.
        if cpu_tensor.device.type != 'cpu':
            raise Exception("To upload data to GPUs, it must already be in the CPU!!!")

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
            'gpu2cpu': self.download_to_cpu,
            'cpu2gpu': self.upload_to_gpu
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

    def move_to(self, text, direction='gpu2cpu'):
        if not isinstance(text.data[0], data_struct):
            level = text.level
            include_special = text.include_special

            # data are tensors.
            data = self.move_tensors(text.data, level,
                                     include_special, direction)

            wrapper = data_struct(
                data, text.include_special,
                text.ntt_state, text.montgomery_state,
                text.origin, text.level, text.hash, text.version
            )

        else:
            wrapper = data_struct(
                [], text.include_special,
                text.ntt_state, text.montgomery_state,
                text.origin, text.level, text.hash, text.version
            )

            for d in text.data:
                moved = self.move_to(d, direction)
                wrapper.data.append(moved)

        return wrapper

        # Shortcuts

    def cpu(self, ct):
        return self.move_to(ct, 'gpu2cpu')

    def cuda(self, ct):
        return self.move_to(ct, 'cpu2gpu')

        # -------------------------------------------------------------------------------------------
        # check device.
        # -------------------------------------------------------------------------------------------

    def tensor_device(self, data):
        # Some data has 1 depth.
        if not isinstance(data[0], list):
            return data[0].device.type
        else:
            return data[0][0].device.type

    def device(self, text):
        if not isinstance(text.data[0], data_struct):
            # data are tensors.
            return self.tensor_device(text.data)
        else:
            return self.device(text.data[0])

        # -------------------------------------------------------------------------------------------
        # Print data structure.
        # -------------------------------------------------------------------------------------------

    def tree_lead_text(self, level, tabs=2, final=False):
        final_char = "└" if final else "├"

        if level == 0:
            leader = " " * tabs
            trailer = "─" * tabs
            lead_text = "─" * tabs + "┬" + trailer

        elif level < 0:
            level = -level
            leader = " " * tabs
            trailer = "─" + "─" * (tabs - 1)
            lead_fence = leader + "│" * (level - 1)
            lead_text = lead_fence + final_char + trailer

        else:
            leader = " " * tabs
            trailer = "┬" + "─" * (tabs - 1)
            lead_fence = leader + "│" * (level - 1)
            lead_text = lead_fence + "├" + trailer

        return lead_text

    def print_data_shapes(self, data, level):
        # Some data structures have 1 depth.
        if isinstance(data[0], list):
            for part_i, part in enumerate(data):
                for device_id, d in enumerate(part):
                    device = self.ntt.devices[device_id]

                    if (device_id == len(part) - 1) and \
                            (part_i == len(data) - 1):
                        final = True
                    else:
                        final = False

                    lead_text = self.tree_lead_text(-level, final=final)

                    print(f"{lead_text} tensor at device {device} with "
                          f"shape {d.shape}.")
        else:
            for device_id, d in enumerate(data):
                device = self.ntt.devices[device_id]

                if device_id == len(data) - 1:
                    final = True
                else:
                    final = False

                lead_text = self.tree_lead_text(-level, final=final)

                print(f"{lead_text} tensor at device {device} with "
                      f"shape {d.shape}.")

    def print_data_structure(self, text, level=0):
        lead_text = self.tree_lead_text(level)
        print(f"{lead_text} {text.origin}")

        if not isinstance(text.data[0], data_struct):
            self.print_data_shapes(text.data, level + 1)
        else:
            for d in text.data:
                self.print_data_structure(d, level + 1)

    # -------------------------------------------------------------------------------------------
    # Save and load.
    # -------------------------------------------------------------------------------------------

    def auto_generate_filename(self, fmt_str='%Y%m%d%H%M%s%f'):
        return datetime.datetime.now().strftime(fmt_str) + '.pkl'

    def save(self, text, filename=None):
        if filename is None:
            filename = self.auto_generate_filename()

        savepath = Path(filename)

        # Check if the text is in the CPU.
        # If not, move to CPU.
        if self.device(text) != 'cpu':
            cpu_text = self.cpu(text)
        else:
            cpu_text = text

        with savepath.open('wb') as f:
            pickle.dump(cpu_text, f)

    def load(self, filename, move_to_gpu=True):
        savepath = Path(filename)
        with savepath.open('rb') as f:
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

    def negate(self, ct: data_struct) -> data_struct:
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])  # ctt
        new_ct = self.clone(ct)

        new_data = new_ct.data
        for part in new_data:
            for d in part:
                d *= -1
            self.ntt.make_signed(part, ct.level)

        return new_ct

        # -------------------------------------------------------------------------------------------
        # scalar ops.
        # -------------------------------------------------------------------------------------------

    def mult_int_scalar(self, ct: data_struct, scalar, evk=None, relin=True):
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])

        device_len = len(ct.data[0])

        int_scalar = int(scalar)
        mont_scalar = [(int_scalar * self.ctx.R) % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[ct.level]

        partitioned_mont_scalar = [[mont_scalar[i] for i in desti] for desti in dest]
        tensorized_scalar = []
        for device_id in range(device_len):
            scal_tensor = torch.tensor(
                partitioned_mont_scalar[device_id],
                dtype=self.ctx.torch_dtype,
                device=self.ntt.devices[device_id]
            )
            tensorized_scalar.append(scal_tensor)

        new_ct = self.clone(ct)
        new_data = new_ct.data
        for i in [0, 1]:
            self.ntt.mont_enter_scalar(new_data[i], tensorized_scalar, ct.level)
            self.ntt.reduce_2q(new_data[i], ct.level)

        return new_ct

    def mult_scalar(self, ct, scalar, evk=None, relin=True):
        device_len = len(ct.data[0])

        scaled_scalar = int(
            scalar * self.scale * np.sqrt(self.deviations[ct.level + 1]) + 0.5)

        mont_scalar = [(scaled_scalar * self.ctx.R) % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[ct.level]

        partitioned_mont_scalar = [[mont_scalar[i] for i in dest_i] for dest_i in dest]
        tensorized_scalar = []
        for device_id in range(device_len):
            scal_tensor = torch.tensor(
                partitioned_mont_scalar[device_id],
                dtype=self.ctx.torch_dtype,
                device=self.ntt.devices[device_id]
            )
            tensorized_scalar.append(scal_tensor)

        new_ct = self.clone(ct)
        new_data = new_ct.data

        for i in [0, 1]:
            self.ntt.mont_enter_scalar(new_data[i], tensorized_scalar, ct.level)
            self.ntt.reduce_2q(new_data[i], ct.level)

        return self.rescale(new_ct)

    def add_scalar(self, ct, scalar):
        device_len = len(ct.data[0])

        scaled_scalar = int(scalar * self.scale * self.deviations[ct.level] + 0.5)

        if self.norm == 'backward':
            scaled_scalar *= self.ctx.N

        scaled_scalar *= self.int_scale

        scaled_scalar = [scaled_scalar % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[ct.level]

        partitioned_mont_scalar = [[scaled_scalar[i] for i in desti] for desti in dest]
        tensorized_scalar = []
        for device_id in range(device_len):
            scal_tensor = torch.tensor(
                partitioned_mont_scalar[device_id],
                dtype=self.ctx.torch_dtype,
                device=self.ntt.devices[device_id]
            )
            tensorized_scalar.append(scal_tensor)

        new_ct = self.clone(ct)
        new_data = new_ct.data

        dc = [d[:, 0] for d in new_data[0]]
        for device_id in range(device_len):
            dc[device_id] += tensorized_scalar[device_id]

        self.ntt.reduce_2q(new_data[0], ct.level)

        return new_ct

    def sub_scalar(self, ct, scalar):
        return self.add_scalar(ct, -scalar)

    def int_scalar_mult(self, scalar, ct, evk=None, relin=True):
        return self.mult_int_scalar(ct, scalar)

    def scalar_mult(self, scalar, ct, evk=None, relin=True):
        return self.mult_scalar(ct, scalar)

    def scalar_add(self, scalar, ct):
        return self.add_scalar(ct, scalar)

    def scalar_sub(self, scalar, ct):
        neg_ct = self.negate(ct)
        return self.add_scalar(neg_ct, scalar)

        # -------------------------------------------------------------------------------------------
        # message ops.
        # -------------------------------------------------------------------------------------------

    def mc_mult(self, m, ct, evk=None, relin=True):
        m = np.array(m) * np.sqrt(self.deviations[ct.level + 1])

        pt = self.encode(m, 0)

        pt_tiled = self.ntt.tile_unsigned(pt, ct.level)

        # Transform ntt to prepare for multiplication.
        self.ntt.enter_ntt(pt_tiled, ct.level)

        # Prepare a new ct.
        new_ct = self.clone(ct)

        self.ntt.enter_ntt(new_ct.data[0], ct.level)
        self.ntt.enter_ntt(new_ct.data[1], ct.level)

        new_d0 = self.ntt.mont_mult(pt_tiled, new_ct.data[0], ct.level)
        new_d1 = self.ntt.mont_mult(pt_tiled, new_ct.data[1], ct.level)

        self.ntt.intt_exit_reduce(new_d0, ct.level)
        self.ntt.intt_exit_reduce(new_d1, ct.level)

        new_ct.data[0] = new_d0
        new_ct.data[1] = new_d1

        return self.rescale(new_ct)

    def mc_add(self, m, ct):
        pt = self.encode(m, ct.level)
        pt_tiled = self.ntt.tile_unsigned(pt, ct.level)

        self.ntt.mont_enter_scale(pt_tiled, ct.level)

        new_ct = self.clone(ct)
        self.ntt.mont_enter(new_ct.data[0], ct.level)
        new_d0 = self.ntt.mont_add(pt_tiled, new_ct.data[0], ct.level)
        self.ntt.mont_redc(new_d0, ct.level)
        self.ntt.reduce_2q(new_d0, ct.level)

        new_ct.data[0] = new_d0

        return new_ct

    def mc_sub(self, m, ct):
        neg_ct = self.negate(ct)
        return self.mc_add(m, neg_ct)

    def cm_mult(self, ct, m, evk=None, relin=True):
        return self.mc_mult(m, ct)

    def cm_add(self, ct, m):
        return self.mc_add(m, ct)

    def cm_sub(self, ct, m):
        return self.mc_add(-np.array(m), ct)

        # -------------------------------------------------------------------------------------------
        # Automatic cc ops.
        # -------------------------------------------------------------------------------------------

    def auto_level(self, ct0, ct1):
        level_diff = ct0.level - ct1.level
        if level_diff < 0:
            new_ct0 = self.level_up(ct0, ct1.level)
            return new_ct0, ct1
        elif level_diff > 0:
            new_ct1 = self.level_up(ct1, ct0.level)
            return ct0, new_ct1
        else:
            return ct0, ct1

    def auto_cc_mult(self, ct0, ct1, evk, relin=True):
        lct0, lct1 = self.auto_level(ct0, ct1)
        return self.cc_mult(lct0, lct1, evk, relin=relin)

    def auto_cc_add(self, ct0, ct1):
        lct0, lct1 = self.auto_level(ct0, ct1)
        return self.cc_add(lct0, lct1)

    def auto_cc_sub(self, ct0, ct1):
        lct0, lct1 = self.auto_level(ct0, ct1)
        return self.cc_sub(lct0, lct1)

        # -------------------------------------------------------------------------------------------
        # Fully automatic ops.
        # -------------------------------------------------------------------------------------------

    def mult(self, a, b, evk=None, relin=True):
        type_a = type(a)
        type_b = type(b)

        try:
            func = self.mult_dispatch_dict[type_a, type_b]
        except Exception as e:
            raise Exception(f"Unsupported data types are input.\n{e}")

        return func(a, b, evk, relin)

    def add(self, a, b):
        type_a = type(a)
        type_b = type(b)

        try:
            func = self.add_dispatch_dict[type_a, type_b]
        except Exception as e:
            raise Exception(f"Unsupported data types are input.\n{e}")

        return func(a, b)

    def sub(self, a, b):
        type_a = type(a)
        type_b = type(b)

        try:
            func = self.sub_dispatch_dict[type_a, type_b]
        except Exception as e:
            raise Exception(f"Unsupported data types are input.\n{e}")

        return func(a, b)

        # -------------------------------------------------------------------------------------------
        # Misc.
        # -------------------------------------------------------------------------------------------

    def refresh(self):
        # Refreshes the rng state.
        self.rng.refresh()

    def reduce_error(self, ct):
        # Reduce the accumulated error in the cipher text.
        return self.mult_scalar(ct, 1.0)

        # -------------------------------------------------------------------------------------------
        # Misc ops.
        # -------------------------------------------------------------------------------------------

    def sum(self, ct: data_struct, gk: data_struct, rescale_every=5) -> data_struct:
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if gk.origin != types.origins["galk"]:
            raise errors.NotMatchType(origin=gk.origin, to=types.origins["galk"])

        new_ct = self.clone(ct)
        for roti in range(self.ctx.logN - 1):
            rot_ct = self.rotate_single(new_ct, gk.data[roti])
            sum_ct = self.add(rot_ct, new_ct)
            del new_ct, rot_ct
            if roti != 0 and (roti % rescale_every) == 0:
                new_ct = self.reduce_error(sum_ct)
            else:
                new_ct = sum_ct
        return new_ct

    def mean(self, ct: data_struct, gk: data_struct, alpha=1, rescale_every=5) -> data_struct:
        # Divide by num_slots.
        # The cipher text is refreshed here, and hence
        # doesn't need to be refreshed at roti=0 in the loop.
        new_ct = self.mult(1 / self.num_slots / alpha, ct)

        for roti in range(self.ctx.logN - 1):
            rotk = gk.data[roti]
            rot_ct = self.rotate_single(new_ct, rotk)
            sum_ct = self.add(rot_ct, new_ct)
            del new_ct, rot_ct
            if ((roti % rescale_every) == 0) and (roti != 0):
                new_ct = self.reduce_error(sum_ct)
            else:
                new_ct = sum_ct
        return new_ct

    def cov(self, ct_a: data_struct, ct_b: data_struct,
            evk: data_struct, gk: data_struct, rescale_every=5) -> data_struct:
        cta_mean = self.mean(ct_a, gk, rescale_every=rescale_every)
        ctb_mean = self.mean(ct_b, gk, rescale_every=rescale_every)

        cta_dev = self.sub(ct_a, cta_mean)
        ctb_dev = self.sub(ct_b, ctb_mean)

        ct_cov = self.mult(self.mult(cta_dev, ctb_dev, evk), 1 / (self.num_slots - 1))

        return ct_cov

    def pow(self, ct: data_struct, power: int, evk: data_struct) -> data_struct:
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
            new_ct = self.auto_cc_mult(new_ct, pow_term, evk)
            remaining_exponent -= 2 ** pow_ind

        return new_ct

    def square(self, ct: data_struct, evk: data_struct, relin=True) -> data_struct:
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

        ct_mult = data_struct(
            data=(d0, d1, d2),
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ctt"],
            level=level,
            hash=self.hash,
            version=self.version
        )
        if relin:
            ct_mult = self.relinearize(ct_triplet=ct_mult, evk=evk)

        return ct_mult

    # -------------------------------------------------------------------------------------------
    # Multiparty.
    # -------------------------------------------------------------------------------------------
    def multiparty_public_crs(self, pk: data_struct):
        crs = self.clone(pk).data[1]
        return crs

    def multiparty_create_public_key(self, sk: data_struct, a=None, include_special=False) -> data_struct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if include_special and not sk.include_special:
            raise errors.SecretKeyNotIncludeSpecialPrime()
        mult_type = -2 if include_special else -1

        level = 0
        e = self.rng.discrete_gaussian(repeats=1)
        e = self.ntt.tile_unsigned(e, level, mult_type)

        self.ntt.enter_ntt(e, level, mult_type)
        repeats = self.ctx.num_special_primes if sk.include_special else 0

        if a is None:
            a = self.rng.randint(
                self.ntt.q_prepack[mult_type][level][0],
                repeats=repeats
            )

        sa = self.ntt.mont_mult(a, sk.data, 0, mult_type)
        pk0 = self.ntt.mont_sub(e, sa, 0, mult_type)
        pk = data_struct(
            data=(pk0, a),
            include_special=include_special,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["pk"],
            level=level,
            hash=self.hash,
            version=self.version
        )
        return pk

    def multiparty_create_collective_public_key(self, pks: list[data_struct]) -> data_struct:
        data, include_special, ntt_state, montgomery_state, origin, level, hash_, version = pks[0]
        mult_type = -2 if include_special else -1
        b = [b.clone() for b in data[0]]  # num of gpus
        a = [a.clone() for a in data[1]]

        for pk in pks[1:]:
            b = self.ntt.mont_add(b, pk.data[0], lvl=0, mult_type=mult_type)

        cpk = data_struct(
            (b, a),
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            origin=types.origins["pk"],
            level=level,
            hash=self.hash,
            version=self.version
        )
        return cpk

    def multiparty_decrypt_head(self, ct: data_struct, sk: data_struct):
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if ct.ntt_state or ct.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct.origin)
        if not sk.ntt_state or not sk.montgomery_state:
            raise errors.NotMatchDataStructState(origin=sk.origin)
        level = ct.level

        ct0 = ct.data[0][0]
        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)

        sk_data = sk.data[0][self.ntt.starts[level][0]:]

        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        pt = self.ntt.mont_add([ct0], sa, level)

        return pt

    def multiparty_decrypt_partial(self, ct: data_struct, sk: data_struct) -> data_struct:
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if ct.ntt_state or ct.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct.origin)
        if not sk.ntt_state or not sk.montgomery_state:
            raise errors.NotMatchDataStructState(origin=sk.origin)

        data, include_special, ntt_state, montgomery_state, origin, level, hash_, version = ct

        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)

        sk_data = sk.data[0][self.ntt.starts[level][0]:]

        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        return sa

    def multiparty_decrypt_fusion(self, pcts: list, level=0, include_special=False):
        pt = [x.clone() for x in pcts[0]]
        for pct in pcts[1:]:
            pt = self.ntt.mont_add(pt, pct, level)

        self.ntt.reduce_2q(pt, level)

        base_at = -self.ctx.num_special_primes - 1 if include_special else -1

        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        final_scalar = self.final_scalar[level]
        scaled = self.ntt.mont_sub([base], [scaler], -1)
        self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
        self.ntt.reduce_2q(scaled, -1)
        self.ntt.make_signed(scaled, -1)

        m = self.decode(m=scaled, level=level)

        return m

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. ROTATION
    #### -------------------------------------------------------------------------------------------

    def multiparty_create_key_switching_key(self, sk_src: data_struct, sk_dst: data_struct, a=None) -> data_struct:
        if sk_src.origin != types.origins["sk"] or sk_src.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin="not a secret key", to=types.origins["sk"])
        if (not sk_src.ntt_state) or (not sk_src.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk_src.origin)
        if (not sk_dst.ntt_state) or (not sk_dst.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk_dst.origin)

        level = 0

        stops = self.ntt.stops[-1]
        Psk_src = [sk_src.data[di][:stops[di]].clone() for di in range(self.ntt.num_devices)]

        self.ntt.mont_enter_scalar(Psk_src, self.mont_PR, level)

        ksk = [[] for _ in range(self.ntt.p.num_partitions + 1)]
        for device_id in range(self.ntt.num_devices):
            for part_id, part in enumerate(self.ntt.p.p[level][device_id]):
                global_part_id = self.ntt.p.part_allocations[device_id][part_id]

                crs = a[global_part_id] if a else None
                pk = self.multiparty_create_public_key(sk_dst, include_special=True, a=crs)
                key = tuple(part)
                astart = part[0]
                astop = part[-1] + 1
                shard = Psk_src[device_id][astart:astop]
                pk_data = pk.data[0][device_id][astart:astop]

                _2q = self.ntt.parts_pack[device_id][key]['_2q']
                update_part = ntt_cuda.mont_add([pk_data], [shard], _2q)[0]
                pk_data.copy_(update_part, non_blocking=True)

                # Name the pk.
                pk_name = f'key switch key part index {global_part_id}'
                pk = pk._replace(origin=pk_name)

                ksk[global_part_id] = pk

        return data_struct(
            data=ksk,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ksk"],
            level=level,
            hash=self.hash,
            version=self.version
        )

    def multiparty_create_rotation_key(self, sk: data_struct, delta: int, a=None) -> data_struct:
        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data)
        sk_new_data = [rotate(s, delta) for s in sk_new_data]
        self.ntt.ntt(sk_new_data)
        sk_rotated = data_struct(
            data=sk_new_data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level=0,
            hash=self.hash,
            version=self.version
        )
        rotk = self.multiparty_create_key_switching_key(sk_rotated, sk, a=a)
        rotk = rotk._replace(origin=types.origins["rotk"] + f"{delta}")
        return rotk

    def multiparty_generate_rotation_key(self, rotks: list[data_struct]) -> data_struct:
        crotk = self.clone(rotks[0])
        for rotk in rotks[1:]:
            for ksk_idx in range(len(rotk.data)):
                update_parts = self.ntt.mont_add(crotk.data[ksk_idx].data[0], rotk.data[ksk_idx].data[0])
                crotk.data[ksk_idx].data[0][0].copy_(update_parts[0], non_blocking=True)
        return crotk

    def generate_rotation_crs(self, rotk: data_struct):
        if types.origins["rotk"] not in rotk.origin and types.origins["ksk"] != rotk.origin:
            raise errors.NotMatchType(origin=rotk.origin, to=types.origins["ksk"])
        crss = []
        for ksk in rotk.data:
            crss.append(ksk.data[1])
        return crss

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. GALOIS
    #### -------------------------------------------------------------------------------------------

    def generate_galois_crs(self, galk: data_struct):
        if galk.origin != types.origins["galk"]:
            raise errors.NotMatchType(origin=galk.origin, to=types.origins["galk"])
        crs_s = []
        for rotk in galk.data:
            crss = [ksk.data[1] for ksk in rotk.data]
            crs_s.append(crss)
        return crs_s

    def multiparty_create_galois_key(self, sk: data_struct, a: list) -> data_struct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        galois_key_parts = [
            self.multiparty_create_rotation_key(sk, self.galois_deltas[idx], a=a[idx])
            for idx in range(len(self.galois_deltas))
        ]

        galois_key = data_struct(
            data=galois_key_parts,
            include_special=True,
            montgomery_state=True,
            ntt_state=True,
            origin=types.origins["galk"],
            level=0,
            hash=self.hash,
            version=self.version
        )
        return galois_key

    def multiparty_generate_galois_key(self, galks: list[data_struct]) -> data_struct:
        cgalk = self.clone(galks[0])
        for galk in galks[1:]:  # galk
            for rotk_idx in range(len(galk.data)):  # rotk
                for ksk_idx in range(len(galk.data[rotk_idx].data)):  # ksk
                    update_parts = self.ntt.mont_add(
                        cgalk.data[rotk_idx].data[ksk_idx].data[0],
                        galk.data[rotk_idx].data[ksk_idx].data[0]
                    )
                    cgalk.data[rotk_idx].data[ksk_idx].data[0][0].copy_(update_parts[0], non_blocking=True)
        return cgalk

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. Evaluation Key
    #### -------------------------------------------------------------------------------------------

    def multiparty_sum_evk_share(self, evks_share: list[data_struct]):
        evk_sum = self.clone(evks_share[0])
        for evk_share in evks_share[1:]:
            for ksk_idx in range(len(evk_sum.data)):
                update_parts = self.ntt.mont_add(evk_sum.data[ksk_idx].data[0], evk_share.data[ksk_idx].data[0])
                for dev_id in range(len(update_parts)):
                    evk_sum.data[ksk_idx].data[0][dev_id].copy_(update_parts[dev_id], non_blocking=True)

        return evk_sum

    def multiparty_mult_evk_share_sum(self, evk_sum: data_struct, sk: data_struct):
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        evk_sum_mult = self.clone(evk_sum)

        for ksk_idx in range(len(evk_sum.data)):
            update_part_b = self.ntt.mont_mult(evk_sum_mult.data[ksk_idx].data[0], sk.data)
            update_part_a = self.ntt.mont_mult(evk_sum_mult.data[ksk_idx].data[1], sk.data)
            for dev_id in range(len(update_part_b)):
                evk_sum_mult.data[ksk_idx].data[0][dev_id].copy_(update_part_b[dev_id], non_blocking=True)
                evk_sum_mult.data[ksk_idx].data[1][dev_id].copy_(update_part_a[dev_id], non_blocking=True)

        return evk_sum_mult

    def multiparty_sum_evk_share_mult(self, evk_sum_mult: list[data_struct]) -> data_struct:
        cevk = self.clone(evk_sum_mult[0])
        for evk in evk_sum_mult[1:]:
            for ksk_idx in range(len(cevk.data)):
                update_part_b = self.ntt.mont_add(cevk.data[ksk_idx].data[0], evk.data[ksk_idx].data[0])
                update_part_a = self.ntt.mont_add(cevk.data[ksk_idx].data[1], evk.data[ksk_idx].data[1])
                for dev_id in range(len(update_part_b)):
                    cevk.data[ksk_idx].data[0][dev_id].copy_(update_part_b[dev_id], non_blocking=True)
                    cevk.data[ksk_idx].data[1][dev_id].copy_(update_part_a[dev_id], non_blocking=True)
        return cevk

    #### -------------------------------------------------------------------------------------------
    ####  Statistics
    #### -------------------------------------------------------------------------------------------

    def sqrt(self, ct: data_struct, evk: data_struct, e=0.0001, alpha=0.0001) -> data_struct:
        a = self.clone(ct)
        b = self.clone(ct)

        while e <= 1 - alpha:
            k = float(np.roots([1 - e ** 3, -6 + 6 * e ** 2, 9 - 9 * e])[1])
            t = self.mult_scalar(a, k, evk)
            b0 = self.sub_scalar(t, 3)
            b1 = self.mult_scalar(b, (k ** 0.5) / 2, evk)
            b = self.cc_mult(b0, b1, evk)

            a0 = self.mult_scalar(a, (k ** 3) / 4)
            t = self.sub_scalar(a, 3 / k)
            a1 = self.square(t, evk)
            a = self.cc_mult(a0, a1, evk)
            e = k * (3 - k) ** 2 / 4

        return b

    def var(self, ct: data_struct, evk: data_struct, gk: data_struct, relin=False) -> data_struct:
        ct_mean = self.mean(ct=ct, gk=gk)
        dev = self.sub(ct, ct_mean)
        dev = self.square(ct=dev, evk=evk, relin=relin)
        if not relin:
            dev = self.relinearize(ct_triplet=dev, evk=evk)
        ct_var = self.mean(ct=dev, gk=gk)
        return ct_var

    def std(self, ct: data_struct, evk: data_struct, gk: data_struct, relin=False) -> data_struct:
        ct_var = self.var(ct=ct, evk=evk, gk=gk, relin=relin)
        ct_std = self.sqrt(ct=ct_var, evk=evk)
        return ct_std
