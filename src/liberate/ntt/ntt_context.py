import datetime
import time

import numpy as np
import torch

from liberate.fhe.context.ckks_context import CkksContext

from . import ntt_cuda
from .rns_partition import rns_partition


class NTTContext:
    def __init__(
        self,
        ctx: CkksContext,
        index_type=torch.int32,
        devices=None,
        verbose=False,
    ):
        # Mark the start time.
        t0 = time.time()

        # Set devices first.
        if devices is None:
            gpu_count = torch.cuda.device_count()
            self.devices = [f"cuda:{i}" for i in range(gpu_count)]
        else:
            self.devices = devices

        self.num_devices = len(self.devices)

        # Transfer input parameters.
        self.index_type = index_type
        self.ckksCtx = ctx

        if verbose:
            print(
                f"[{str(datetime.datetime.now())}] I have received the context:\n"
            )
            self.ckksCtx.init_print()
            print(
                f"[{str(datetime.datetime.now())}] Requested devices for computation are {self.devices}."
            )

        self.num_ordinary_primes = self.ckksCtx.num_scales + 1
        self.num_special_primes = self.ckksCtx.num_special_primes

        self.p = rns_partition(
            self.num_ordinary_primes, self.num_special_primes, self.num_devices
        )
        if verbose:
            print(
                f"[{str(datetime.datetime.now())}] I have generated a partitioning scheme."
            )
            print(
                f"[{str(datetime.datetime.now())}] I have in total {self.num_levels} levels available."
            )
            print(
                f"[{str(datetime.datetime.now())}] I have {self.num_ordinary_primes} ordinary primes."
            )
            print(
                f"[{str(datetime.datetime.now())}] I have {self.num_special_primes} special primes."
            )

        self.prepare_parameters()

        if verbose:
            print(
                f"[{str(datetime.datetime.now())}] I prepared ntt parameters."
            )

        t1 = time.time()
        if verbose:
            print(
                f"[{str(datetime.datetime.now())}] ntt initialization took {(t1 - t0):.2f} seconds."
            )

        self.qlists = [qi.tolist() for qi in self.q]

        astop_special = [
            len(d) for d in self.p.destination_arrays_with_special[0]
        ]
        astop_ordinary = [len(d) for d in self.p.destination_arrays[0]]
        self.starts = self.p.diff

        self.stops = [astop_special, astop_ordinary]

        self.generate_parts_pack()
        self.pre_package()

    @property
    def num_levels(self) -> int:
        return self.ckksCtx.num_scales + 1

    # -------------------------------------------------------------------------------------------------
    # Arrange according to partitioning scheme input variables, and copy to GPUs for fast access.
    # -------------------------------------------------------------------------------------------------

    def partition_variable(self, variable):
        np_v = np.array(variable, dtype=self.ckksCtx.numpy_dtype)

        v_special = []
        dest = self.p.d_special
        for dev_id in range(self.num_devices):
            d = dest[dev_id]
            parted_v = np_v[d]
            v_special.append(
                torch.from_numpy(parted_v).to(self.devices[dev_id])
            )

        return v_special

    def copy_to_devices(self, variable):
        return [
            torch.tensor(variable, dtype=self.index_type, device=device)
            for device in self.devices
        ]

    def psi_enter(self):
        Rs = self.Rs
        ql = self.ql
        qh = self.qh
        kl = self.kl
        kh = self.kh

        p = self.psi

        a = [psi.view(psi.size(0), -1) for psi in p]

        ntt_cuda.mont_enter(a, Rs, ql, qh, kl, kh)

        p = self.ipsi
        a = [psi.view(psi.size(0), -1) for psi in p]
        ntt_cuda.mont_enter(a, Rs, ql, qh, kl, kh)

    def Ninv_enter(self):
        self.Ninv = [
            (self.ckksCtx.N_inv[i] * self.ckksCtx.R) % self.ckksCtx.q[i]
            for i in range(len(self.ckksCtx.q))
        ]

    def prepare_parameters(self):
        scale = 2**self.ckksCtx.scale_bits
        self.Rs_scale = self.partition_variable(
            [
                (Rs * scale) % q
                for Rs, q in zip(self.ckksCtx.R_square, self.ckksCtx.q)
            ]
        )

        self.Rs = self.partition_variable(self.ckksCtx.R_square)

        self.q = self.partition_variable(self.ckksCtx.q)
        self._2q = self.partition_variable(self.ckksCtx.q_double)
        self.ql = self.partition_variable(self.ckksCtx.q_lower_bits)
        self.qh = self.partition_variable(self.ckksCtx.q_higher_bits)
        self.kl = self.partition_variable(self.ckksCtx.k_lower_bits)
        self.kh = self.partition_variable(self.ckksCtx.k_higher_bits)

        self.even = self.copy_to_devices(self.ckksCtx.forward_even_indices)
        self.odd = self.copy_to_devices(self.ckksCtx.forward_odd_indices)
        self.ieven = self.copy_to_devices(self.ckksCtx.backward_even_indices)
        self.iodd = self.copy_to_devices(self.ckksCtx.backward_odd_indices)

        self.psi = self.partition_variable(self.ckksCtx.forward_psi)
        self.ipsi = self.partition_variable(self.ckksCtx.backward_psi_inv)

        self.Ninv_enter()
        self.Ninv = self.partition_variable(self.Ninv)

        self.psi_enter()

        self.mont_pack0 = [self.ql, self.qh, self.kl, self.kh]

        self.ntt_pack0 = [
            self.even,
            self.odd,
            self.psi,
            self._2q,
            self.ql,
            self.qh,
            self.kl,
            self.kh,
        ]

        self.intt_pack0 = [
            self.ieven,
            self.iodd,
            self.ipsi,
            self.Ninv,
            self._2q,
            self.ql,
            self.qh,
            self.kl,
            self.kh,
        ]

    def param_pack(self, param, astart, astop, remove_empty=True):
        pack = [
            param[dev_id][astart[dev_id] : astop[dev_id]]
            for dev_id in range(self.num_devices)
        ]

        remove_empty_f = lambda x: [xi for xi in x if len(xi) > 0]
        if remove_empty:
            pack = remove_empty_f(pack)
        return pack

    def mont_pack(self, astart, astop, remove_empty=True):
        return [
            self.param_pack(param, astart, astop, remove_empty)
            for param in self.mont_pack0
        ]

    def ntt_pack(self, astart, astop, remove_empty=True):
        remove_empty_f_x = lambda x: [xi for xi in x if len(xi) > 0]

        remove_empty_f_xy = lambda x, y: [
            xi for xi, yi in zip(x, y) if len(yi) > 0
        ]

        even_odd = self.ntt_pack0[:2]
        rest = [
            self.param_pack(param, astart, astop, remove_empty=False)
            for param in self.ntt_pack0[2:]
        ]

        if remove_empty:
            even_odd = [remove_empty_f_xy(eo, rest[0]) for eo in even_odd]
            rest = [remove_empty_f_x(r) for r in rest]

        return even_odd + rest

    def intt_pack(self, astart, astop, remove_empty=True):
        remove_empty_f_x = lambda x: [xi for xi in x if len(xi) > 0]

        remove_empty_f_xy = lambda x, y: [
            xi for xi, yi in zip(x, y) if len(yi) > 0
        ]

        even_odd = self.intt_pack0[:2]
        rest = [
            self.param_pack(param, astart, astop, remove_empty=False)
            for param in self.intt_pack0[2:]
        ]

        if remove_empty:
            even_odd = [remove_empty_f_xy(eo, rest[0]) for eo in even_odd]
            rest = [remove_empty_f_x(r) for r in rest]

        return even_odd + rest

    def start_stop(self, lvl, mult_type):
        return self.starts[lvl], self.stops[mult_type]

    # -------------------------------------------------------------------------------------------------
    # Package by parts.
    # -------------------------------------------------------------------------------------------------

    def params_pack_device(self, device_id, astart, astop):
        starts = [0] * self.num_devices
        stops = [0] * self.num_devices

        starts[device_id] = astart
        stops[device_id] = astop + 1

        stst = [starts, stops]

        item = {}

        item["mont_pack"] = self.mont_pack(*stst)
        item["ntt_pack"] = self.ntt_pack(*stst)
        item["intt_pack"] = self.intt_pack(*stst)
        item["Rs"] = self.param_pack(self.Rs, *stst)
        item["Rs_scale"] = self.param_pack(self.Rs_scale, *stst)
        item["_2q"] = self.param_pack(self._2q, *stst)
        item["qlist"] = self.param_pack(self.qlists, *stst)

        return item

    def generate_parts_pack(self):
        blank_L_enter = [None] * self.num_devices

        self.parts_pack = []

        for device_id in range(self.num_devices):
            self.parts_pack.append({})

            for i in range(
                len(self.p.destination_arrays_with_special[0][device_id])
            ):
                self.parts_pack[device_id][i,] = self.params_pack_device(
                    device_id, i, i
                )

            for level in range(self.num_levels):
                for mult_type in [-1, -2]:
                    starts, stops = self.start_stop(level, mult_type)
                    astart = starts[device_id]

                    astop = stops[device_id] - 1

                    key = tuple(range(astart, astop + 1))

                    if len(key) > 0:
                        if key not in self.parts_pack[device_id]:
                            self.parts_pack[device_id][key] = (
                                self.params_pack_device(
                                    device_id, astart, astop
                                )
                            )

                for p in self.p.p_special[level][device_id]:
                    key = tuple(p)
                    if key not in self.parts_pack[device_id].keys():
                        astart = p[0]
                        astop = p[-1]
                        self.parts_pack[device_id][key] = (
                            self.params_pack_device(device_id, astart, astop)
                        )

        for device_id in range(self.num_devices):
            for level in range(self.num_levels):
                # We do basis extension for only ordinary parts.
                for part_index, part in enumerate(
                    self.p.destination_parts[level][device_id]
                ):
                    key = tuple(self.p.p[level][device_id][part_index])

                    # Check if Y and L are already calculated for this part.
                    if (
                        "Y_scalar"
                        not in self.parts_pack[device_id][key].keys()
                    ):
                        alpha = len(part)
                        m = [self.ckksCtx.q[idx] for idx in part]
                        L = [m[0]]

                        for i in range(1, alpha - 1):
                            L.append(L[-1] * m[i])

                        Y_scalar = []
                        L_scalar = []
                        for i in range(alpha - 1):
                            L_inv = pow(L[i], -1, m[i + 1])
                            L_inv_R = (L_inv * self.ckksCtx.R) % m[i + 1]
                            Y_scalar.append(L_inv_R)

                            if (i + 2) < alpha:
                                L_scalar.append([])
                                for j in range(i + 2, alpha):
                                    L_scalar[i].append(
                                        (L[i] * self.ckksCtx.R) % m[j]
                                    )

                        L_enter_devices = []
                        for target_device_id in range(self.num_devices):
                            dest = self.p.destination_arrays_with_special[0][
                                target_device_id
                            ]
                            q = [self.ckksCtx.q[idx] for idx in dest]
                            Rs = [self.ckksCtx.R_square[idx] for idx in dest]

                            L_enter = []
                            for i in range(alpha - 1):
                                L_enter.append([])
                                for j in range(len(dest)):
                                    L_Rs = (L[i] * Rs[j]) % q[j]
                                    L_enter[i].append(L_Rs)
                            L_enter_devices.append(L_enter)

                        device = self.devices[device_id]

                        if len(Y_scalar) > 0:
                            Y_scalar = torch.tensor(
                                Y_scalar,
                                dtype=self.ckksCtx.torch_dtype,
                                device=device,
                            )
                            self.parts_pack[device_id][key][
                                "Y_scalar"
                            ] = Y_scalar

                            for target_device_id in range(self.num_devices):
                                target_device = self.devices[target_device_id]

                                L_enter_devices[target_device_id] = [
                                    torch.tensor(
                                        Li,
                                        dtype=self.ckksCtx.torch_dtype,
                                        device=target_device,
                                    )
                                    for Li in L_enter_devices[target_device_id]
                                ]

                            self.parts_pack[device_id][key][
                                "L_enter"
                            ] = L_enter_devices

                        else:
                            self.parts_pack[device_id][key]["Y_scalar"] = None
                            self.parts_pack[device_id][key][
                                "L_enter"
                            ] = blank_L_enter

                        if len(L_scalar) > 0:
                            L_scalar = [
                                torch.tensor(
                                    Li,
                                    dtype=self.ckksCtx.torch_dtype,
                                    device=device,
                                )
                                for Li in L_scalar
                            ]
                            self.parts_pack[device_id][key][
                                "L_scalar"
                            ] = L_scalar
                        else:
                            self.parts_pack[device_id][key]["L_scalar"] = None

    # -------------------------------------------------------------------------------------------------
    # Pre-packaging.
    # -------------------------------------------------------------------------------------------------
    def pre_package(self):
        self.mont_prepack = []
        self.ntt_prepack = []
        self.intt_prepack = []
        self.Rs_prepack = []
        self.Rs_scale_prepack = []
        self._2q_prepack = []

        # q_prepack is a list of lists, not tensors.
        # We need this for generating uniform samples.
        self.q_prepack = []

        for device_id in range(self.num_devices):
            mont_prepack = []
            ntt_prepack = []
            intt_prepack = []
            Rs_prepack = []
            Rs_scale_prepack = []
            _2q_prepack = []
            q_prepack = []
            for lvl in range(self.num_levels):
                mont_prepack_part = []
                ntt_prepack_part = []
                intt_prepack_part = []
                Rs_prepack_part = []
                Rs_scale_prepack_part = []
                _2q_prepack_part = []
                q_prepack_part = []
                for part in self.p.p_special[lvl][device_id]:
                    key = tuple(part)
                    item = self.parts_pack[device_id][key]

                    mont_prepack_part.append(item["mont_pack"])
                    ntt_prepack_part.append(item["ntt_pack"])
                    intt_prepack_part.append(item["intt_pack"])
                    Rs_prepack_part.append(item["Rs"])
                    Rs_scale_prepack_part.append(item["Rs_scale"])
                    _2q_prepack_part.append(item["_2q"])
                    q_prepack_part.append(item["qlist"])

                for mult_type in [-2, -1]:
                    starts, stops = self.start_stop(lvl, mult_type)
                    astart = starts[device_id]

                    astop = stops[device_id] - 1

                    key = tuple(range(astart, astop + 1))

                    if len(key) > 0:
                        item = self.parts_pack[device_id][key]

                        mont_prepack_part.append(item["mont_pack"])
                        ntt_prepack_part.append(item["ntt_pack"])
                        intt_prepack_part.append(item["intt_pack"])
                        Rs_prepack_part.append(item["Rs"])
                        Rs_scale_prepack_part.append(item["Rs_scale"])
                        _2q_prepack_part.append(item["_2q"])
                        q_prepack_part.append(item["qlist"])

                    else:
                        mont_prepack_part.append(None)
                        ntt_prepack_part.append(None)
                        intt_prepack_part.append(None)
                        Rs_prepack_part.append(None)
                        Rs_scale_prepack_part.append(None)
                        _2q_prepack_part.append(None)
                        q_prepack_part.append(None)

                mont_prepack.append(mont_prepack_part)
                ntt_prepack.append(ntt_prepack_part)
                intt_prepack.append(intt_prepack_part)
                Rs_prepack.append(Rs_prepack_part)
                Rs_scale_prepack.append(Rs_scale_prepack_part)
                _2q_prepack.append(_2q_prepack_part)
                q_prepack.append(q_prepack_part)

            self.mont_prepack.append(mont_prepack)
            self.ntt_prepack.append(ntt_prepack)
            self.intt_prepack.append(intt_prepack)
            self.Rs_prepack.append(Rs_prepack)
            self.Rs_scale_prepack.append(Rs_scale_prepack)
            self._2q_prepack.append(_2q_prepack)
            self.q_prepack.append(q_prepack)

        for mult_type in [-2, -1]:
            mont_prepack = []
            ntt_prepack = []
            intt_prepack = []
            Rs_prepack = []
            Rs_scale_prepack = []
            _2q_prepack = []
            q_prepack = []
            for lvl in range(self.num_levels):
                stst = self.start_stop(lvl, mult_type)
                mont_prepack.append([self.mont_pack(*stst)])
                ntt_prepack.append([self.ntt_pack(*stst)])
                intt_prepack.append([self.intt_pack(*stst)])
                Rs_prepack.append([self.param_pack(self.Rs, *stst)])
                Rs_scale_prepack.append(
                    [self.param_pack(self.Rs_scale, *stst)]
                )
                _2q_prepack.append([self.param_pack(self._2q, *stst)])
                q_prepack.append([self.param_pack(self.qlists, *stst)])
            self.mont_prepack.append(mont_prepack)
            self.ntt_prepack.append(ntt_prepack)
            self.intt_prepack.append(intt_prepack)
            self.Rs_prepack.append(Rs_prepack)
            self.Rs_scale_prepack.append(Rs_scale_prepack)
            self._2q_prepack.append(_2q_prepack)
            self.q_prepack.append(q_prepack)

    # -------------------------------------------------------------------------------------------------
    # Helper functions to do the Montgomery and NTT operations.
    # -------------------------------------------------------------------------------------------------

    def mont_enter(self, a, lvl=0, mult_type=-1, part=0):
        ntt_cuda.mont_enter(
            a,
            self.Rs_prepack[mult_type][lvl][part],
            *self.mont_prepack[mult_type][lvl][part],
        )

    def mont_enter_scale(self, a, lvl=0, mult_type=-1, part=0):
        ntt_cuda.mont_enter(
            a,
            self.Rs_scale_prepack[mult_type][lvl][part],
            *self.mont_prepack[mult_type][lvl][part],
        )

    def mont_enter_scalar(self, a, b, lvl=0, mult_type=-1, part=0):
        ntt_cuda.mont_enter(a, b, *self.mont_prepack[mult_type][lvl][part])

    def mont_mult(self, a, b, lvl=0, mult_type=-1, part=0):
        return ntt_cuda.mont_mult(
            a, b, *self.mont_prepack[mult_type][lvl][part]
        )

    def ntt(self, a, lvl=0, mult_type=-1, part=0):
        ntt_cuda.ntt(a, *self.ntt_prepack[mult_type][lvl][part])

    def enter_ntt(self, a, lvl=0, mult_type=-1, part=0):
        ntt_cuda.enter_ntt(
            a,
            self.Rs_prepack[mult_type][lvl][part],
            *self.ntt_prepack[mult_type][lvl][part],
        )

    def intt(self, a, lvl=0, mult_type=-1, part=0):
        ntt_cuda.intt(a, *self.intt_prepack[mult_type][lvl][part])

    def mont_redc(self, a, lvl=0, mult_type=-1, part=0):
        ntt_cuda.mont_redc(a, *self.mont_prepack[mult_type][lvl][part])

    def intt_exit(self, a, lvl=0, mult_type=-1, part=0):
        ntt_cuda.intt_exit(a, *self.intt_prepack[mult_type][lvl][part])

    def intt_exit_reduce(self, a, lvl=0, mult_type=-1, part=0):
        ntt_cuda.intt_exit_reduce(a, *self.intt_prepack[mult_type][lvl][part])

    def intt_exit_reduce_signed(self, a, lvl=0, mult_type=-1, part=0):
        ntt_cuda.intt_exit_reduce_signed(
            a, *self.intt_prepack[mult_type][lvl][part]
        )

    def reduce_2q(self, a, lvl=0, mult_type=-1, part=0):
        ntt_cuda.reduce_2q(a, self._2q_prepack[mult_type][lvl][part])

    def make_signed(self, a, lvl=0, mult_type=-1, part=0):
        ntt_cuda.make_signed(a, self._2q_prepack[mult_type][lvl][part])

    def make_unsigned(self, a, lvl=0, mult_type=-1, part=0):
        ntt_cuda.make_unsigned(a, self._2q_prepack[mult_type][lvl][part])

    def mont_add(self, a, b, lvl=0, mult_type=-1, part=0):
        return ntt_cuda.mont_add(a, b, self._2q_prepack[mult_type][lvl][part])

    def mont_sub(self, a, b, lvl=0, mult_type=-1, part=0):
        return ntt_cuda.mont_sub(a, b, self._2q_prepack[mult_type][lvl][part])

    def tile_unsigned(self, a, lvl=0, mult_type=-1, part=0):
        return ntt_cuda.tile_unsigned(
            a, self._2q_prepack[mult_type][lvl][part]
        )
