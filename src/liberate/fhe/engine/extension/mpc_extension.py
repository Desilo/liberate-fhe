from liberate.fhe.typing import *
from liberate.fhe.encdec import rotate
from .. import errors
from liberate.ntt import ntt_cuda
from ..ckks_engine import CkksEngine
from liberate.fhe.typing import *
from liberate.utils.mvc import strictype


class CkksEngineMPCExtension(CkksEngine):

    # -------------------------------------------------------------------------------------------
    # Multiparty.
    # -------------------------------------------------------------------------------------------

    def multiparty_public_crs(self, pk: PublicKey):
        crs = self.clone(pk).data[1]
        return crs

    @strictype
    def multiparty_create_public_key(
        self, sk: SecretKey, a=None, include_special=False
    ) -> PublicKey:
        # if sk.origin != origin_names["sk"]:
        #     raise errors.NotMatchType(origin=sk.origin, to=origin_names["sk"])
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
                self.ntt.q_prepack[mult_type][level][0], repeats=repeats
            )

        sa = self.ntt.mont_mult(a, sk.data, 0, mult_type)
        pk0 = self.ntt.mont_sub(e, sa, 0, mult_type)
        pk = PublicKey(
            data=[pk0, a],
            include_special=include_special,
            ntt_state=True,
            montgomery_state=True,
            # origin=origin_names["pk"],
            level=level,
            hash=self.hash,
            # version=self.version,
        )
        return pk

    def multiparty_create_collective_public_key(
        self, pks: list[DataStruct]
    ) -> PublicKey:
        (
            data,
            include_special,
            ntt_state,
            montgomery_state,
            origin,
            level,
            hash_,
            version,
        ) = pks[0]
        mult_type = -2 if include_special else -1
        b = [b.clone() for b in data[0]]  # num of gpus
        a = [a.clone() for a in data[1]]

        for pk in pks[1:]:
            b = self.ntt.mont_add(b, pk.data[0], lvl=0, mult_type=mult_type)

        cpk = PublicKey(
            (b, a),
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            # origin=origin_names["pk"],
            level=level,
            hash=self.hash,
            # version=self.version,
        )
        return cpk

    @strictype
    def multiparty_decrypt_head(self, ct: Ciphertext, sk: SecretKey):
        # if ct.origin != origin_names["ct"]:
        #     raise errors.NotMatchType(origin=ct.origin, to=origin_names["ct"])
        # if sk.origin != origin_names["sk"]:
        #     raise errors.NotMatchType(origin=sk.origin, to=origin_names["sk"])
        
        # if ct.ntt_state or ct.montgomery_state:
        #     raise errors.NotMatchDataStructState(origin=ct.origin)
        # if not sk.ntt_state or not sk.montgomery_state:
        #     raise errors.NotMatchDataStructState(origin=sk.origin)
        
        if ct.ntt_state:
            raise errors.NTTStateError(expected=False)
        if ct.montgomery_state:
            raise errors.MontgomeryStateError(expected=False)
        if not sk.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not sk.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)
        
        level = ct.level

        ct0 = ct.data[0][0]
        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)

        sk_data = sk.data[0][self.ntt.starts[level][0] :]

        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        pt = self.ntt.mont_add([ct0], sa, level)

        return pt

    @strictype
    def multiparty_decrypt_partial(
        self, ct: Ciphertext, sk: SecretKey
    ) -> DataStruct:
        # if ct.origin != origin_names["ct"]:
        #     raise errors.NotMatchType(origin=ct.origin, to=origin_names["ct"])
        # if sk.origin != origin_names["sk"]:
        #     raise errors.NotMatchType(origin=sk.origin, to=origin_names["sk"])
        
        # if ct.ntt_state or ct.montgomery_state:
        #     raise errors.NotMatchDataStructState(origin=ct.origin)
        # if not sk.ntt_state or not sk.montgomery_state:
        #     raise errors.NotMatchDataStructState(origin=sk.origin)
        
        if ct.ntt_state:
            raise errors.NTTStateError(expected=False)
        if ct.montgomery_state:
            raise errors.MontgomeryStateError(expected=False)
        if not sk.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not sk.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)

        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], ct.level)

        sk_data = sk.data[0][self.ntt.starts[ct.level][0] :]

        sa = self.ntt.mont_mult([a], [sk_data], ct.level)
        self.ntt.intt_exit(sa, ct.level)

        return sa

    def multiparty_decrypt_fusion(
        self, pcts: list, level=0, include_special=False
    ):
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

    @strictype
    def multiparty_create_key_switching_key(
        self, sk_src: SecretKey, sk_dst: SecretKey, a=None
    ) -> KeySwitchKey:
        # if (
        #     sk_src.origin != origin_names["sk"]
        #     or sk_src.origin != origin_names["sk"]
        # ):
        #     raise errors.NotMatchType(
        #         origin="not a secret key", to=origin_names["sk"]
        #     )
        
        # if (not sk_src.ntt_state) or (not sk_src.montgomery_state):
        #     raise errors.NotMatchDataStructState(origin=sk_src.origin)
        # if (not sk_dst.ntt_state) or (not sk_dst.montgomery_state):
        #     raise errors.NotMatchDataStructState(origin=sk_dst.origin)
        
        if not sk_src.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not sk_src.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)
        if not sk_dst.ntt_state:
            raise errors.NTTStateError(expected=True)
        if not sk_dst.montgomery_state:
            raise errors.MontgomeryStateError(expected=True)

        level = 0

        stops = self.ntt.stops[-1]
        Psk_src = [
            sk_src.data[di][: stops[di]].clone()
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
                pk = self.multiparty_create_public_key(
                    sk_dst, include_special=True, a=crs
                )
                key = tuple(part)
                astart = part[0]
                astop = part[-1] + 1
                shard = Psk_src[device_id][astart:astop]
                pk_data = pk.data[0][device_id][astart:astop]

                _2q = self.ntt.parts_pack[device_id][key]["_2q"]
                update_part = ntt_cuda.mont_add([pk_data], [shard], _2q)[0]
                pk_data.copy_(update_part, non_blocking=True)

                # Name the pk.
                pk_name = f"key switch key part index {global_part_id}"
                pk = pk._replace(origin=pk_name)

                ksk[global_part_id] = pk

        return KeySwitchKey(
            data=ksk,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            # origin=origin_names["ksk"],
            level=level,
            hash=self.hash,
            # version=self.version,
        )

    @strictype
    def multiparty_create_rotation_key(
        self, sk: SecretKey, delta: int, a=None
    ) -> RotationKey:
        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data)
        sk_new_data = [rotate(s, delta) for s in sk_new_data]
        self.ntt.ntt(sk_new_data)
        sk_rotated = DataStruct(
            data=sk_new_data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            # origin=origin_names["sk"],
            level=0,
            hash=self.hash,
            # version=self.version,
        )
        rotk = RotationKey.wrap(
            self.multiparty_create_key_switching_key(sk_rotated, sk, a=a)
        )
        # rotk = rotk._replace(origin=origin_names["rotk"] + f"{delta}")
        return rotk

    @strictype
    def multiparty_generate_rotation_key(
        self, rotks: List[RotationKey]
    ) -> RotationKey:
        crotk = self.clone(rotks[0])
        for rotk in rotks[1:]:
            for ksk_idx in range(len(rotk.data)):
                update_parts = self.ntt.mont_add(
                    crotk.data[ksk_idx].data[0], rotk.data[ksk_idx].data[0]
                )
                crotk.data[ksk_idx].data[0][0].copy_(
                    update_parts[0], non_blocking=True
                )
        return crotk

    @strictype
    def generate_rotation_crs(self, rotk: Union[RotationKey, KeySwitchKey]):
        # if (
        #     origin_names["rotk"] not in rotk.origin
        #     and origin_names["ksk"] != rotk.origin
        # ):
        #     raise errors.NotMatchType(
        #         origin=rotk.origin, to=origin_names["ksk"]
        #     )
        crss = []
        for ksk in rotk.data:
            crss.append(ksk.data[1])
        return crss

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. GALOIS
    #### -------------------------------------------------------------------------------------------

    @strictype
    def generate_galois_crs(self, galk: GaloisKey):
        # if galk.origin != origin_names["galk"]:
        #     raise errors.NotMatchType(
        #         origin=galk.origin, to=origin_names["galk"]
        #     )
        crs_s = []
        for rotk in galk.data:
            crss = [ksk.data[1] for ksk in rotk.data]
            crs_s.append(crss)
        return crs_s

    @strictype
    def multiparty_create_galois_key(
        self, sk: SecretKey, a: list
    ) -> GaloisKey:
        # if sk.origin != origin_names["sk"]:
        #     raise errors.NotMatchType(origin=sk.origin, to=origin_names["sk"])
        
        galois_deltas = [2**i for i in range(self.ctx.logN - 1)]
        galois_key_parts = [
            self.multiparty_create_rotation_key(
                sk, galois_deltas[idx], a=a[idx]
            )
            for idx in range(len(galois_deltas))
        ]

        galois_key = GaloisKey(
            data=galois_key_parts,
            include_special=True,
            montgomery_state=True,
            ntt_state=True,
            # origin=origin_names["galk"],
            level=0,
            hash=self.hash,
            # version=self.version,
        )
        return galois_key

    def multiparty_generate_galois_key(
        self, galks: list[DataStruct]
    ) -> DataStruct:
        cgalk = self.clone(galks[0])
        for galk in galks[1:]:  # galk
            for rotk_idx in range(len(galk.data)):  # rotk
                for ksk_idx in range(len(galk.data[rotk_idx].data)):  # ksk
                    update_parts = self.ntt.mont_add(
                        cgalk.data[rotk_idx].data[ksk_idx].data[0],
                        galk.data[rotk_idx].data[ksk_idx].data[0],
                    )
                    cgalk.data[rotk_idx].data[ksk_idx].data[0][0].copy_(
                        update_parts[0], non_blocking=True
                    )
        return cgalk

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. Evaluation Key
    #### -------------------------------------------------------------------------------------------

    def multiparty_sum_evk_share(self, evks_share: list[DataStruct]):
        evk_sum = self.clone(evks_share[0])
        for evk_share in evks_share[1:]:
            for ksk_idx in range(len(evk_sum.data)):
                update_parts = self.ntt.mont_add(
                    evk_sum.data[ksk_idx].data[0],
                    evk_share.data[ksk_idx].data[0],
                )
                for dev_id in range(len(update_parts)):
                    evk_sum.data[ksk_idx].data[0][dev_id].copy_(
                        update_parts[dev_id], non_blocking=True
                    )

        return evk_sum

    @strictype
    def multiparty_mult_evk_share_sum(
        self, evk_sum: DataStruct, sk: SecretKey
    ) -> DataStruct:
        # if sk.origin != origin_names["sk"]:
        #     raise errors.NotMatchType(origin=sk.origin, to=origin_names["sk"])

        evk_sum_mult = self.clone(evk_sum)

        for ksk_idx in range(len(evk_sum.data)):
            update_part_b = self.ntt.mont_mult(
                evk_sum_mult.data[ksk_idx].data[0], sk.data
            )
            update_part_a = self.ntt.mont_mult(
                evk_sum_mult.data[ksk_idx].data[1], sk.data
            )
            for dev_id in range(len(update_part_b)):
                evk_sum_mult.data[ksk_idx].data[0][dev_id].copy_(
                    update_part_b[dev_id], non_blocking=True
                )
                evk_sum_mult.data[ksk_idx].data[1][dev_id].copy_(
                    update_part_a[dev_id], non_blocking=True
                )

        return evk_sum_mult

    def multiparty_sum_evk_share_mult(
        self, evk_sum_mult: list[DataStruct]
    ) -> DataStruct:
        cevk = self.clone(evk_sum_mult[0])
        for evk in evk_sum_mult[1:]:
            for ksk_idx in range(len(cevk.data)):
                update_part_b = self.ntt.mont_add(
                    cevk.data[ksk_idx].data[0], evk.data[ksk_idx].data[0]
                )
                update_part_a = self.ntt.mont_add(
                    cevk.data[ksk_idx].data[1], evk.data[ksk_idx].data[1]
                )
                for dev_id in range(len(update_part_b)):
                    cevk.data[ksk_idx].data[0][dev_id].copy_(
                        update_part_b[dev_id], non_blocking=True
                    )
                    cevk.data[ksk_idx].data[1][dev_id].copy_(
                        update_part_a[dev_id], non_blocking=True
                    )
        return cevk
