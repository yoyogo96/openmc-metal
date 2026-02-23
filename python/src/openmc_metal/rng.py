import math

PHILOX_M: int = 0xD2511F53
PHILOX_W: int = 0x9E3779B9
UINT32_TO_FLOAT: float = 2.3283064365e-10


def philox2x32_round(ctr_lo: int, ctr_hi: int, key: int) -> tuple[int, int]:
    product = PHILOX_M * ctr_lo  # 64-bit result
    hi = (product >> 32) & 0xFFFFFFFF
    lo = product & 0xFFFFFFFF
    return (hi ^ key ^ ctr_hi) & 0xFFFFFFFF, lo


def philox2x32_10(counter_lo: int, counter_hi: int, key: int) -> tuple[int, int]:
    ctr_lo = counter_lo & 0xFFFFFFFF
    ctr_hi = counter_hi & 0xFFFFFFFF
    k = key & 0xFFFFFFFF
    for _ in range(10):
        ctr_lo, ctr_hi = philox2x32_round(ctr_lo, ctr_hi, k)
        k = (k + PHILOX_W) & 0xFFFFFFFF
    return ctr_lo, ctr_hi


class PhiloxRNG:
    def __init__(self, key: int, counter_hi: int = 0, counter_lo: int = 0) -> None:
        self.key = key & 0xFFFFFFFF
        self.counter_hi = counter_hi & 0xFFFFFFFF
        self.counter_lo = counter_lo & 0xFFFFFFFF

    def uniform(self) -> float:
        result_lo, _result_hi = philox2x32_10(self.counter_lo, self.counter_hi, self.key)
        self.counter_lo = (self.counter_lo + 1) & 0xFFFFFFFF
        return result_lo * UINT32_TO_FLOAT

    def sample_isotropic_direction(self) -> tuple[float, float, float]:
        cos_theta = 2.0 * self.uniform() - 1.0
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi = 2.0 * math.pi * self.uniform()
        return (sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta)

    def sample_discrete(self, cdf: list[float]) -> int:
        xi = self.uniform()
        for i, cum_prob in enumerate(cdf):
            if xi < cum_prob:
                return i
        return len(cdf) - 1
