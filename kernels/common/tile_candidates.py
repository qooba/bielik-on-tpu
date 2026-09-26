_BLOCK_M_POOL = (8, 16, 32, 64, 128, 256, 512)
_BLOCK_N_OR_K_POOL = (128, 256, 512)
_BLOCK_N_OR_K_WIDE_POOL = (128, 256, 512, 1024, 2048, 4096)


def block_m_candidates(M: int) -> list[int]:
    cands = {c for c in _BLOCK_M_POOL if c <= M}
    cands.add(M)
    return sorted(cands)


def block_n_or_k_candidates(dim: int) -> list[int]:
    cands = {c for c in _BLOCK_N_OR_K_POOL if c <= dim}
    cands.add(dim)
    return sorted(cands)


def block_n_or_k_candidates_wide(dim: int) -> list[int]:
    cands = {c for c in _BLOCK_N_OR_K_WIDE_POOL if c <= dim}
    cands.add(dim)
    return sorted(cands)


def block_k_candidates_when_m_lt_8(K: int, wide: bool = False) -> list[int]:
    pool = block_n_or_k_candidates_wide(K) if wide else block_n_or_k_candidates(K)
    return [c for c in pool if K % c == 0]
