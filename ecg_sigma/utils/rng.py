"""Deterministic per-record RNG helpers.

Reproducibility is critical: the same dataset + config must always produce
byte-identical outputs. We derive a child seed from a stable string key
(typically the patient/record id) so parallel workers do not collide.
"""

from __future__ import annotations

import hashlib

import numpy as np


def seeded_rng(base_seed: int, key: str = "") -> np.random.Generator:
    """Build a numpy ``Generator`` deterministically seeded by ``(base_seed, key)``.

    ``key`` should be a stable, non-secret identifier such as a record name
    or patient id; mixing it into the seed lets us run the pipeline
    in parallel while still getting deterministic output per record.
    """
    if not key:
        return np.random.default_rng(base_seed)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    salt = int.from_bytes(digest, "big", signed=False)
    return np.random.default_rng((base_seed ^ salt) & 0xFFFFFFFFFFFFFFFF)
