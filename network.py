"""
network.py — Kademlia overlay network construction.

Builds the network layer used by all three protocols:
  - Random 16-bit node IDs
  - Per-node Kademlia k-bucket routing tables (XOR-metric, globally precomputed)
  - Two-tier bandwidth assignment (80% low, 20% high)
  - On-demand LogNormal latency sampling
"""

import math
import random
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


BANDWIDTH_TIERS = {
    'high': 1024 * 1024 * 1024,   # 1 GB/s
    'low':  50   * 1024 * 1024,   # 50 MB/s
}

# LogNormal latency parameters (for one-way link latency in milliseconds).
# These correspond to a median latency of ~50ms with moderate variance,
# If X ~ LogNormal(mu, sigma), then median = exp(mu).
_LATENCY_MU    = math.log(50)   # median = 50 ms
_LATENCY_SIGMA = 0.8            # moderate spread; 90th percentile ~120ms

# Number of ID bits → number of buckets.
ID_BITS = 16
NUM_BUCKETS = ID_BITS  # bucket k covers XOR distance [2^k, 2^(k+1))

# Max nodes stored per bucket (standard Kademlia uses k=20).
# Each node randomly samples up to BUCKET_CAP peers per bucket at construction time.
BUCKET_CAP = 10


# ---------------------------------------------------------------------------
# Node ID assignment
# ---------------------------------------------------------------------------

def assign_node_ids(n: int, seed: int) -> dict:
    """
    Assign a unique random ID to each of the n nodes.

    Parameters
    ----------
    n    : number of nodes
    seed : RNG seed for reproducibility

    Returns
    -------
    node_ids : dict[node_index -> 16-bit int]
        e.g. {0: 52341, 1: 3892, ...}
    """
    rng = random.Random(seed)
    all_ids = rng.sample(range(2**ID_BITS), n)  # sampling ensures uniqueness
    return {i: all_ids[i] for i in range(n)}


# ---------------------------------------------------------------------------
# Kademlia routing table construction
# ---------------------------------------------------------------------------

def _bucket_level(xor_dist: int) -> int:
    """
    Given an XOR distance, return the bucket index k such that
    XOR distance falls in [2^k, 2^(k+1)).

    XOR distance 0 (a node with itself) is not a valid routing entry
    and should never be passed here.
    """
    return xor_dist.bit_length() - 1


def build_kademlia_tables(node_ids: dict, seed: int) -> dict:
    """
    Precompute the Kademlia routing table for every node.
 
    For each node u, bucket k contains up to BUCKET_CAP randomly sampled
    nodes v such that:
        2^k <= XOR(id[u], id[v]) < 2^(k+1)
 
    All candidates at each bucket level are first collected, then randomly
    sampled down to BUCKET_CAP. The seed ensures reproducibility.
 
    Parameters
    ----------
    node_ids : dict[node_index -> 16-bit int]
    seed     : RNG seed for reproducible bucket sampling
 
    Returns
    -------
    tables : dict[node_index -> list of 16 buckets]
        tables[u][k] = list of up to BUCKET_CAP node indices at bucket level k from u.
    """
    rng = random.Random(seed)
    nodes = list(node_ids.keys())
 
    # First pass: collect all candidates per (node, bucket) pair.
    candidates = {u: [[] for _ in range(NUM_BUCKETS)] for u in nodes}
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            xor_dist = node_ids[u] ^ node_ids[v]
            k = _bucket_level(xor_dist)
            candidates[u][k].append(v)
 
    # Second pass: sample down to BUCKET_CAP per bucket.
    tables = {u: [[] for _ in range(NUM_BUCKETS)] for u in nodes}
    for u in nodes:
        for k in range(NUM_BUCKETS):
            pool = candidates[u][k]
            if len(pool) <= BUCKET_CAP:
                tables[u][k] = pool[:]
            else:
                tables[u][k] = rng.sample(pool, BUCKET_CAP)
 
    return tables

# ---------------------------------------------------------------------------
# Bandwidth tier assignment
# ---------------------------------------------------------------------------

def assign_bandwidth_tiers(n: int, seed: int) -> dict:
    """
    Assign bandwidth tiers to nodes: 20% 'high', 80% 'low'.

    Parameters
    ----------
    n    : number of nodes
    seed : RNG seed for reproducibility

    Returns
    -------
    tiers : dict[node_index -> 'high' or 'low']
    """
    rng = random.Random(seed)
    n_high = max(1, int(0.2 * n))
    high_nodes = set(rng.sample(range(n), n_high))
    return {i: ('high' if i in high_nodes else 'low') for i in range(n)}


# ---------------------------------------------------------------------------
# Latency sampling
# ---------------------------------------------------------------------------

def sample_latency(rng: np.random.Generator) -> float:
    """
    Sample a one-way link latency in milliseconds from a LogNormal distribution.

    Each call represents independent per-packet jitter — the same link
    can have a different latency each time a packet traverses it.

    Parameters
    ----------
    rng : numpy Generator (e.g. np.random.default_rng(seed))
        Passed in from the simulation context for reproducibility.

    Returns
    -------
    latency_ms : float
    """
    return rng.lognormal(mean=_LATENCY_MU, sigma=_LATENCY_SIGMA)
