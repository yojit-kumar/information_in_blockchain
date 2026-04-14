"""
benchmark.py — Entry point for broadcast protocol simulation benchmarks.

Runs Kadcast, KadRLNC, and OPTIMUMP2P over multiple seeds and blocks,
writing raw per-node delivery timestamps and control message logs to results/.

Usage:
    python benchmark.py

Results are written to results/ as CSVs, one per protocol per seed.
Analyse with Jupyter notebooks.
"""

import random
import sys
import os
import numpy as np

# Make sure sim/ root is on the path when running from any directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simpy_engine import build_context, run_simulation, Message
from metrics import MetricsCollector
import protocols.kadcast   as kadcast
import protocols.kadrlnc   as kadrlnc
import protocols.optimump2p as optimump2p

# ===========================================================================
# CONFIG — all simulation parameters in one place
# ===========================================================================

# --- Network ---
N_NODES    = 1000
SEED_START = 0
N_SEEDS    = 10       # number of independent runs per protocol

# --- Block ---
N_BLOCKS    = 5                    # blocks published per run per seed
BLOCK_SIZE  = 1024 * 1024          # 1 MB

# --- Shared ---
K           = 32                   # shards / chunks needed to decode
SHARD_SIZE  = BLOCK_SIZE // K      # bytes per shard/chunk

# --- Kadcast ---
F           = 0.15                 # FEC overhead fraction
BETA        = 3                    # peers per bucket for forwarding

# --- KadRLNC ---
RLNC_DELAY  = 3.0                  # simulated decode+recode delay in ms

# --- OPTIMUMP2P ---
P           = 8                    # shard multiplier (total shards = k*p)
D_MESH      = 4                    # mesh degree (peers per node)

# --- Source node selection ---
# 'fixed'  : same source node (node 0) for every block
# 'random' : random source node per block
SOURCE_MODE = 'random'

# --- Results ---
RESULTS_DIR = 'results/base'

# ===========================================================================
# Derived config dicts passed into SimContext
# ===========================================================================

KADCAST_CONFIG = {
    'k'               : K,
    'f'               : F,
    'beta'            : BETA,
    'chunk_size_bytes': SHARD_SIZE,
    'shard_size_bytes': SHARD_SIZE,
    'p'               : P,
    'rlnc_delay_ms'   : RLNC_DELAY,
    'beta_rlnc'       : 2,
    'r'               : K // 2,
}

KADRLNC_CONFIG = {
    'k'               : K,
    'f'               : F,
    'beta'            : BETA,
    'chunk_size_bytes': SHARD_SIZE,
    'shard_size_bytes': SHARD_SIZE,
    'p'               : P,
    'rlnc_delay_ms'   : RLNC_DELAY,
    'beta_rlnc'       : 2,
    'r'               : K // 2,
}

OPTIMUMP2P_CONFIG = {
    'k'               : K,
    'f'               : F,
    'beta'            : BETA,
    'chunk_size_bytes': SHARD_SIZE,
    'shard_size_bytes': SHARD_SIZE,
    'p'               : P,
    'rlnc_delay_ms'   : RLNC_DELAY,
    'beta_rlnc'       : 2,
    'r'               : K // 2,
}

# ===========================================================================
# Source node selection
# ===========================================================================

def get_source_node(block_idx: int, seed: int, rng: random.Random) -> int:
    """
    Return the source node index for a given block.
    'fixed'  → always node 0
    'random' → random node drawn from rng (reproducible per seed)
    """
    if SOURCE_MODE == 'fixed':
        return 0
    else:
        return rng.randint(0, N_NODES - 1)


# ===========================================================================
# Single protocol run
# ===========================================================================

def run_protocol(
    protocol_name : str,
    handler,
    config        : dict,
    seed          : int,
) -> None:
    """
    Run one protocol for one seed over N_BLOCKS blocks.

    Each block gets its own fresh SimPy environment and node states,
    but reuses the same topology (node IDs, routing tables, bandwidth tiers,
    mesh peers) built from the seed.

    Parameters
    ----------
    protocol_name : string label used in CSV filenames
    handler       : handle_message function from the protocol module
    config        : parameter dict passed into SimContext
    seed          : RNG seed for this run
    """
    print(f"  [{protocol_name}] seed={seed} ...", end=' ', flush=True)

    metrics   = MetricsCollector()
    rng_numpy = np.random.default_rng(seed)
    rng_src   = random.Random(seed)   # separate rng for source selection

    for block_idx in range(N_BLOCKS):
        # Fresh SimPy environment and node states for each block.
        ctx = build_context(
            n      = N_NODES,
            seed   = seed,
            metrics= metrics,
            rng    = rng_numpy,
            d_mesh = D_MESH,
            config = config,
        )

        source  = get_source_node(block_idx, seed, rng_src)
        block_id= block_idx

        # Synthetic PUBLISH message to seed the simulation.
        publish_msg = Message(
            msg_type  = 'PUBLISH',
            sender    = source,
            receiver  = source,
            block_id  = block_id,
            size_bytes= 0,
        )

        run_simulation(
            ctx             = ctx,
            source_node     = source,
            block_id        = block_id,
            protocol_handler= handler,
            publish_msg     = publish_msg,
        )

    # Flush all blocks' metrics to CSV after all blocks complete.
    metrics.flush(protocol_name, seed, RESULTS_DIR)
    print("done.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Running benchmarks: {N_SEEDS} seeds x {N_BLOCKS} blocks x {N_NODES} nodes")
    print(f"Source mode: {SOURCE_MODE}")
    print(f"Results → {RESULTS_DIR}/\n")

    protocols = [
        ('kadcast',    kadcast.handle_message,    KADCAST_CONFIG),
        ('kadrlnc',    kadrlnc.handle_message,    KADRLNC_CONFIG),
        ('optimump2p', optimump2p.handle_message, OPTIMUMP2P_CONFIG),
    ]

    for protocol_name, handler, config in protocols:
        print(f"[{protocol_name}]")
        for seed in range(SEED_START, SEED_START + N_SEEDS):
            run_protocol(protocol_name, handler, config, seed)
        print()

    print("All benchmarks complete.")


if __name__ == '__main__':
    main()
