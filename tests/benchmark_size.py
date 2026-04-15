import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simpy_engine import build_context, run_simulation, Message
from metrics import MetricsCollector
import protocols.kadcast    as kadcast
import protocols.kadrlnc    as kadrlnc
import protocols.optimump2p as optimump2p

# ===========================================================================
# CONFIG
# ===========================================================================

# --- Network ---
N_NODES    = 1000
SEED_START = 0
N_SEEDS    = 10

# --- Block sizes to sweep ---
BLOCK_SIZES = [
    128  * 1024,    # 128 KB
    512  * 1024,    # 512 KB
    1024 * 1024,    #   1 MB
    4096 * 1024,    #   4 MB
]

# --- Fixed protocol params ---
K          = 32
F          = 0.15
BETA       = 3
RLNC_DELAY = 3.0
P          = 8
D_MESH     = 4
N_BLOCKS   = 5

# --- Source mode ---
SOURCE_MODE = 'random'   # 'fixed' or 'random'

# --- Results ---
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/size"))

# ===========================================================================
# Build config dict for a given block size
# ===========================================================================

def make_config(block_size: int) -> dict:
    shard_size = block_size // K
    return {
        'k'               : K,
        'f'               : F,
        'beta'            : BETA,
        'chunk_size_bytes': shard_size,
        'shard_size_bytes': shard_size,
        'p'               : P,
        'rlnc_delay_ms'   : RLNC_DELAY,
        'beta_rlnc'       : 2,
        'r'               : K // 2,

    }


# ===========================================================================
# Source node selection
# ===========================================================================

def get_source_node(rng: random.Random) -> int:
    if SOURCE_MODE == 'fixed':
        return 0
    return rng.randint(0, N_NODES - 1)


# ===========================================================================
# Single protocol run at one block size
# ===========================================================================

def run_size_protocol(
    protocol_name: str,
    handler,
    block_size   : int,
    seed         : int,
):
    size_kb = block_size // 1024
    print(f"  [{protocol_name}] seed={seed} size={size_kb}KB ...", end=' ', flush=True)

    config    = make_config(block_size)
    metrics   = MetricsCollector()
    rng_numpy = np.random.default_rng(seed)
    rng_src   = random.Random(seed)

    for block_idx in range(N_BLOCKS):
        ctx = build_context(
            n      = N_NODES,
            seed   = seed,
            metrics= metrics,
            rng    = rng_numpy,
            d_mesh = D_MESH,
            config = config,
        )

        source   = get_source_node(rng_src)
        block_id = block_idx

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

    label = f"{size_kb}kb"
    metrics.flush(protocol_name, seed, RESULTS_DIR, label)
    print("done.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sizes_str = ', '.join(f"{s//1024}KB" for s in BLOCK_SIZES)
    print(f"Size benchmark: {N_SEEDS} seeds x {len(BLOCK_SIZES)} sizes x {N_NODES} nodes")
    print(f"Block sizes: {sizes_str}")
    print(f"Results → {RESULTS_DIR}/\n")

    protocols = [
        ('kadcast',    kadcast.handle_message),
        ('kadrlnc',    kadrlnc.handle_message),
        ('optimump2p', optimump2p.handle_message),
    ]

    for protocol_name, handler in protocols:
        print(f"[{protocol_name}]")
        for block_size in BLOCK_SIZES:
            for seed in range(SEED_START, SEED_START + N_SEEDS):
                run_size_protocol(protocol_name, handler, block_size, seed)
        print()

    print("All size benchmarks complete.")


if __name__ == '__main__':
    main()
