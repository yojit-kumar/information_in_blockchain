import os
import sys
import random
import numpy as np
import simpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simpy_engine import build_context, Message, SimContext, get_or_create_state
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

# --- Timing ---
W        = 2.0    # measurement window in seconds
COOLDOWN = 1.0    # cooldown in seconds (injection continues, not tracked)

# --- Publish rates to sweep (msg/s) ---
PUBLISH_RATES = [1, 2, 4, 8, 16, 32]

# --- Shared protocol params ---
K          = 32
BLOCK_SIZE = 1024 * 1024   # 1 MB
SHARD_SIZE = BLOCK_SIZE // K
F          = 0.15
BETA       = 3
RLNC_DELAY = 3.0
P          = 8
D_MESH     = 4

CONFIG = {
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

# --- Results ---
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/rate"))

# ===========================================================================
# Tracked metrics collector — wraps MetricsCollector to filter untracked blocks
# ===========================================================================

class RateMetricsCollector(MetricsCollector):
    def __init__(self):
        super().__init__()
        self.tracked_blocks  = set()   # block_ids injected during W
        self.inject_times    = {}      # block_id -> inject_time_ms

    def register_block(self, block_id: int, inject_time_ms: float, tracked: bool):
        self.inject_times[block_id] = inject_time_ms
        if tracked:
            self.tracked_blocks.add(block_id)

    def record_delivery(self, block_id: int, node_id: int, time_ms: float):
        if block_id in self.tracked_blocks:
            super().record_delivery(block_id, node_id, time_ms)

    def record_message(self, block_id, msg_type, sender, receiver, time_ms):
        if block_id in self.tracked_blocks:
            super().record_message(block_id, msg_type, sender, receiver, time_ms)

    def record_shard(self, block_id: int) -> None:
        if block_id in self.tracked_blocks:
            super().record_shard(block_id)

    def flush(self, protocol: str, seed: int, rate: int, results_dir: str):
        """
        Write CSVs with inject_time_ms column added to deliveries.
        """
        import csv
        os.makedirs(results_dir, exist_ok=True)

        # Deliveries CSV — add inject_time_ms column
        delivery_path = os.path.join(results_dir, f"{protocol}_{seed}_{rate}_deliveries.csv")
        with open(delivery_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['block_id', 'node_id', 'time_ms', 'inject_time_ms'])
            for e in self._deliveries:
                inject_t = self.inject_times.get(e.block_id, 0.0)
                writer.writerow([e.block_id, e.node_id, e.time_ms, inject_t])

        # Control messages CSV
        control_path = os.path.join(results_dir, f"{protocol}_{seed}_{rate}_messages.csv")
        with open(control_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['block_id', 'msg_type', 'sender', 'receiver', 'time_ms'])
            for e in self._controls:
                writer.writerow([e.block_id, e.msg_type, e.sender, e.receiver, e.time_ms])

        shard_path = os.path.join(results_dir, f"{protocol}_{seed}_{rate}_shards.csv")
        with open(shard_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['block_id', 'shard_count'])
            for block_id, count in self._shard_counts.items():
                writer.writerow([block_id, count])
        

        self._shard_counts.clear()
        self._deliveries.clear()
        self._controls.clear()


# ===========================================================================
# Block injection process
# ===========================================================================

def inject_blocks(
    env          : simpy.Environment,
    ctx          : SimContext,
    handler,
    metrics      : RateMetricsCollector,
    rate         : float,
    w_ms         : float,
    total_ms     : float,
    rng_src      : random.Random,
    block_counter: list,   # mutable counter [int]
):

    interval_ms = 1000.0 / rate

    while env.now < total_ms:
        block_id     = block_counter[0]
        block_counter[0] += 1
        inject_time  = env.now
        tracked      = inject_time < w_ms

        metrics.register_block(block_id, inject_time, tracked)

        source = rng_src.randint(0, N_NODES - 1)
        get_or_create_state(source, block_id, ctx)

        publish_msg = Message(
            msg_type  = 'PUBLISH',
            sender    = source,
            receiver  = source,
            block_id  = block_id,
            size_bytes= 0,
        )
        env.process(_deliver_publish(publish_msg, ctx, handler))

        yield env.timeout(interval_ms)


def _deliver_publish(msg: Message, ctx: SimContext, handler) -> object:
    get_or_create_state(msg.receiver, msg.block_id, ctx)
    yield from handler(msg, ctx)


# ===========================================================================
# Single protocol run at one rate
# ===========================================================================

def run_rate_protocol(
    protocol_name: str,
    handler,
    config       : dict,
    seed         : int,
    rate         : float,
):
    print(f"  [{protocol_name}] seed={seed} rate={rate} msg/s ...", end=' ', flush=True)

    w_ms     = W * 1000.0
    total_ms = (W + COOLDOWN) * 1000.0

    metrics   = RateMetricsCollector()
    rng_numpy = np.random.default_rng(seed)
    rng_src   = random.Random(seed)

    # Single shared SimContext for all concurrent blocks.
    ctx = build_context(
        n      = N_NODES,
        seed   = seed,
        metrics= metrics,
        rng    = rng_numpy,
        d_mesh = D_MESH,
        config = config,
    )

    block_counter = [0]   # mutable so inject_blocks can increment it

    ctx.env.process(inject_blocks(
        env           = ctx.env,
        ctx           = ctx,
        handler       = handler,
        metrics       = metrics,
        rate          = rate,
        w_ms          = w_ms,
        total_ms      = total_ms,
        rng_src       = rng_src,
        block_counter = block_counter,
    ))

    ctx.env.run()

    metrics.flush(protocol_name, seed, int(rate), RESULTS_DIR)
    print("done.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Rate benchmark: {N_SEEDS} seeds x {len(PUBLISH_RATES)} rates x {N_NODES} nodes")
    print(f"Window={W}s  Cooldown={COOLDOWN}s")
    print(f"Results → {RESULTS_DIR}/\n")

    protocols = [
        ('kadcast',    kadcast.handle_message,    CONFIG),
        ('kadrlnc',    kadrlnc.handle_message,    CONFIG),
        ('optimump2p', optimump2p.handle_message, CONFIG),
    ]

    for protocol_name, handler, config in protocols:
        print(f"[{protocol_name}]")
        for rate in PUBLISH_RATES:
            for seed in range(SEED_START, SEED_START + N_SEEDS):
                run_rate_protocol(protocol_name, handler, config, seed, rate)
        print()

    print("All rate benchmarks complete.")


if __name__ == '__main__':
    main()
