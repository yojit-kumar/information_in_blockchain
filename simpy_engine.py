"""
simpy_engine.py 

Responsibilities:
  - Message dataclass (shared across all protocols)
  - SimContext: shared simulation state passed to all protocol handlers
  - Uplink queue serialisation: one transmission at a time per node
  - Latency-delayed delivery: each message experiences independent LogNormal latency
  - Protocol dispatch: each protocol plugs in as a handle_message(msg, ctx) generator
  - Context construction and simulation entry point

"""

import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, List, Optional, Tuple

import network
from node import NodeState
from metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Message:
    msg_type  : str            # 'SHARD' | 'IHAVE' | 'IWANT' | 'IDONTWANT'
    sender    : int            # node index
    receiver  : int            # node index
    block_id  : int

    # Optional based on protocol
    chunk_id  : Optional[int]  = None   # Kadcast
    innovative: Optional[int]  = None   # KadRLNC
    shard_idx : Optional[int]  = None   # Optimump2p

    size_bytes: int            = 0


# ---------------------------------------------------------------------------
# SimContext
# ---------------------------------------------------------------------------

@dataclass
class SimContext:
    """
    Shared simulation state. One instance per protocol per benchmark run.
    Passed by reference to all protocol handlers and engine functions.
    """
    env          : simpy.Environment
    node_ids     : Dict[int, int]                      # node_index -> 16-bit ID
    kad_tables   : Dict[int, List[List[int]]]          # node_index -> buckets
    bw_tiers     : Dict[int, str]                      # node_index -> 'high'|'low'
    states       : Dict[Tuple[int, int], NodeState]    # (node, block_id) -> NodeState
    metrics      : MetricsCollector
    rng          : np.random.Generator
    uplink_queues: Dict[int, simpy.Resource]
    mesh_peers   : Dict[int, List[int]]                # node_index -> D mesh peers (OPTIMUMP2P)
    n_nodes      : int
    config       : Dict                                # protocol parameters from benchmark.py


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

def build_context(
    n        : int,
    seed     : int,
    metrics  : MetricsCollector,
    rng      : np.random.Generator,
    d_mesh   : int,
    config   : dict,
) -> SimContext:

    env       = simpy.Environment()
    node_ids  = network.assign_node_ids(n, seed)
    kad_tables= network.build_kademlia_tables(node_ids, seed)
    bw_tiers  = network.assign_bandwidth_tiers(n, seed)

    # Uplink queue: capacity=1 serialises outgoing transmissions per node.
    uplink_queues = {i: simpy.Resource(env, capacity=1) for i in range(n)}

    # OPTIMUMP2P mesh peers: D random peers per node, fixed for the whole run.
    import random
    mesh_rng = random.Random(seed)
    mesh_peers = {}
    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        mesh_peers[i] = mesh_rng.sample(candidates, min(d_mesh, len(candidates)))

    return SimContext(
        env           = env,
        node_ids      = node_ids,
        kad_tables    = kad_tables,
        bw_tiers      = bw_tiers,
        states        = {},
        metrics       = metrics,
        rng           = rng,
        uplink_queues = uplink_queues,
        mesh_peers    = mesh_peers,
        n_nodes       = n,
        config        = config,
    )


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def get_or_create_state(node_id: int, block_id: int, ctx: SimContext) -> NodeState:
    key = (node_id, block_id)
    if key not in ctx.states:
        ctx.states[key] = NodeState(
            node_id      = node_id,
            block_id     = block_id,
            op_mesh_peers= ctx.mesh_peers[node_id],
        )
    return ctx.states[key]


# ---------------------------------------------------------------------------
# Sending pipeline
# ---------------------------------------------------------------------------

def send_message(
    msg             : Message,
    ctx             : SimContext,
    protocol_handler: Callable,
) -> Generator:

    bw = network.BANDWIDTH_TIERS[ctx.bw_tiers[msg.sender]]

    with ctx.uplink_queues[msg.sender].request() as req:
        yield req
        # Transmission delay: time to push all bytes onto the wire.
        tx_delay = (msg.size_bytes / bw) * 1000.0  # convert to ms
        yield ctx.env.timeout(tx_delay)

    # Propagation delay: independent per packet.
    latency = network.sample_latency(ctx.rng)
    yield ctx.env.timeout(latency)

    # Deliver: spawn protocol handler as a new SimPy process.
    ctx.env.process(deliver_message(msg, ctx, protocol_handler))


def deliver_message(
    msg             : Message,
    ctx             : SimContext,
    protocol_handler: Callable,
) -> Generator:

    get_or_create_state(msg.receiver, msg.block_id, ctx)
    yield from protocol_handler(msg, ctx)


# ---------------------------------------------------------------------------
# Simulation entry point
# ---------------------------------------------------------------------------

def run_simulation(
    ctx             : SimContext,
    source_node     : int,
    block_id        : int,
    protocol_handler: Callable,
    publish_msg     : Message,
) -> None:

    # Initialise source node state.
    get_or_create_state(source_node, block_id, ctx)

    # Inject the publish event at t=0.
    ctx.env.process(deliver_message(publish_msg, ctx, protocol_handler))

    # Run until no events remain.
    ctx.env.run()
