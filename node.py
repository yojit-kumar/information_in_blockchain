"""
node.py — Per-(node, block) state for all three protocols.

One NodeState instance is created for each (node, block_id) pair when a node
first hears about a block. The simulation engine maintains a dict:
    states: dict[(node_index, block_id) -> NodeState]

Kept as a dataclass with __slots__ for memory efficiency across 1000 nodes
and multiple blocks in flight simultaneously.
"""

from dataclasses import dataclass, field
from typing import List, Set


@dataclass(slots=True)
class NodeState:
    # -----------------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------------
    node_id  : int
    block_id : int

    # -----------------------------------------------------------------------
    # Kadcast state
    # Tracks received FEC chunk indices. Decoded once len(kd_received) >= k.
    # -----------------------------------------------------------------------
    kd_received     : Set[int] = field(default_factory=set)
    kd_decoded      : bool     = False
    kd_forward_level: int      = -1   # bucket level of the sender who triggered decoding

    # -----------------------------------------------------------------------
    # KadRLNC state
    # rl_innovative: count of linearly independent (innovative) packets received.
    # rl_no_send: per-bucket set of nodes that replied IDONTWANT — skip these
    #             when choosing the next peer to send IHAVE in that bucket.
    # rl_publisher: True once decoded; node begins sending IHAVE to one new
    #               peer per bucket that is not in rl_no_send.
    # -----------------------------------------------------------------------
    rl_innovative : int       = 0
    rl_decoded    : bool      = False
    rl_no_send    : Set[int]  = field(default_factory=set)  # flat set of node indices to skip
    rl_publisher  : bool      = False
    rl_forward_level    : int  = -1     # bucket level of sender who triggered first IWANT
    rl_forwarding_started: bool = False  # True once r shards received and IHAVE cascade begun

    # -----------------------------------------------------------------------
    # OPTIMUMP2P state
    # op_shards: received shard indices (decoded once enough shards collected).
    # op_mesh_peers: D fixed peers assigned at node initialisation, same for
    #                all blocks — duplicated here for locality of access.
    # op_ihave_sent: peers already sent IHAVE for this block (avoid duplicates).
    # -----------------------------------------------------------------------
    op_shards     : Set[int]  = field(default_factory=set)
    op_decoded    : bool      = False
    op_mesh_peers : List[int] = field(default_factory=list)
    op_ihave_sent : Set[int]  = field(default_factory=set)  # peers already sent IHAVE for this block
    op_is_done    : Set[int]  = field(default_factory=set)  # peers who sent IDONTWANT (already decoded)
    op_forwarded : bool = False   # True once r=k/2 shards received and forwarding triggered
