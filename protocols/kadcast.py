"""
kadcast.py — Kadcast broadcast protocol.

Algorithm summary:
  1. Source generates k*(1+f) FEC chunks (chunk_ids 0 to n_chunks-1).
  2. Source sends all chunks to β peers per bucket across all 16 bucket levels.
  3. Receiver accumulates chunks. On receiving a full set from one sender:
       - Decodes (marks kd_decoded = True) if not already decoded.
       - Records kd_forward_level = bucket_level(sender XOR receiver).
       - Forwards all chunks to β peers in each sub-bucket (levels 0 to h-1).
  4. Any chunks arriving after decoding are silently dropped.

Reference: Rohrer & Tschorsch, "Kadcast: A Structured Broadcast Protocol
for Blockchain Data Dissemination", 2019.

Parameters (from benchmark.py config):
  k       : number of chunks needed to decode
  f       : FEC overhead fraction (0.15 → 15% redundancy)
  beta    : number of peers selected per bucket for forwarding (3)
  chunk_size_bytes: size of each FEC chunk in bytes
"""

import random
from typing import Generator

from simpy_engine import Message, SimContext, send_message
from network import ID_BITS, _bucket_level


def _n_chunks(k: int, f: float) -> int:
    """Total number of FEC chunks generated: k*(1+f), rounded up."""
    return int(k * (1 + f))


def _send_all_chunks(
    sender   : int,
    receiver : int,
    block_id : int,
    n_chunks : int,
    chunk_size: int,
    ctx      : SimContext,
) -> None:
    """Spawn send_message processes for all chunks from sender to receiver."""
    for chunk_id in range(n_chunks):
        msg = Message(
            msg_type   = 'SHARD',
            sender     = sender,
            receiver   = receiver,
            block_id   = block_id,
            chunk_id   = chunk_id,
            size_bytes = chunk_size,
        )
        ctx.metrics.record_shard(block_id)
        ctx.env.process(send_message(msg, ctx, handle_message))


def _forward(
    node     : int,
    block_id : int,
    h        : int,
    n_chunks : int,
    chunk_size: int,
    beta     : int,
    ctx      : SimContext,
) -> None:
    """
    Forward all chunks to β peers in each sub-bucket (levels 0 to h-1).
    Peers are sampled without replacement from the bucket; if a bucket has
    fewer than β peers, all peers in that bucket are used.
    """
    rng_local = random.Random(ctx.rng.integers(0, 2**32).item())
    for level in range(h):
        bucket = ctx.kad_tables[node][level]
        if not bucket:
            continue
        targets = rng_local.sample(bucket, min(beta, len(bucket)))
        for peer in targets:
            _send_all_chunks(node, peer, block_id, n_chunks, chunk_size, ctx)


def handle_message(msg: Message, ctx: SimContext) -> Generator:
    """
    Protocol handler called by the engine on message delivery.

    Handles:
      'PUBLISH' — source node initiates broadcast.
      'SHARD'   — receiver accumulates chunks, decodes, and forwards.
    """
    # Retrieve config stored in ctx by benchmark.py.
    k          = ctx.config['k']
    f          = ctx.config['f']
    beta       = ctx.config['beta']
    chunk_size = ctx.config['chunk_size_bytes']
    n_chunks   = _n_chunks(k, f)

    state = ctx.states[(msg.receiver, msg.block_id)]

    if msg.msg_type == 'PUBLISH':
        # Source node: send all chunks to β peers in every bucket.
        _forward(
            node       = msg.receiver,
            block_id   = msg.block_id,
            h          = ID_BITS,      # source uses all 16 levels
            n_chunks   = n_chunks,
            chunk_size = chunk_size,
            beta       = beta,
            ctx        = ctx,
        )
        # Mark source as decoded immediately — it created the block.
        state.kd_decoded      = True
        state.kd_forward_level= ID_BITS
        ctx.metrics.record_delivery(msg.block_id, msg.receiver, ctx.env.now)

    elif msg.msg_type == 'SHARD':
        # Already decoded: drop silently.
        if state.kd_decoded:
            return
            yield  # make this a generator

        # Accumulate chunk.
        state.kd_received.add(msg.chunk_id)

        # Check if we now have enough to decode.
        if len(state.kd_received) >= k:
            state.kd_decoded = True

            # Determine forwarding level from this sender's bucket level.
            h = _bucket_level(
                ctx.node_ids[msg.sender] ^ ctx.node_ids[msg.receiver]
            )
            state.kd_forward_level = h

            ctx.metrics.record_delivery(msg.block_id, msg.receiver, ctx.env.now)

            # Forward to sub-buckets only if h > 0.
            if h > 0:
                _forward(
                    node       = msg.receiver,
                    block_id   = msg.block_id,
                    h          = h,
                    n_chunks   = n_chunks,
                    chunk_size = chunk_size,
                    beta       = beta,
                    ctx        = ctx,
                )

    yield ctx.env.timeout(0)  # no-op yield to satisfy SimPy generator requirement
