import random
from typing import Generator

from simpy_engine import Message, SimContext, send_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_k_shards(k: int, p: int, rng_local: random.Random) -> list:
    return rng_local.sample(range(k * p), k)


def _send_shards(
    sender    : int,
    receiver  : int,
    block_id  : int,
    shard_idxs: list,
    shard_size: int,
    ctx       : SimContext,
) -> None:
    for shard_idx in shard_idxs:
        msg = Message(
            msg_type   = 'SHARD',
            sender     = sender,
            receiver   = receiver,
            block_id   = block_id,
            shard_idx  = shard_idx,
            size_bytes = shard_size,
        )
        ctx.metrics.record_shard(block_id)
        ctx.env.process(send_message(msg, ctx, handle_message))


def _send_idontwant(sender: int, receiver: int, block_id: int, ctx: SimContext) -> None:
    msg = Message(
        msg_type   = 'IDONTWANT',
        sender     = sender,
        receiver   = receiver,
        block_id   = block_id,
        size_bytes = 64,
    )
    ctx.metrics.record_message(block_id, 'IDONTWANT', sender, receiver, ctx.env.now)
    ctx.env.process(send_message(msg, ctx, handle_message))


# ---------------------------------------------------------------------------
# Protocol handler
# ---------------------------------------------------------------------------

def handle_message(msg: Message, ctx: SimContext) -> Generator:
    k          = ctx.config['k']
    p          = ctx.config['p']
    shard_size = ctx.config['shard_size_bytes']
    rlnc_delay_ms = ctx.config['rlnc_delay_ms']

    rng_local  = random.Random(ctx.rng.integers(0, 2**32).item())

    # -----------------------------------------------------------------------
    # PUBLISH — source node initiates broadcast
    # -----------------------------------------------------------------------
    if msg.msg_type == 'PUBLISH':
        state             = ctx.states[(msg.receiver, msg.block_id)]
        state.op_decoded  = True
        ctx.metrics.record_delivery(msg.block_id, msg.receiver, ctx.env.now)

        # Publisher flooding: send k random shards to every mesh peer.
        for peer in state.op_mesh_peers:
            shards = _random_k_shards(k, p, rng_local)
            _send_shards(msg.receiver, peer, msg.block_id, shards, shard_size, ctx)

        return
        yield

    # -----------------------------------------------------------------------
    # SHARD — accumulate; decode and forward once k shards received
    # -----------------------------------------------------------------------
    if msg.msg_type == 'SHARD':
        state = ctx.states[(msg.receiver, msg.block_id)]

        # Drop if already decoded.
        if state.op_decoded:
            return
            yield

        state.op_shards.add(msg.shard_idx)

        # Forwarding threshold r = k//2 — trigger once
        if len(state.op_shards) >= k // 2 and not state.op_forwarded:
            state.op_forwarded = True
            yield ctx.env.timeout(rlnc_delay_ms)
            for peer in state.op_mesh_peers:
                if peer != msg.sender and peer not in state.op_is_done:
                    shards = _random_k_shards(k, p, rng_local)
                    _send_shards(msg.receiver, peer, msg.block_id, shards, shard_size, ctx)

        # Decode threshold k
        if len(state.op_shards) >= k and not state.op_decoded:
            state.op_decoded = True
            yield ctx.env.timeout(rlnc_delay_ms)
            ctx.metrics.record_delivery(msg.block_id, msg.receiver, ctx.env.now)
            for peer in state.op_mesh_peers:
                if peer not in state.op_is_done:
                    _send_idontwant(msg.receiver, peer, msg.block_id, ctx)

     
    # -----------------------------------------------------------------------
    # IDONTWANT — peer has decoded; add to op_is_done
    # -----------------------------------------------------------------------
    elif msg.msg_type == 'IDONTWANT':
        state = ctx.states[(msg.receiver, msg.block_id)]
        state.op_is_done.add(msg.sender)

    yield ctx.env.timeout(0)  # no-op yield to satisfy SimPy generator requirement
