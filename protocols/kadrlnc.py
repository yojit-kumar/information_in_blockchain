import random
from typing import Generator, Optional

from simpy_engine import Message, SimContext, send_message
from network import ID_BITS, NUM_BUCKETS, _bucket_level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_peer(node: int, block_id: int, bucket: int, ctx: SimContext) -> Optional[int]:
    state    = ctx.states[(node, block_id)]
    no_send  = state.rl_no_send
    for peer in ctx.kad_tables[node][bucket]:
        if peer not in no_send:
            return peer
    return None


def _send_ihave(sender: int, receiver: int, block_id: int, ctx: SimContext) -> None:
    msg = Message(
        msg_type   = 'IHAVE',
        sender     = sender,
        receiver   = receiver,
        block_id   = block_id,
        size_bytes = 64,   # control message — small fixed size
    )
    ctx.metrics.record_message(block_id, 'IHAVE', sender, receiver, ctx.env.now)
    ctx.env.process(send_message(msg, ctx, handle_message))


def _send_reply(msg_type: str, sender: int, receiver: int, block_id: int, ctx: SimContext) -> None:
    msg = Message(
        msg_type   = msg_type,
        sender     = sender,
        receiver   = receiver,
        block_id   = block_id,
        size_bytes = 64,
    )
    ctx.metrics.record_message(block_id, msg_type, sender, receiver, ctx.env.now)
    ctx.env.process(send_message(msg, ctx, handle_message))


def _publish(node: int, block_id: int, h: int, beta_rlnc: int, ctx: SimContext) -> None:
    rng_local = random.Random(ctx.rng.integers(0, 2**32).item())
    for bucket in range(h):   # only sub-buckets 0 to h-1
        peers = [p for p in ctx.kad_tables[node][bucket] 
                 if p not in ctx.states[(node, block_id)].rl_no_send]
        if not peers:
            continue
        targets = rng_local.sample(peers, min(beta_rlnc, len(peers)))
        for peer in targets:
            _send_ihave(node, peer, block_id, ctx)


# ---------------------------------------------------------------------------
# Protocol handler
# ---------------------------------------------------------------------------

def handle_message(msg: Message, ctx: SimContext) -> Generator:
    k             = ctx.config['k']
    shard_size    = ctx.config['shard_size_bytes']
    rlnc_delay_ms = ctx.config['rlnc_delay_ms']
    beta_rlnc     = ctx.config['beta_rlnc']
    r             = ctx.config['r']

    # -----------------------------------------------------------------------
    # PUBLISH — source node initiates broadcast
    # -----------------------------------------------------------------------
    if msg.msg_type == 'PUBLISH':
        state = ctx.states[(msg.receiver, msg.block_id)]
        # Source is trivially decoded.
        state.rl_innovative = k
        state.rl_decoded    = True
        state.rl_publisher  = True
        ctx.metrics.record_delivery(msg.block_id, msg.receiver, ctx.env.now)
        _publish(msg.receiver, msg.block_id, NUM_BUCKETS, beta_rlnc, ctx)
        return
        yield

    # -----------------------------------------------------------------------
    # IHAVE — decide whether to request shards or decline
    # -----------------------------------------------------------------------
    if msg.msg_type == 'IHAVE':
        state = ctx.states[(msg.receiver, msg.block_id)]
        if state.rl_decoded or state.rl_innovative >= k:
            # Already have the block — tell sender not to bother.
            _send_reply('IDONTWANT', msg.receiver, msg.sender, msg.block_id, ctx)
        else:
            _send_reply('IWANT', msg.receiver, msg.sender, msg.block_id, ctx)
        return
        yield

    # -----------------------------------------------------------------------
    # IWANT — send k shards to the requester
    # -----------------------------------------------------------------------
    if msg.msg_type == 'IWANT':
        for shard_idx in range(k):
            shard = Message(
                msg_type   = 'SHARD',
                sender     = msg.receiver,
                receiver   = msg.sender,
                block_id   = msg.block_id,
                shard_idx  = shard_idx,
                size_bytes = shard_size,
            )
            ctx.metrics.record_shard(msg.block_id)
            ctx.env.process(send_message(shard, ctx, handle_message))
        return
        yield

    # -----------------------------------------------------------------------
    # IDONTWANT — peer has decoded; mark no_send and try next peer in bucket
    # -----------------------------------------------------------------------
    if msg.msg_type == 'IDONTWANT':
        state = ctx.states[(msg.receiver, msg.block_id)]
        state.rl_no_send.add(msg.sender)

        # Determine which bucket the sender belongs to.
        bucket = _bucket_level(ctx.node_ids[msg.receiver] ^ ctx.node_ids[msg.sender])

        # Try the next available peer in that bucket.
        peer = _next_peer(msg.receiver, msg.block_id, bucket, ctx)
        if peer is not None:
            _send_ihave(msg.receiver, peer, msg.block_id, ctx)
        # If None: bucket exhausted, give up on this bucket.
        return
        yield

    # -----------------------------------------------------------------------
    # SHARD — accumulate; decode and become publisher once k shards received
    # -----------------------------------------------------------------------
    if msg.msg_type == 'SHARD':
        state = ctx.states[(msg.receiver, msg.block_id)]
        state.rl_innovative += 1

        # Record forward level from first shard's sender
        if state.rl_forward_level == -1:
            state.rl_forward_level = _bucket_level(
                ctx.node_ids[msg.sender] ^ ctx.node_ids[msg.receiver]
            )

        # Forwarding threshold r — trigger IHAVE cascade once
        if state.rl_innovative >= r and not state.rl_forwarding_started:
            state.rl_forwarding_started = True
            yield ctx.env.timeout(rlnc_delay_ms)   # recode delay
            if not state.rl_decoded:
                _publish(msg.receiver, msg.block_id, state.rl_forward_level, beta_rlnc, ctx)

        # Decode threshold k
        if state.rl_innovative >= k and not state.rl_decoded:
            yield ctx.env.timeout(rlnc_delay_ms)   # decode delay
            if not state.rl_decoded:
                state.rl_decoded   = True
                state.rl_publisher = True
                ctx.metrics.record_delivery(msg.block_id, msg.receiver, ctx.env.now)
