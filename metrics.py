"""
metrics.py — In-memory metrics collection for broadcast protocol simulation.

Two event types are recorded:
  - Delivery events: when a node successfully decodes a block.
  - Control message events: IHAVE, IWANT, IDONTWANT exchanges.
"""

import csv
import os
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Raw event records (lightweight named tuples via dataclass)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DeliveryEvent:
    block_id : int
    node_id  : int
    time_ms  : float


@dataclass(slots=True)
class ControlEvent:
    block_id : int
    msg_type : str   # 'IHAVE' | 'IWANT' | 'IDONTWANT'
    sender   : int
    receiver : int
    time_ms  : float


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

@dataclass
class MetricsCollector:

    _deliveries : List[DeliveryEvent] = field(default_factory=list, repr=False)
    _controls   : List[ControlEvent]  = field(default_factory=list, repr=False)
    _shard_counts : dict = field(default_factory=dict)

    def record_delivery(self, block_id: int, node_id: int, time_ms: float) -> None:
        self._deliveries.append(DeliveryEvent(block_id, node_id, time_ms))

    def record_message(
        self,
        block_id : int,
        msg_type : str,
        sender   : int,
        receiver : int,
        time_ms  : float,
    ) -> None:
        self._controls.append(ControlEvent(block_id, msg_type, sender, receiver, time_ms))

    def flush(self, protocol: str, seed: int, results_dir: str, label: str = '') -> None:
        os.makedirs(results_dir, exist_ok=True)

        # Deliveries CSV
        delivery_path = os.path.join(results_dir, f"{protocol}_{seed}{'_'+label if label else ''}_deliveries.csv")
        with open(delivery_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['block_id', 'node_id', 'time_ms'])
            for e in self._deliveries:
                writer.writerow([e.block_id, e.node_id, e.time_ms])

        # Control messages CSV
        control_path = os.path.join(results_dir, f"{protocol}_{seed}{'_'+label if label else ''}_messages.csv")
        with open(control_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['block_id', 'msg_type', 'sender', 'receiver', 'time_ms'])
            for e in self._controls:
                writer.writerow([e.block_id, e.msg_type, e.sender, e.receiver, e.time_ms])
        
        shard_path = os.path.join(results_dir, f"{protocol}_{seed}{'_'+label if label else ''}_shards.csv")
        with open(shard_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['block_id', 'shard_count'])
            for block_id, count in self._shard_counts.items():
                writer.writerow([block_id, count])
            
        # Clear buffers
        self._shard_counts.clear()
        self._deliveries.clear()
        self._controls.clear()

    def record_shard(self, block_id: int) -> None:
        self._shard_counts[block_id] = self._shard_counts.get(block_id, 0) + 1
