"""
smoke_test.py — Quick end-to-end verification of the simulation.

Runs all three protocols with:
  - 50 nodes (instead of 1000)
  - 1 seed
  - 1 block
  - Results written to results/smoke/

Check that:
  1. No exceptions are raised.
  2. results/smoke/ contains 6 CSVs (2 per protocol).
  3. deliveries CSVs have close to 50 rows each (ideally all nodes decoded).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override benchmark config before importing main
import benchmark
benchmark.N_NODES    = 50
benchmark.N_SEEDS    = 1
benchmark.N_BLOCKS   = 1
benchmark.SEED_START = 0
benchmark.RESULTS_DIR= 'results/smoke'
benchmark.SOURCE_MODE= 'fixed'

benchmark.main()

# Quick sanity print
import csv
print("\n--- Delivery counts ---")
for protocol in ['kadcast', 'kadrlnc', 'optimump2p']:
    path = f"results/smoke/{protocol}_0_deliveries.csv"
    with open(path) as f:
        rows = list(csv.reader(f))
    print(f"  {protocol}: {len(rows)-1} nodes decoded (out of 50)")
