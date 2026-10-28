import json
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from aeqga_ibm_noise import SnapshotBackend

b = SnapshotBackend('marrakesh', 156)
adj = {q: set() for q in range(156)}
for a, c in b.target['cz']:
    adj[a].add(c); adj[c].add(a)
blocks = {frozenset([q]) for q in adj}
for _ in range(3):
    blocks = {s | {v} for s in blocks for u in s for v in adj[u] - s}
blocks = sorted(tuple(sorted(s)) for s in blocks)
row, col = [], []
for j, block in enumerate(blocks):
    row.extend(block); col.extend([j]*4)
a = coo_matrix((np.ones(len(row)), (row, col)), shape=(156,len(blocks))).tocsc()
result = milp(-np.ones(len(blocks)), integrality=np.ones(len(blocks)),
              bounds=Bounds(0,1), constraints=LinearConstraint(a, 0, 1),
              options={'time_limit':120, 'mip_rel_gap':0})
selected = [] if result.x is None else [list(blocks[j]) for j in np.flatnonzero(result.x > .5)]
report = {'width':4, 'enumerated_connected_blocks':len(blocks),
          'status':int(result.status), 'success':bool(result.success), 'message':result.message,
          'maximum_blocks':len(selected), 'used_qubits':4*len(selected),
          'mip_gap':getattr(result,'mip_gap',None), 'placements':selected}
(Path(__file__).resolve().parent/'marrakesh_4q_layout.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
