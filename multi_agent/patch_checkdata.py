#!/usr/bin/env python3
"""
Patch pnp_infer_action_e2e.py on H800:
Replace copy.deepcopy(i) in check_data with a safe copy that handles
torch Tensors (CUDA or non-leaf) that would otherwise fail deepcopy.
"""
import sys

path = "/data/jichengzhi_v2x/V2Xverse/simulation/leaderboard/team_code/pnp_infer_action_e2e.py"

with open(path, 'r') as f:
    content = f.read()

# Check the target line
target = 'data.append(self.preprocess_data(copy.deepcopy(i), car=car))'
if target not in content:
    print("ERROR: target line not found in file!")
    print("Searching for similar lines:")
    for ln, line in enumerate(content.split('\n'), 1):
        if 'deepcopy' in line:
            print(f"  line {ln}: {repr(line)}")
    sys.exit(1)

print(f"Found target: {repr(target)}")

# The _safe_copy_raw_elem helper to insert before check_data
# Tabs used to match class method indentation
helper_fn = '''
\tdef _safe_copy_raw_elem(self, elem):
\t\t"""Safe copy of a raw sensor dict for preprocess_data.
\t\tAvoids copy.deepcopy which fails on non-leaf CUDA Tensors (torch 2.x).
\t\tpreprocess_data only reads: measurements (dict of scalars/np.ndarray),
\t\tlidar (np.ndarray), rgb_* (np.ndarray). Copy those by value; skip
\t\ttorch Tensors that may have been added by perception_dataloader.
\t\t"""
\t\tif elem is None:
\t\t\treturn None
\t\tout = {}
\t\tfor k, v in elem.items():
\t\t\ttry:
\t\t\t\timport numpy as _np
\t\t\t\timport torch as _torch
\t\t\t\tif isinstance(v, _np.ndarray):
\t\t\t\t\tout[k] = _np.array(v)  # new numpy copy
\t\t\t\telif isinstance(v, _torch.Tensor):
\t\t\t\t\t# Skip: preprocess_data re-creates tensors from numpy
\t\t\t\t\tcontinue
\t\t\t\telif isinstance(v, dict):
\t\t\t\t\t# measurements dict: copy scalars and numpy arrays
\t\t\t\t\tinner = {}
\t\t\t\t\tfor mk, mv in v.items():
\t\t\t\t\t\tif isinstance(mv, _np.ndarray):
\t\t\t\t\t\t\tinner[mk] = _np.array(mv)
\t\t\t\t\t\telif isinstance(mv, _torch.Tensor):
\t\t\t\t\t\t\tcontinue  # skip CUDA tensors
\t\t\t\t\t\telse:
\t\t\t\t\t\t\tinner[mk] = mv  # scalar or list, safe
\t\t\t\t\tout[k] = inner
\t\t\t\telse:
\t\t\t\t\tout[k] = v  # int/float/str/list, no copy needed
\t\t\texcept Exception:
\t\t\t\tout[k] = v  # fallback: keep reference
\t\treturn out

'''

# Replace the deepcopy call with our safe copy
new_content = content.replace(
    target,
    'data.append(self.preprocess_data(self._safe_copy_raw_elem(i), car=car))'
)

# Verify replacement happened
if target in new_content:
    print("ERROR: replacement did not happen!")
    sys.exit(1)

# Insert the helper function before check_data
# Find "def check_data" with tab prefix
check_data_marker = '\tdef check_data(self, raw_data, car=True):'
if check_data_marker not in new_content:
    print("ERROR: check_data not found!")
    sys.exit(1)

new_content = new_content.replace(check_data_marker, helper_fn + check_data_marker)

with open(path, 'w') as f:
    f.write(new_content)

print("Patch written OK.")
print(f"Verifying...")

# Verify
with open(path, 'r') as f:
    verify = f.read()

assert '_safe_copy_raw_elem' in verify, "helper fn not in patched file"
assert 'copy.deepcopy(i)' not in verify, "deepcopy still present"
assert 'self._safe_copy_raw_elem(i)' in verify, "call not in patched file"
assert '\tdef check_data(self, raw_data, car=True):' in verify, "check_data not found"
print("Verify OK: deepcopy replaced with _safe_copy_raw_elem")
