#!/usr/bin/env python3
"""Fix process_b_server.py: remove broken debug prints and add correct ones."""
path = "/data/jichengzhi_v2x/V2Xverse/simulation/leaderboard/team_code/closedloop/process_b_server.py"

with open(path) as f:
    lines = f.readlines()

# Find and remove all broken [B-DEBUG] lines
new_lines = []
skip_next = False
for ln, line in enumerate(lines):
    stripped = line.strip()
    # Remove broken debug lines (they contain print( with unquoted strings)
    if '[B-DEBUG]' in line:
        continue
    # Fix: remove the "if car_data_raw and car_data_raw[0] is not None:" block that wraps debug
    # but only if the debug lines were removed (the if block becomes empty)
    new_lines.append(line)

content = "".join(new_lines)

# Now add clean debug print BEFORE the controls = pnp.get_action_from_list_inter call
old = "                controls = pnp.get_action_from_list_inter("
new = (
    "                if car_data_raw and car_data_raw[0] is not None:\n"
    "                    _elem0 = car_data_raw[0]\n"
    "                    _keys = list(_elem0.keys()) if hasattr(_elem0, 'keys') else type(_elem0)\n"
    "                    print('[B-DEBUG] car_data_raw[0] keys: ' + str(_keys), flush=True)\n"
    "                    if hasattr(_elem0, 'get') and 'measurements' in _elem0:\n"
    "                        _mkeys = list(_elem0['measurements'].keys()) if hasattr(_elem0['measurements'], 'keys') else type(_elem0['measurements'])\n"
    "                        print('[B-DEBUG] measurements keys: ' + str(_mkeys), flush=True)\n"
    "                        print('[B-DEBUG] x=' + str(_elem0['measurements'].get('x', 'MISSING')), flush=True)\n"
    "                    else:\n"
    "                        print('[B-DEBUG] NO measurements key in elem0', flush=True)\n"
    "                controls = pnp.get_action_from_list_inter("
)

if old not in content:
    print("ERROR: target line not found")
    for i, line in enumerate(content.split('\n')):
        if 'pnp.get_action_from_list_inter' in line:
            print(f"  line {i}: {repr(line)}")
    import sys; sys.exit(1)

content2 = content.replace(old, new)

with open(path, "w") as f:
    f.write(content2)

# Verify syntax
import py_compile, sys
try:
    py_compile.compile(path, doraise=True)
    print("syntax OK")
except py_compile.PyCompileError as e:
    print("SYNTAX ERROR:", e)
    sys.exit(1)

print("debug patch OK")
