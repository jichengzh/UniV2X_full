#!/usr/bin/env python3
"""Rewrite the broken try block in process_b_server.py infer_fn."""
import py_compile, sys

path = "/data/jichengzhi_v2x/V2Xverse/simulation/leaderboard/team_code/closedloop/process_b_server.py"

with open(path) as f:
    content = f.read()

# Find the broken section: from "            try:" to "return {'ok': True, 'controls': result}"
# We know the structure:
OLD_BLOCK = """            try:
                if car_data_raw and car_data_raw[0] is not None:
                    elem0 = car_data_raw[0]
                    if measurements in elem0:
                    else:
                if car_data_raw and car_data_raw[0] is not None:
                    _elem0 = car_data_raw[0]
                    _keys = list(_elem0.keys()) if hasattr(_elem0, 'keys') else type(_elem0)
                    print('[B-DEBUG] car_data_raw[0] keys: ' + str(_keys), flush=True)
                    if hasattr(_elem0, 'get') and 'measurements' in _elem0:
                        _mkeys = list(_elem0['measurements'].keys()) if hasattr(_elem0['measurements'], 'keys') else type(_elem0['measurements'])
                        print('[B-DEBUG] measurements keys: ' + str(_mkeys), flush=True)
                        print('[B-DEBUG] x=' + str(_elem0['measurements'].get('x', 'MISSING')), flush=True)
                    else:
                        print('[B-DEBUG] NO measurements key in elem0', flush=True)
                controls = pnp.get_action_from_list_inter(
                    car_data_raw, rsu_data_raw, step, timestamp)
                # controls is dict {vehicle_idx: carla.VehicleControl (shim)}
                result = {str(k): _control_to_dict(v) for k, v in controls.items()}
                return {'ok': True, 'controls': result}"""

NEW_BLOCK = """            try:
                # Debug: log incoming car_data_raw structure on first call
                if car_data_raw and car_data_raw[0] is not None:
                    _elem0 = car_data_raw[0]
                    _keys = list(_elem0.keys()) if hasattr(_elem0, 'keys') else str(type(_elem0))
                    print('[B-DEBUG] car_data_raw[0] keys: ' + str(_keys), flush=True)
                    if hasattr(_elem0, 'get') and 'measurements' in _elem0:
                        _mes = _elem0['measurements']
                        _mkeys = list(_mes.keys()) if hasattr(_mes, 'keys') else str(type(_mes))
                        print('[B-DEBUG] measurements keys: ' + str(_mkeys), flush=True)
                        print('[B-DEBUG] x=' + str(_mes.get('x', 'MISSING')), flush=True)
                    else:
                        print('[B-DEBUG] NO measurements key in car_data_raw[0]', flush=True)
                controls = pnp.get_action_from_list_inter(
                    car_data_raw, rsu_data_raw, step, timestamp)
                # controls is list [carla.VehicleControl, ...]
                result = {str(k): _control_to_dict(v) for k, v in enumerate(controls)}
                return {'ok': True, 'controls': result}"""

if OLD_BLOCK not in content:
    print("ERROR: OLD_BLOCK not found, showing context:")
    idx = content.find("controls = pnp.get_action_from_list_inter")
    print(repr(content[max(0,idx-600):idx+200]))
    sys.exit(1)

content2 = content.replace(OLD_BLOCK, NEW_BLOCK)

with open(path, "w") as f:
    f.write(content2)

try:
    py_compile.compile(path, doraise=True)
    print("syntax OK")
except py_compile.PyCompileError as e:
    print("SYNTAX ERROR:", e)
    sys.exit(1)

print("fix applied OK")
