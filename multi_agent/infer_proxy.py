# -*- coding: utf-8 -*-
"""
infer_proxy.py  --  Process-A side IPC proxy for the PnP_infer brain.

Replaces self.infer in pnp_agent_e2e.py setup() when USE_INFER_SERVER=1.
Sends car_data_raw / rsu_data_raw to process B (torch2/py3.10) via
zmq DEALER + msgpack_numpy, receives back VehicleControl dicts.

LIGHTWEIGHT: this module imports ONLY zmq, msgpack_numpy, numpy, carla.
DO NOT import torch or any perception/planning code here.

IPC contract (matching infer_server.py InferServer):
  send:  {'cmd': 'infer', 'frame_id': int, 'payload': <dict>}
  recv:  {'frame_id': int, 'latency_ms': float, 'latency_source': str,
          'measured_forward_ms': float,
          'result': {'ok': bool, 'controls': {str(idx): {throttle,steer,...}}}}

Serialization notes:
  - measurements dict contains only Python scalars + numpy arrays -> OK for msgpack_numpy
  - lidar is numpy Nx4 -> OK
  - drivable_area is numpy 192x96 -> OK
  - rgb_front/left/right/rear are numpy H x W x 3 -> OK (large but valid)
  - bev is a PIL Image -> EXCLUDED (not used by B inference path)
  - prev_control / save_path mimic the PnP_infer interface pnp_agent_e2e calls
"""
import os
import time

import numpy as np
import zmq
import msgpack
import msgpack_numpy as mnp
mnp.patch()

import carla  # real carla in process A


class InferProxy:
    """Synchronous IPC proxy: submit + blocking poll per inference call.

    pnp_agent_e2e.py accesses on self.infer:
      .get_action_from_list_inter(car_data_raw, rsu_data_raw, step, timestamp)
      .prev_control          (dict {idx: carla.VehicleControl})
      .save_path             (used by get_save_path(); set to None for IPC mode)
      ._l1_plan_stores       (None -> L1 disabled in smoke)
      .run_l1_step(...)      (only called when _l1_plan_stores is not None)
    """

    TIMEOUT_MS = 60000   # 60s: generous for first call which loads GPU memory

    def __init__(self, port, ego_num):
        import pathlib, datetime
        self.port = port
        self.ego_num = ego_num
        # save_path must be a real Path so leaderboard evaluator can create log/
        _save_root = os.environ.get('SAVE_PATH', '/data/jichengzhi_v2x/eval_smoke_saves')
        _tag = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.save_path = pathlib.Path(_save_root) / ('ipc_proxy_' + _tag)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self._l1_plan_stores = None   # L1 disabled in smoke
        self.prev_control = {i: carla.VehicleControl() for i in range(ego_num)}

        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect("tcp://127.0.0.1:%d" % port)
        self._frame_id = 0
        print("[InferProxy] connected to tcp://127.0.0.1:%d ego_num=%d" % (port, ego_num), flush=True)

        # Verify server is up
        self._ping()

    def _ping(self):
        self._sock.send(msgpack.packb({'cmd': 'ping'}, use_bin_type=True))
        if not self._sock.poll(5000):
            raise RuntimeError("[InferProxy] ping timeout — is process B running on port %d?" % self.port)
        self._sock.recv()
        print("[InferProxy] ping OK", flush=True)

    def _sanitize_data_elem(self, d):
        """Strip non-serializable fields from one car_data_raw element.
        PIL Image ('bev') cannot be msgpack-serialized; omit it.
        numpy arrays pass through msgpack_numpy cleanly.
        """
        if d is None:
            return None
        out = {}
        for k, v in d.items():
            if k == 'bev':
                continue  # PIL Image, not used by B inference path
            if isinstance(v, np.ndarray):
                out[k] = v
            elif isinstance(v, dict):
                # measurements dict: keep numpy arrays AS numpy (msgpack_numpy handles them);
                # only convert non-serializable types. lists/scalars pass through.
                inner = {}
                for mk, mv in v.items():
                    if isinstance(mv, np.ndarray):
                        inner[mk] = mv  # keep as ndarray for B-side torch.from_numpy()
                    elif isinstance(mv, list):
                        # lists of lists (camera matrices) -> numpy for msgpack_numpy
                        inner[mk] = mv
                    else:
                        inner[mk] = mv
                out[k] = inner
            else:
                out[k] = v
        return out

    def get_action_from_list_inter(self, car_data_raw, rsu_data_raw, step, timestamp):
        """Send sensor data to B, block until reply, return list of VehicleControl."""
        self._frame_id += 1
        fid = self._frame_id

        sanitized_car = [self._sanitize_data_elem(d) for d in car_data_raw]
        sanitized_rsu = [self._sanitize_data_elem(d) for d in rsu_data_raw]

        payload = {
            'method': 'get_action_from_list_inter',
            'args': {
                'car_data_raw': sanitized_car,
                'rsu_data_raw': sanitized_rsu,
                'step': step,
                'timestamp': float(timestamp),
            }
        }

        t_send = time.time()
        self._sock.send(msgpack.packb(
            {'cmd': 'infer', 'frame_id': fid, 'payload': payload},
            use_bin_type=True))

        timeout = self.TIMEOUT_MS if fid <= 2 else 15000
        if not self._sock.poll(timeout):
            raise RuntimeError("[InferProxy] get_action timeout frame_id=%d (waited %dms)" % (fid, timeout))

        raw = self._sock.recv()
        reply = msgpack.unpackb(raw, raw=False)
        rtt_ms = (time.time() - t_send) * 1000.0

        result = reply.get('result', {})
        lat_ms = reply.get('latency_ms', 0.0)
        print("[InferProxy] frame=%d latency_ms=%.1f rtt_ms=%.1f src=%s" % (
            fid, lat_ms, rtt_ms, reply.get('latency_source', '?')), flush=True)

        if not result.get('ok', False):
            err = result.get('error', 'unknown')
            print("[InferProxy] B returned error: %s" % err, flush=True)
            tb = result.get('traceback', '')
            print("[InferProxy] traceback: %s" % tb, flush=True)
            # Return prev_control as fallback (zero-order hold)
            return list(self.prev_control.values())

        controls_raw = result.get('controls', {})
        control_all = []
        for i in range(self.ego_num):
            cd = controls_raw.get(str(i))
            if cd is None:
                control_all.append(None)
            else:
                ctrl = carla.VehicleControl()
                ctrl.throttle = float(cd.get('throttle', 0.0))
                ctrl.steer    = float(cd.get('steer', 0.0))
                ctrl.brake    = float(cd.get('brake', 0.0))
                ctrl.hand_brake        = bool(cd.get('hand_brake', False))
                ctrl.reverse           = bool(cd.get('reverse', False))
                ctrl.manual_gear_shift = bool(cd.get('manual_gear_shift', False))
                ctrl.gear              = int(cd.get('gear', 1))
                self.prev_control[i] = ctrl
                control_all.append(ctrl)
        return control_all
