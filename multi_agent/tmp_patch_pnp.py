#!/usr/bin/env python3
"""Patch pnp_agent_e2e.py: add USE_INFER_SERVER gate in setup()."""
path = "/home/jichengzhi/V2Xverse/simulation/leaderboard/team_code/pnp_agent_e2e.py"
with open(path) as f:
    lines = f.readlines()

# Find start and end line indices (0-based)
# Start: first line containing "perception_dataloader = build_dataset(perception_hypes"
# End: first line after start containing "device=device)" after the PnP_infer block
start_line = None
end_line = None
for i, line in enumerate(lines):
    if start_line is None and "perception_dataloader = build_dataset(perception_hypes" in line:
        start_line = i
    if start_line is not None and "device=device)" in line and i > start_line + 20:
        end_line = i
        break

print(f"Replacing lines {start_line+1}..{end_line+1}")
print(repr(lines[start_line][:80]))
print(repr(lines[end_line][:80]))

# Build replacement block with correct 8-space indentation
new_block = (
    "        # *** USE_INFER_SERVER gate: when set, skip local model loading\n"
    "        # and connect self.infer to process B (torch2/py3.10) IPC proxy.\n"
    "        if int(os.environ.get('USE_INFER_SERVER', '0')):\n"
    "            _port = int(os.environ.get('INFER_SERVER_PORT', '5557'))\n"
    "            print('[pnp_agent] USE_INFER_SERVER=1 -> InferProxy port=%d' % _port, flush=True)\n"
    "            from team_code.closedloop.infer_proxy import InferProxy\n"
    "            self.infer = InferProxy(port=_port, ego_num=self.ego_vehicles_num)\n"
    "            self.perception_model = None\n"
    "            self.planning_model   = None\n"
    "        else:\n"
    "            perception_dataloader = build_dataset(perception_hypes, visualize=True, train=False)\n"
    "            print('Creating perception Model')\n"
    "            self.perception_model = create_perception_model(perception_hypes)\n"
    "            print('Loading perception Model from checkpoint')\n"
    "            resume_epoch, self.perception_model = load_perception_model(self.config['perception']['perception_model_dir'], self.perception_model)\n"
    "            print(f'resume from {resume_epoch} epoch.')\n"
    "            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "            self.perception_model.to(device)\n"
    "            self.perception_model.eval()\n"
    "\n"
    "            # load planning model\n"
    "            planner_config = load_config_from_yaml(self.config['planning']['planner_config'])\n"
    "            planning_model_config = planner_config['model']\n"
    "            print('Creating planning Model')\n"
    "            planning_model = build_planning_model(\n"
    "                CODRIVING_REGISTRY,\n"
    "                planning_model_config,\n"
    "            )\n"
    "            print('Loading planning Model from checkpoint')\n"
    "            load_planning_model_checkpoint(self.config['planning']['planner_model_checkpoint'], device, planning_model)\n"
    "            model_decoration_config = planner_config['model_decoration']\n"
    "            decorate_model(planning_model, **model_decoration_config)\n"
    "            planning_model.to(device)\n"
    "            planning_model.eval()\n"
    "            self.planning_model = planning_model\n"
    "\n"
    "            # core module, infer the action from sensor data\n"
    "            if self.config['planning']['core_method'] == 'MotionNet':\n"
    "                self.infer = PnP_infer(config=self.config,\n"
    "                                    ego_vehicles_num=self.ego_vehicles_num,\n"
    "                                    perception_model=self.perception_model,\n"
    "                                    planning_model=planning_model,\n"
    "                                    perception_dataloader=perception_dataloader,\n"
    "                                    device=device)\n"
)

new_lines = lines[:start_line] + [new_block] + lines[end_line+1:]
with open(path, "w") as f:
    f.writelines(new_lines)
print(f"patch written OK — {len(new_lines)} total lines")

# Verify
with open(path) as f:
    content = f.read()
assert "USE_INFER_SERVER" in content, "patch verify FAILED"
assert "InferProxy" in content, "InferProxy not found"
assert "build_dataset" in content, "original path missing"
print("verify OK")
