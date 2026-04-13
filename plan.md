# UniV2X 后续优化计划

**最后更新**：2026-04-09  
**早期 TRT 迁移计划**：见 `plan_trt_migration.md`

---

## 当前状态

| 指标 | PyTorch 基准 | 当前最优（全链路 TRT） |
|------|:----------:|:------------------:|
| AMOTA | 0.338 | **0.341** |
| V2X 延迟（N_inf=50） | ~5,640 ms | **~90 ms（63× 加速）** |
| 下游头延迟 | ~100 ms | **32.5 ms（INT8）** |
| BEV 引擎大小（ego） | — | 75 MB (FP16) / 43 MB (INT8) |
| 检测头引擎大小（ego） | — | 33.4 MB (FP16) / 18.2 MB (INT8) |
| 下游头引擎大小（ego） | — | 152 MB (FP16) / 74 MB (INT8) |

**推荐配置**：
- **最优精度**：Hook A(FP16)+B+C+D(FP16)+E = 全链路 FP16 TRT（AMOTA 0.345）
- **体积优先**：Hook A(FP16)+B+C+D(INT8) = FP16 BEV + INT8 检测头（AMOTA 0.332，检测头 18.2 MB）
- **无 Hook D**：Hook A(INT8)+B+C+E（AMOTA 0.341，BEV 43 MB）
- ⚠️ INT8 BEV + Hook D 不可组合（误差叠加致 AMOTA 0.241）

---

## 优化方向（按优先级排序）

### P0：Infra BEV 编码器 INT8 PTQ

**目标**：infra BEV 引擎从 41 MB (FP16) 压缩至 ~25 MB (INT8)

**方案**：
1. 收集 infra BEV 校准数据（50 帧 temporal，参照 ego 做法）
2. 使用 `build_trt_int8_univ2x.py --target bev_encoder` 构建 INT8 引擎
3. 端到端验证 AMOTA 不退化

**预期收益**：引擎减 ~16 MB，延迟可能小幅降低  
**难度**：低  
**依赖**：需先 dump infra 校准数据（`dump_univ2x_calibration.py` 适配 infra 模型）

---

### P1：检测头 INT8 PTQ ✅ 量化完成 / ⚠️ Hook D 精度受限

**目标**：检测头引擎从 33.4 MB (FP16) 压缩至 ~20 MB (INT8)

**实际结果**：33.4 MB → **18.2 MB**（-45.5%），延迟 3.45ms → **2.92ms**（-15%）

**完成的步骤**：
1. ✅ 校准数据采集（`dump_heads_calibration.py`，50 帧，1101-query 零填充）
2. ✅ INT8 引擎构建（复用 `build_trt_int8_univ2x.py --target heads`）
3. ✅ MSDAPlugin INT8 兼容性验证（6 个 plugin 层自动保持 FP16）
4. ✅ 端到端 AMOTA 验证

**精度结果**：
- FP16 BEV + INT8 检测头 = AMOTA 0.332（-0.013，可接受）✅
- INT8 BEV + 检测头 TRT = AMOTA 0.241（-0.100，不可接受）❌ — 原因是 INT8 BEV 偏移经 Decoder Cross-Attention 超线性放大

**结论**：检测头 INT8 在 FP16 BEV 条件下可用。INT8 BEV + Hook D 不可组合（见 P1a）

---

### P1a：INT8 BEV + Hook D 误差叠加修复

**问题**：INT8 BEV + Hook D 组合导致 AMOTA 0.241（-0.100），但单独使用均正常

**方案选项**（按优先级）：
1. **用 INT8 BEV 输出重新校准检测头**：当前检测头校准数据来自 FP16 BEV 输出，与 INT8 BEV 部署时的输入分布不匹配。用 INT8 BEV 引擎的输出重新采集校准数据并重建引擎
2. **放弃 INT8 BEV + Hook D 组合**：接受限制，使用 FP16 BEV + INT8 Hook D（AMOTA 0.332）或 INT8 BEV + Hook E（AMOTA 0.341）
3. **Attention Mask**：在 Decoder Self-Attention 中屏蔽零填充 query，减少误差放大通道

**难度**：方案 1 低（仅需修改校准数据采集流程），方案 3 中-高  
**依赖**：无

---

### P2：对称 AdaRound + 完整 W+A Q/DQ（研究性）

**目标**：INT8 精度超越 Vanilla PTQ（目标 AMOTA >= 0.370）

**方案**：
1. 改 `UniformAffineQuantizer` 使用对称 INT8（`sym=True`，range [-127, 127]，zp=0）
2. AdaRound 优化在对称 INT8 网格上进行
3. 导出 ONNX 时同时插入权重 Q/DQ 和激活 Q/DQ
4. TRT 显式量化模式下正确 fuse

**预期收益**：若成功，INT8 精度接近甚至超过 FP16  
**难度**：高（需重写量化器 + 全面的 Q/DQ 节点插入）  
**风险**：工程复杂度高，收益不确定

---

### P3：Backbone QAT（DCNv2）

**目标**：将 backbone 也纳入 TRT 加速

**方案**：需 C++ 量化插件支持 DCNv2  
**难度**：极高  
**状态**：暂不考虑

---

## 已完成的验证清单

- [x] Phase 1：BEV 编码器 TRT FP16（ego + infra）
- [x] Phase 2：检测头 TRT FP16
- [x] Phase 3：下游头 TRT FP16
- [x] Phase V2X：V2X 融合向量化（392-1191x 加速）
- [x] Phase 3C：V2X 检测头 TRT（1101-query）
- [x] Phase E：BEV 编码器 INT8 PTQ（ego）
- [x] Phase DS：下游头 INT8 PTQ（ego + infra）
- [x] Infra BEV 引擎权重修复 + 验证（2026-04-08）
- [x] Hook E 无损验证（三组配置 AMOTA 一致）
- [x] AdaRound 方案 A/B 实验（已确认失败，记录根因）
- [ ] P0：Infra BEV INT8 PTQ
- [x] P1：检测头 INT8 PTQ（FP16 BEV 下可用，AMOTA 0.332）
- [ ] P1a：INT8 BEV + Hook D 误差叠加修复（重新校准或 attention mask）
- [ ] P2：对称 AdaRound + W+A Q/DQ
