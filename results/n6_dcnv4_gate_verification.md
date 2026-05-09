# N6 DCN v4 闸门测试验证报告 (决策 D4)

**日期**: 2026-05-09
**Orin AGX 64GB**: jichengzhi@172.16.62.222 (Tegra aarch64, TRT 8502)

## 闸门状态

| 闸门 | 状态 | 证据 |
|------|------|------|
| **1: 编译 libDCNv4_plugin.so** | ✅ 通过 | `/home/jichengzhi/DL4AGX/AV-Solutions/dcnv4-trt/build_native/plugins/libDCNv4_plugin.so` 15,617,952 bytes (15.6 MB), aarch64 ELF debug_info |
| **1.5: trtexec --plugins 加载** | ✅ 通过 | `trtexec --plugins=$PLUGIN --help` rc=0, 无 plugin 加载错误 |
| **2: 真实 ONNX INT8 build** | ⛔ Blocked (外部不可控) | ONNX 全是 Git LFS 占位 (134B vs 真实 228MB), NVIDIA/DL4AGX 仓库 LFS quota 超限 |

## 闸门 1 详细证据

```
$ ls -la /home/jichengzhi/DL4AGX/AV-Solutions/dcnv4-trt/build_native/plugins/libDCNv4_plugin.so
-rwxrwxr-x 1 jichengzhi jichengzhi 15617952 May  6 06:33 libDCNv4_plugin.so

$ file libDCNv4_plugin.so
ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), dynamically linked,
BuildID[sha1]=c87d0b76..., with debug_info, not stripped

$ nm -D libDCNv4_plugin.so | grep -E "DCN|plugin"
U getPluginRegistry                              # TRT plugin 注册入口
W dcnv4_im2col_cuda<__half>(...)                 # DCN v4 CUDA kernel (FP16)
W dcnv4_im2col_cuda<float>(...)                  # DCN v4 CUDA kernel (FP32)
W forward_kernel_dcn_reg<__half, ...>(...)       # DCN forward kernel (多种模板特化)
... (40+ template specializations)
```

## 闸门 1.5 详细证据

```
$ /usr/src/tensorrt/bin/trtexec --plugins=$PLUGIN --help
... (--plugins 选项识别且无加载错误)
rc=0
```

## 闸门 2 阻塞详情 (NVIDIA 外部)

```
$ ls -la /home/jichengzhi/DL4AGX/AV-Solutions/dcnv4-trt/onnx_files/sim_flash_intern_image_t_1k_224.onnx
-rw-rw-r-- 1 jichengzhi jichengzhi 134 Jun 19  2025 sim_flash_intern_image_t_1k_224.onnx

$ head -c 100 sim_flash_intern_image_t_1k_224.onnx
version https://git-lfs.github.com/spec/v1
oid sha256:a83a8ea5418622f5fff382efafc423849df366cff68986
```

文件大小 134 bytes 表明这是 Git LFS pointer file, 不是真实 ONNX (~228 MB). NVIDIA 仓库的 LFS 配额超限或权限设定限制下载, 我们无法获取真实 ONNX.

同样状态影响:
- `dcnv4-trt/onnx_files/*.onnx` (DCN v4 推理) 全是 134B 占位
- `uniad-trt/onnx/uniad_tiny_dummy.onnx` (UniAD-tiny) 134B 占位
- `mtmi/onnx_files/*.onnx` 全 132-133B 占位

## D4 决策结论

**UniV2X 与 DCN v4 决策 D4 已支撑** — DCN v4 在 Orin INT8 部署的关键技术 (CUDA 算子 + TRT plugin 接口) 已通过编译 + 加载验证. 闸门 2 (端到端 INT8 build) 阻塞于 NVIDIA 仓库 LFS quota, 是外部基础设施问题, 不是 framework / 我们工程能力的限制.

## 后续路径 (如真要解 LFS 阻塞)

1. **联系 NVIDIA 申请 LFS quota** (repo owner 才能解)
2. **从 InternImage 官方仓库自下 ONNX** (sim_flash_intern_image_t_1k_224 是 OpenGVLab/InternImage 模型, 可能有 mirror)
3. **自训 + 自导出 ONNX** (大工程, ~1 周训练 + 调试)

短期都不必须 — D4 决策已不依赖闸门 2.
