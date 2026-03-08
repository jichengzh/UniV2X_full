// Inverse TRT Plugin – wraps cublas batched matrix inverse
// Adapted from BEVFormer_tensorrt (MIT License)
#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cublas_v2.h>
#include <string>
#include <vector>

namespace nvinfer1 {

class InversePlugin : public IPluginV2DynamicExt {
public:
    InversePlugin() noexcept {}
    InversePlugin(const std::string name, const void* serialData, size_t serialLength);
    ~InversePlugin() override = default;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override { return "InversePlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* buffer) const noexcept override {}
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(const char* ns) noexcept override { mPluginNamespace = ns; }
    const char* getPluginNamespace() const noexcept override { return mPluginNamespace.c_str(); }

    // IPluginV2Ext methods
    DataType getOutputDataType(int32_t index, const DataType* inputTypes,
                               int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs,
                                  int32_t nbInputs,
                                  IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut,
                                   int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
                         const DynamicPluginTensorDesc* out,
                         int32_t nbOutputs) noexcept override {}
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs,
                            const PluginTensorDesc* outputs,
                            int32_t nbOutputs) const noexcept override { return 0; }
    int32_t enqueue(const PluginTensorDesc* inputDesc,
                    const PluginTensorDesc* outputDesc,
                    const void* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                         IGpuAllocator* allocator) noexcept override;
    void detachFromContext() noexcept override {}

private:
    std::string mPluginNamespace;
#if NV_TENSORRT_MAJOR >= 10 && NV_TENSORRT_MINOR >= 4
    cublasHandle_t m_cublas_handle{nullptr};
#else
    cublasHandle_t m_cublas_handle{nullptr};
#endif
};


class InversePluginCreator : public IPluginCreator {
public:
    InversePluginCreator();
    ~InversePluginCreator() override = default;
    const char* getPluginName() const noexcept override { return "InversePlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }
    IPluginV2* createPlugin(const char* name,
                            const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData,
                                 size_t serialLength) noexcept override;
    void setPluginNamespace(const char* ns) noexcept override { mPluginNamespace = ns; }
    const char* getPluginNamespace() const noexcept override { return mPluginNamespace.c_str(); }

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace;
};

REGISTER_TENSORRT_PLUGIN(InversePluginCreator);

} // namespace nvinfer1
