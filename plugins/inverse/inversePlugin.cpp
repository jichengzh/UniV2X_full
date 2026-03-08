// Inverse TRT Plugin – wraps cublas batched matrix inverse
// Adapted from BEVFormer_tensorrt (MIT License) with TRT10 support
#include "inversePlugin.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace nvinfer1;

PluginFieldCollection InversePluginCreator::mFC{};
std::vector<PluginField> InversePluginCreator::mPluginAttributes{};

InversePlugin::InversePlugin(const std::string name, const void* data, size_t length) {}

int32_t InversePlugin::initialize() noexcept {
#if !(NV_TENSORRT_MAJOR >= 10 && NV_TENSORRT_MINOR >= 4)
    cublasCreate(&m_cublas_handle);
#endif
    return 0;
}

void InversePlugin::terminate() noexcept {
#if !(NV_TENSORRT_MAJOR >= 10 && NV_TENSORRT_MINOR >= 4)
    if (m_cublas_handle) cublasDestroy(m_cublas_handle);
    m_cublas_handle = nullptr;
#endif
}

void InversePlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                                    IGpuAllocator* allocator) noexcept {
#if NV_TENSORRT_MAJOR >= 10 && NV_TENSORRT_MINOR >= 4
    cublasCreate(&m_cublas_handle);
#else
    m_cublas_handle = cublas;
#endif
}

DimsExprs InversePlugin::getOutputDimensions(int32_t outputIndex,
                                              const DimsExprs* inputs,
                                              int32_t nbInputs,
                                              IExprBuilder& exprBuilder) noexcept {
    return inputs[0]; // output same shape as input
}

bool InversePlugin::supportsFormatCombination(int32_t pos,
                                               const PluginTensorDesc* inOut,
                                               int32_t nbInputs,
                                               int32_t nbOutputs) noexcept {
    return (inOut[pos].type == DataType::kFLOAT) &&
           (inOut[pos].format == TensorFormat::kLINEAR);
}

DataType InversePlugin::getOutputDataType(int32_t index, const DataType* inputTypes,
                                           int32_t nbInputs) const noexcept {
    return inputTypes[0];
}

IPluginV2DynamicExt* InversePlugin::clone() const noexcept {
    try {
        auto* plugin = new InversePlugin();
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        plugin->initialize();
#if NV_TENSORRT_MAJOR >= 10 && NV_TENSORRT_MINOR >= 4
        plugin->m_cublas_handle = m_cublas_handle;
#endif
        return plugin;
    } catch (std::exception const& e) {
        return nullptr;
    }
}

// Compute matrix inverse using cublas batched LU decomposition
template <typename T>
static int computeBatchedInverse(cublasHandle_t handle, int batch, int n,
                                  const T* input, T* output, cudaStream_t stream);

template <>
int computeBatchedInverse<float>(cublasHandle_t handle, int batch, int n,
                                  const float* input, float* output,
                                  cudaStream_t stream) {
    // Allocate device arrays of pointers
    std::vector<float*> h_A_ptrs(batch), h_C_ptrs(batch);
    float **d_A_ptrs, **d_C_ptrs;
    int* d_pivots;
    int* d_info;

    cudaMalloc((void**)&d_A_ptrs, batch * sizeof(float*));
    cudaMalloc((void**)&d_C_ptrs, batch * sizeof(float*));
    cudaMalloc((void**)&d_pivots, batch * n * sizeof(int));
    cudaMalloc((void**)&d_info, batch * sizeof(int));

    // Set pointers (input is row-major, cublas is col-major, but for symmetric inverse it's fine)
    // We do in-place: copy input to output first
    cudaMemcpyAsync(output, input, batch * n * n * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    for (int i = 0; i < batch; i++) {
        h_A_ptrs[i] = output + i * n * n;
        h_C_ptrs[i] = output + i * n * n; // in-place not supported, but ok for identity init
    }

    cudaMemcpyAsync(d_A_ptrs, h_A_ptrs.data(), batch * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C_ptrs, h_C_ptrs.data(), batch * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);

    // LU factorization
    cublasSgetrfBatched(handle, n, d_A_ptrs, n, d_pivots, d_info, batch);

    // Allocate separate output
    float* d_inv = nullptr;
    cudaMalloc((void**)&d_inv, batch * n * n * sizeof(float));
    float** d_inv_ptrs;
    cudaMalloc((void**)&d_inv_ptrs, batch * sizeof(float*));
    std::vector<float*> h_inv_ptrs(batch);
    for (int i = 0; i < batch; i++) h_inv_ptrs[i] = d_inv + i * n * n;
    cudaMemcpyAsync(d_inv_ptrs, h_inv_ptrs.data(), batch * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);

    // Inverse from LU
    cublasSgetriBatched(handle, n, (const float**)d_A_ptrs, n, d_pivots,
                        d_inv_ptrs, n, d_info, batch);

    cudaMemcpyAsync(output, d_inv, batch * n * n * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    cudaFree(d_A_ptrs);
    cudaFree(d_C_ptrs);
    cudaFree(d_pivots);
    cudaFree(d_info);
    cudaFree(d_inv);
    cudaFree(d_inv_ptrs);
    return 0;
}

int32_t InversePlugin::enqueue(const PluginTensorDesc* inputDesc,
                                const PluginTensorDesc* outputDesc,
                                const void* const* inputs, void* const* outputs,
                                void* workspace, cudaStream_t stream) noexcept {
#if NV_TENSORRT_MAJOR >= 10 && NV_TENSORRT_MINOR >= 4
    cublasSetStream(m_cublas_handle, stream);
#endif
    auto data_type = inputDesc[0].type;
    // Input shape: (..., N, N) — flatten leading dims to batch
    int n = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    int total_elems = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
        total_elems *= inputDesc[0].dims.d[i];
    int batch = total_elems / (n * n);

    switch (data_type) {
    case DataType::kFLOAT:
        computeBatchedInverse<float>(m_cublas_handle, batch, n,
                                     (const float*)inputs[0],
                                     (float*)outputs[0], stream);
        break;
    default:
        return -1;
    }
    return 0;
}

// Plugin creator
InversePluginCreator::InversePluginCreator() {
    mFC.nbFields = 0;
    mFC.fields = nullptr;
}

IPluginV2* InversePluginCreator::createPlugin(const char* name,
                                               const PluginFieldCollection* fc) noexcept {
    try {
        auto* plugin = new InversePlugin();
        return plugin;
    } catch (...) { return nullptr; }
}

IPluginV2* InversePluginCreator::deserializePlugin(const char* name,
                                                    const void* serialData,
                                                    size_t serialLength) noexcept {
    try {
        return new InversePlugin(name, serialData, serialLength);
    } catch (...) { return nullptr; }
}
