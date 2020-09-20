#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

template <typename scalar_t>
__global__ void adaptive_gridsampler_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> img,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> kernels,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> offsets_h,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> offsets_v,
    const int offset_unit,
    const int padding,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
    const size_t n) {
    auto global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_idx >= n) return;

    auto dim_b = output.size(0);
    auto dim_c = output.size(1);
    auto dim_h = output.size(2);
    auto dim_w = output.size(3);

    auto idb = (global_idx / (dim_c * dim_h * dim_w)) % dim_b;
    auto idc = (global_idx / (dim_h * dim_w)) % dim_c;
    auto idy = (global_idx / dim_w) % dim_h;
    auto idx = global_idx % dim_w;

    if(idx >= dim_w || idy >= dim_h)
        return;

    int k_size = sqrt(float(kernels.size(1)));
    float w = float(img.size(3) - 2 * padding);
    float h = float(img.size(2) - 2 * padding);

    scalar_t result = 0;
    for (int k_y = 0; k_y < k_size; ++k_y) {
        for(int k_x = 0; k_x < k_size; ++k_x) {
            scalar_t offset_h = offsets_h[idb][k_size * k_y + k_x][idy][idx] * offset_unit;
            scalar_t offset_v = offsets_v[idb][k_size * k_y + k_x][idy][idx] * offset_unit;

            scalar_t p_x = static_cast<scalar_t>(idx + 0.5) / dim_w * w + k_x + offset_h - 0.5;
            scalar_t p_y = static_cast<scalar_t>(idy + 0.5) / dim_h * h + k_y + offset_v - 0.5;
            scalar_t alpha = p_x - floor(p_x);
            scalar_t beta = p_y - floor(p_y);

            int xL = max(min(int(floor(p_x)), int(w + 2 * padding - 1)), 0);
            int xR = max(min(xL + 1, int(w + 2 * padding - 1)), 0);
            int yT = max(min(int(floor(p_y)), int(h + 2 * padding - 1)), 0);
            int yB = max(min(yT + 1, int(h + 2 * padding - 1)), 0);

            scalar_t val = 0;
            val += (1 - alpha) * (1 - beta) * img[idb][idc][yT][xL];
            val += alpha * (1 - beta) * img[idb][idc][yT][xR];
            val += (1 - alpha) * beta * img[idb][idc][yB][xL];
            val += alpha * beta * img[idb][idc][yB][xR];

            result += val * kernels[idb][k_size * k_y + k_x][idy][idx];
        }
    }
    output[idb][idc][idy][idx] = result;
}

template <typename scalar_t>
__global__ void adaptive_gridsampler_backward_kernel(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_kernels,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_offsets_h,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_offsets_v,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> img,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> kernels,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> offsets_h,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> offsets_v,
    const int offset_unit,
    const int padding,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
    const size_t n) {
    auto global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_idx >= n) return;

    auto dim_b = output.size(0);
    auto dim_c = output.size(1);
    auto dim_h = output.size(2);
    auto dim_w = output.size(3);

    auto idb = (global_idx / (dim_c * dim_h * dim_w)) % dim_b;
    auto idc = (global_idx / (dim_h * dim_w)) % dim_c;
    auto idy = (global_idx / dim_w) % dim_h;
    auto idx = global_idx % dim_w;

    if(idx >= dim_w || idy >= dim_h)
        return;

    int k_size = sqrt(float(kernels.size(1)));
    float w = float(img.size(3) - 2 * padding);
    float h = float(img.size(2) - 2 * padding);

    for (int k_y = 0; k_y < k_size; ++k_y) {
        for(int k_x = 0; k_x < k_size; ++k_x) {
            scalar_t kernel = kernels[idb][k_size * k_y + k_x][idy][idx];
            scalar_t offset_h = offsets_h[idb][k_size * k_y + k_x][idy][idx] * offset_unit;
            scalar_t offset_v = offsets_v[idb][k_size * k_y + k_x][idy][idx] * offset_unit;

            scalar_t p_x = static_cast<scalar_t>(idx + 0.5) / dim_w * w + k_x + offset_h - 0.5;
            scalar_t p_y = static_cast<scalar_t>(idy + 0.5) / dim_h * h + k_y + offset_v - 0.5;
            scalar_t alpha = p_x - floor(p_x);
            scalar_t beta = p_y - floor(p_y);

            int xL = max(min(int(floor(p_x)), int(w + 2 * padding - 1)), 0);
            int xR = max(min(xL + 1, int(w + 2 * padding - 1)), 0);
            int yT = max(min(int(floor(p_y)), int(h + 2 * padding - 1)), 0);
            int yB = max(min(yT + 1, int(h + 2 * padding - 1)), 0);

            scalar_t val = 0;
            val += (1 - alpha) * (1 - beta) * img[idb][idc][yT][xL];
            val += alpha * (1 - beta) * img[idb][idc][yT][xR];
            val += (1 - alpha) * beta * img[idb][idc][yB][xL];
            val += alpha * beta * img[idb][idc][yB][xR];
            d_kernels[idb][k_size * k_y + k_x][idy][idx] += val;

            val = 0;
            val += kernel * (1 - beta) * (img[idb][idc][yT][xR] - img[idb][idc][yT][xL]);
            val += beta * (img[idb][idc][yB][xR] - img[idb][idc][yB][xL]);
            d_offsets_v[idb][k_size * k_y + k_x][idy][idx] += val;

            val = 0;
            val += kernel * (1 - alpha) * (img[idb][idc][yB][xL] - img[idb][idc][yT][xL]);
            val += alpha * (img[idb][idc][yB][xR] - img[idb][idc][yT][xR]);
            d_offsets_h[idb][k_size * k_y + k_x][idy][idx] += val;
        }
    }
}

void adaptive_gridsampler_cuda_forward(
    torch::Tensor img,
    torch::Tensor kernels,
    torch::Tensor offsets_h,
    torch::Tensor offsets_v,
    int offset_unit,
    int padding,
    torch::Tensor output) {
    const auto numel = output.numel();

    const int threads = 256;
    const dim3 blocks((numel + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(img.type(), "adaptive_gridsampler_cuda_forward", ([&] {
        adaptive_gridsampler_forward_kernel<scalar_t><<<blocks, threads>>>(
            img.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            kernels.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            offsets_h.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            offsets_v.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            offset_unit,
            padding,
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            numel);
    }));
}

std::vector<torch::Tensor> adaptive_gridsampler_cuda_backward(
    torch::Tensor img,
    torch::Tensor kernels,
    torch::Tensor offsets_h,
    torch::Tensor offsets_v,
    int offset_unit,
    int padding,
    torch::Tensor output) {
    auto d_kernels = torch::zeros_like(kernels);
    auto d_offsets_h = torch::zeros_like(offsets_h);
    auto d_offsets_v = torch::zeros_like(offsets_v);

    const auto numel = output.numel();

    const int threads = 256;
    const dim3 blocks((numel + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(img.type(), "adaptive_gridsampler_cuda_backward", ([&] {
        adaptive_gridsampler_backward_kernel<scalar_t><<<blocks, threads>>>(
            d_kernels.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            d_offsets_h.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            d_offsets_v.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            img.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            kernels.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            offsets_h.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            offsets_v.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            offset_unit,
            padding,
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            numel);
    }));

    return {d_kernels, d_offsets_h, d_offsets_v};
}
