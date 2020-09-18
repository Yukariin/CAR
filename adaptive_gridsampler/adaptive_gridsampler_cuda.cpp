#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void adaptive_gridsampler_cuda_forward(
    torch::Tensor img,
    torch::Tensor kernels,
    torch::Tensor offsets_h,
    torch::Tensor offsets_v,
    int offset_unit,
    int padding,
    torch::Tensor output);

std::vector<torch::Tensor> adaptive_gridsampler_cuda_backward(
    torch::Tensor img,
    torch::Tensor kernels,
    torch::Tensor offsets_h,
    torch::Tensor offsets_v,
    int offset_unit,
    int padding,
    torch::Tensor output);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int adaptive_gridsampler_forward(
    torch::Tensor img,
    torch::Tensor kernels,
    torch::Tensor offsets_h,
    torch::Tensor offsets_v,
    int offset_unit,
    int padding,
    torch::Tensor output) {
    CHECK_INPUT(img);
    CHECK_INPUT(kernels);
    CHECK_INPUT(offsets_h);
    CHECK_INPUT(offsets_v);
    CHECK_INPUT(output);

    adaptive_gridsampler_cuda_forward(
        img,
        kernels,
        offsets_h, offsets_v,
        offset_unit,
        padding,
        output
    );
    return 1;
}

std::vector<torch::Tensor> adaptive_gridsampler_backward(
    torch::Tensor img,
    torch::Tensor kernels,
    torch::Tensor offsets_h,
    torch::Tensor offsets_v,
    int offset_unit,
    int padding,
    torch::Tensor output) {
    CHECK_INPUT(img);
    CHECK_INPUT(kernels);
    CHECK_INPUT(offsets_h);
    CHECK_INPUT(offsets_v);
    CHECK_INPUT(output);

    return adaptive_gridsampler_cuda_backward(
        img,
        kernels,
        offsets_h, offsets_v,
        offset_unit,
        padding,
        output
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &adaptive_gridsampler_forward, "adaptive gridsampler forward (CUDA)");
    m.def("backward", &adaptive_gridsampler_backward, "adaptive gridsampler backward (CUDA)");
}
