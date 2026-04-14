#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
  int device = 4;
  int m = 128;
  int n = 128;
  int k = 128;
  int iterations = 20;
  int warmup = 5;
  std::string dtype = "f16";
};

void checkCuda(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
  }
}

void checkCublas(cublasStatus_t status, const char* message) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(message);
  }
}

Options parseArgs(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      opts.device = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
      opts.m = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      opts.n = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
      opts.k = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      opts.iterations = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      opts.warmup = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
      opts.dtype = argv[++i];
    }
  }
  if (opts.dtype != "f16") {
    throw std::runtime_error("only --dtype f16 is supported in the MVP");
  }
  return opts;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options opts = parseArgs(argc, argv);

    checkCuda(cudaSetDevice(opts.device), "cudaSetDevice failed");

    cudaDeviceProp prop{};
    checkCuda(cudaGetDeviceProperties(&prop, opts.device), "cudaGetDeviceProperties failed");

    cublasLtHandle_t lt_handle;
    checkCublas(cublasLtCreate(&lt_handle), "cublasLtCreate failed");

    const size_t elements_a = static_cast<size_t>(opts.m) * static_cast<size_t>(opts.k);
    const size_t elements_b = static_cast<size_t>(opts.k) * static_cast<size_t>(opts.n);
    const size_t elements_c = static_cast<size_t>(opts.m) * static_cast<size_t>(opts.n);

    __half* a = nullptr;
    __half* b = nullptr;
    __half* c = nullptr;

    checkCuda(cudaMalloc(&a, elements_a * sizeof(__half)), "cudaMalloc A failed");
    checkCuda(cudaMalloc(&b, elements_b * sizeof(__half)), "cudaMalloc B failed");
    checkCuda(cudaMalloc(&c, elements_c * sizeof(__half)), "cudaMalloc C failed");

    checkCuda(cudaMemset(a, 0, elements_a * sizeof(__half)), "cudaMemset A failed");
    checkCuda(cudaMemset(b, 0, elements_b * sizeof(__half)), "cudaMemset B failed");
    checkCuda(cudaMemset(c, 0, elements_c * sizeof(__half)), "cudaMemset C failed");

    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    cublasLtMatmulDesc_t operation_desc;
    checkCublas(
        cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
        "cublasLtMatmulDescCreate failed");
    checkCublas(
        cublasLtMatmulDescSetAttribute(
            operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
        "set TRANSA failed");
    checkCublas(
        cublasLtMatmulDescSetAttribute(
            operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
        "set TRANSB failed");

    cublasLtMatrixLayout_t layout_a;
    cublasLtMatrixLayout_t layout_b;
    cublasLtMatrixLayout_t layout_c;
    checkCublas(
        cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_16F, opts.m, opts.k, opts.m),
        "layout A create failed");
    checkCublas(
        cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_16F, opts.k, opts.n, opts.k),
        "layout B create failed");
    checkCublas(
        cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_16F, opts.m, opts.n, opts.m),
        "layout C create failed");

    cublasLtMatmulPreference_t preference;
    checkCublas(cublasLtMatmulPreferenceCreate(&preference), "preference create failed");
    size_t workspace_size = 1 << 22;
    void* workspace = nullptr;
    checkCuda(cudaMalloc(&workspace, workspace_size), "workspace alloc failed");
    checkCublas(
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size,
            sizeof(workspace_size)),
        "set workspace pref failed");

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results = 0;
    checkCublas(
        cublasLtMatmulAlgoGetHeuristic(
            lt_handle,
            operation_desc,
            layout_a,
            layout_b,
            layout_c,
            layout_c,
            preference,
            1,
            &heuristic,
            &returned_results),
        "cublasLtMatmulAlgoGetHeuristic failed");
    if (returned_results == 0) {
      throw std::runtime_error("no cublasLt heuristic returned for requested GEMM");
    }

    auto run_once = [&]() {
      checkCublas(
          cublasLtMatmul(
              lt_handle,
              operation_desc,
              &alpha,
              a,
              layout_a,
              b,
              layout_b,
              &beta,
              c,
              layout_c,
              c,
              layout_c,
              &heuristic.algo,
              workspace,
              workspace_size,
              0),
          "cublasLtMatmul failed");
    };

    for (int i = 0; i < opts.warmup; ++i) {
      run_once();
    }
    checkCuda(cudaDeviceSynchronize(), "warmup synchronize failed");

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start) failed");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");
    checkCuda(cudaEventRecord(start), "cudaEventRecord(start) failed");
    for (int i = 0; i < opts.iterations; ++i) {
      run_once();
    }
    checkCuda(cudaEventRecord(stop), "cudaEventRecord(stop) failed");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop) failed");

    float elapsed_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime failed");
    const double latency_us = (elapsed_ms * 1000.0) / static_cast<double>(opts.iterations);

    std::cout << "{"
              << "\"latency_us\":" << latency_us << ","
              << "\"iterations\":" << opts.iterations << ","
              << "\"device\":" << opts.device << ","
              << "\"gpu_name\":\"" << prop.name << "\""
              << "}" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatmulDescDestroy(operation_desc);
    cublasLtDestroy(lt_handle);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
  } catch (const std::exception& exc) {
    std::fprintf(stderr, "cublaslt_gemm_bench failed: %s\n", exc.what());
    return 1;
  }
}
