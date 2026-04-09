#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <stdexcept>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

namespace {

struct Options {
  int device = 0;
  int m = 128;
  int n = 128;
  int k = 128;
  int tb_m = 128;
  int tb_n = 128;
  int tb_k = 32;
  int split_k = 1;
  int iterations = 20;
  int warmup = 5;
  bool quiet = false;
  std::string swizzle = "Identity";
};

struct BenchmarkRun {
  double latency_us = 0.0;
  std::string_view profile = "aligned";
};

void check_cuda(cudaError_t status, char const* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

Options parse_args(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> char const* {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + arg);
      }
      return argv[++i];
    };
    if (arg == "--device") options.device = std::atoi(next());
    else if (arg == "--m") options.m = std::atoi(next());
    else if (arg == "--n") options.n = std::atoi(next());
    else if (arg == "--k") options.k = std::atoi(next());
    else if (arg == "--tb-m") options.tb_m = std::atoi(next());
    else if (arg == "--tb-n") options.tb_n = std::atoi(next());
    else if (arg == "--tb-k") options.tb_k = std::atoi(next());
    else if (arg == "--split-k") options.split_k = std::atoi(next());
    else if (arg == "--iterations") options.iterations = std::atoi(next());
    else if (arg == "--warmup") options.warmup = std::atoi(next());
    else if (arg == "--quiet") options.quiet = true;
    else if (arg == "--swizzle") options.swizzle = next();
    else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
}

template <typename ThreadblockShape, typename WarpShape, typename Swizzle>
int run_bench(Options const& options) {
  using Element = cutlass::half_t;
  using Layout = cutlass::layout::RowMajor;
  using PermissiveLayoutA = cutlass::layout::RowMajor;
  using PermissiveLayoutB = cutlass::layout::ColumnMajor;
  using PermissiveLayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using MmaOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using PermissiveArch = cutlass::arch::Sm75;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using PermissiveInstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using AlignedEpilogueOp = cutlass::epilogue::thread::LinearCombination<
      Element, 8, ElementAccumulator, ElementCompute>;
  using PermissiveEpilogueOp = cutlass::epilogue::thread::LinearCombination<
      Element, 1, ElementAccumulator, ElementCompute>;
  using AlignedConvertScaledOp = cutlass::epilogue::thread::Convert<
      ElementAccumulator, AlignedEpilogueOp::kCount, ElementAccumulator>;
  using PermissiveConvertScaledOp = cutlass::epilogue::thread::Convert<
      ElementAccumulator, PermissiveEpilogueOp::kCount, ElementAccumulator>;
  using AlignedReductionOp = cutlass::reduction::thread::ReduceAdd<
      ElementAccumulator,
      typename AlignedEpilogueOp::ElementAccumulator,
      AlignedEpilogueOp::kCount>;
  using PermissiveReductionOp = cutlass::reduction::thread::ReduceAdd<
      ElementAccumulator,
      typename PermissiveEpilogueOp::ElementAccumulator,
      PermissiveEpilogueOp::kCount>;
  constexpr int NumStages = 4;
  constexpr int PermissiveStages = 2;

  using AlignedGemm = cutlass::gemm::device::Gemm<
      Element,
      Layout,
      Element,
      Layout,
      Element,
      Layout,
      ElementAccumulator,
      MmaOp,
      SmArch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      AlignedEpilogueOp,
      Swizzle,
      NumStages>;

  using PermissiveGemm = cutlass::gemm::device::Gemm<
      Element,
      PermissiveLayoutA,
      Element,
      PermissiveLayoutB,
      Element,
      PermissiveLayoutC,
      ElementAccumulator,
      MmaOp,
      PermissiveArch,
      ThreadblockShape,
      WarpShape,
      PermissiveInstructionShape,
      PermissiveEpilogueOp,
      Swizzle,
      PermissiveStages,
      1,
      1>;

  using AlignedSplitKGemm = cutlass::gemm::device::GemmSplitKParallel<
      Element,
      Layout,
      Element,
      Layout,
      Element,
      Layout,
      ElementAccumulator,
      MmaOp,
      SmArch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      AlignedEpilogueOp,
      AlignedConvertScaledOp,
      AlignedReductionOp,
      Swizzle,
      NumStages>;

  using PermissiveSplitKGemm = cutlass::gemm::device::GemmSplitKParallel<
      Element,
      PermissiveLayoutA,
      Element,
      PermissiveLayoutB,
      Element,
      PermissiveLayoutC,
      ElementAccumulator,
      MmaOp,
      PermissiveArch,
      ThreadblockShape,
      WarpShape,
      PermissiveInstructionShape,
      PermissiveEpilogueOp,
      PermissiveConvertScaledOp,
      PermissiveReductionOp,
      Swizzle,
      PermissiveStages,
      1,
      1>;

  check_cuda(cudaSetDevice(options.device), "cudaSetDevice");

  cutlass::gemm::GemmCoord problem_size(options.m, options.n, options.k);
  cutlass::HostTensor<Element, Layout> tensor_a(problem_size.mk());
  cutlass::HostTensor<Element, Layout> tensor_b(problem_size.kn());
  cutlass::HostTensor<Element, Layout> tensor_c(problem_size.mn());
  cutlass::HostTensor<Element, Layout> tensor_d(problem_size.mn());
  cutlass::HostTensor<Element, PermissiveLayoutA> permissive_tensor_a(problem_size.mk());
  cutlass::HostTensor<Element, PermissiveLayoutB> permissive_tensor_b(problem_size.kn());
  cutlass::HostTensor<Element, PermissiveLayoutC> permissive_tensor_c(problem_size.mn());
  cutlass::HostTensor<Element, PermissiveLayoutC> permissive_tensor_d(problem_size.mn());

  cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 1, Element(2), Element(-2), 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 2, Element(2), Element(-2), 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_c.host_view(), 3, Element(2), Element(-2), 0);
  cutlass::reference::host::TensorFill(tensor_d.host_view());
  cutlass::reference::host::TensorFillRandomUniform(
      permissive_tensor_a.host_view(), 11, Element(2), Element(-2), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      permissive_tensor_b.host_view(), 12, Element(2), Element(-2), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      permissive_tensor_c.host_view(), 13, Element(2), Element(-2), 0);
  cutlass::reference::host::TensorFill(permissive_tensor_d.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  permissive_tensor_a.sync_device();
  permissive_tensor_b.sync_device();
  permissive_tensor_c.sync_device();
  permissive_tensor_d.sync_device();

  Swizzle swizzle;
  cutlass::gemm::GemmCoord threadblock_shape(
      ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK);
  cutlass::gemm::GemmCoord tiled_shape =
      swizzle.get_tiled_shape(problem_size, threadblock_shape, options.split_k);
  int swizzle_log_tile = swizzle.get_log_tile(tiled_shape);
  int task_count = tiled_shape.m() * tiled_shape.n() * tiled_shape.k();

  auto try_run = [&](auto tag,
                     auto ref_a,
                     auto ref_b,
                     auto ref_c,
                     auto ref_d,
                     std::string_view profile_name,
                     BenchmarkRun& run) -> bool {
    using GemmOp = decltype(tag);
    typename GemmOp::Arguments arguments(
        problem_size,
        ref_a,
        ref_b,
        ref_c,
        ref_d,
        {ElementCompute(1), ElementCompute(0)},
        options.split_k);

    size_t workspace_size = GemmOp::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    GemmOp gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }
    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("initialize failed");
    }

    for (int iter = 0; iter < options.warmup; ++iter) {
      status = gemm_op();
      if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("warmup launch failed");
      }
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int iter = 0; iter < options.iterations; ++iter) {
      status = gemm_op();
      if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("profile launch failed");
      }
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    run.latency_us = (static_cast<double>(total_ms) * 1000.0) / options.iterations;
    run.profile = profile_name;
    return true;
  };

  BenchmarkRun run;
  bool did_run = false;
  if (options.split_k > 1) {
    did_run = try_run(
                  AlignedSplitKGemm{},
                  tensor_a.device_ref(),
                  tensor_b.device_ref(),
                  tensor_c.device_ref(),
                  tensor_d.device_ref(),
                  "aligned",
                  run)
        || try_run(
                  PermissiveSplitKGemm{},
                  permissive_tensor_a.device_ref(),
                  permissive_tensor_b.device_ref(),
                  permissive_tensor_c.device_ref(),
                  permissive_tensor_d.device_ref(),
                  "permissive",
                  run);
  } else {
    did_run = try_run(
                  AlignedGemm{},
                  tensor_a.device_ref(),
                  tensor_b.device_ref(),
                  tensor_c.device_ref(),
                  tensor_d.device_ref(),
                  "aligned",
                  run)
        || try_run(
                  PermissiveGemm{},
                  permissive_tensor_a.device_ref(),
                  permissive_tensor_b.device_ref(),
                  permissive_tensor_c.device_ref(),
                  permissive_tensor_d.device_ref(),
                  "permissive",
                  run);
  }

  if (!did_run) {
    std::cerr << "can_implement failed\n";
    return 2;
  }

  if (!options.quiet) {
    std::cout
        << "{"
        << "\"kernel_profile\":\"" << run.profile << "\","
        << "\"task_count\":" << task_count << ","
        << "\"swizzle_log_tile\":" << swizzle_log_tile << ","
        << "\"grid_tiled_shape\":[" << tiled_shape.m() << "," << tiled_shape.n() << "," << tiled_shape.k() << "],"
        << "\"iterations\":" << options.iterations << ","
        << "\"latency_us\":" << run.latency_us
        << "}\n";
  }

  return 0;
}

template <typename ThreadblockShape, typename WarpShape>
int dispatch_swizzle(Options const& options) {
  if (options.swizzle == "Identity") {
    return run_bench<ThreadblockShape, WarpShape, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>>(options);
  }
  if (options.swizzle == "Identity2") {
    return run_bench<ThreadblockShape, WarpShape, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>>(options);
  }
  if (options.swizzle == "Identity4") {
    return run_bench<ThreadblockShape, WarpShape, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>(options);
  }
  throw std::runtime_error("unsupported swizzle: " + options.swizzle);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_args(argc, argv);
    if (options.tb_m == 128 && options.tb_n == 128 && options.tb_k == 32) {
      return dispatch_swizzle<cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>>(options);
    }
    if (options.tb_m == 128 && options.tb_n == 64 && options.tb_k == 32) {
      return dispatch_swizzle<cutlass::gemm::GemmShape<128, 64, 32>, cutlass::gemm::GemmShape<64, 64, 32>>(options);
    }
    std::cerr << "unsupported threadblock shape\n";
    return 6;
  } catch (std::exception const& error) {
    std::cerr << error.what() << "\n";
    return 7;
  }
}
