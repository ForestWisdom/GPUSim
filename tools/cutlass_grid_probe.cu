#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct BlockRecord {
  int tile_m;
  int tile_n;
  int tile_k;
};

struct Options {
  int m = 0;
  int n = 0;
  int k = 0;
  int tb_m = 0;
  int tb_n = 0;
  int tb_k = 0;
  int split_k = 1;
  int device = 0;
  std::string swizzle = "Identity";
  std::string dtype = "f16";
};

static int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

static int round_up(int x, int y) {
  return ceil_div(x, y) * y;
}

static int dtype_bits(std::string const &dtype) {
  if (dtype == "f16" || dtype == "bf16") {
    return 16;
  }
  if (dtype == "f32" || dtype == "tf32") {
    return 32;
  }
  if (dtype == "int8") {
    return 8;
  }
  return 16;
}

static Options parse_args(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next_value = [&]() -> char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--m") options.m = std::atoi(next_value());
    else if (arg == "--n") options.n = std::atoi(next_value());
    else if (arg == "--k") options.k = std::atoi(next_value());
    else if (arg == "--tb-m") options.tb_m = std::atoi(next_value());
    else if (arg == "--tb-n") options.tb_n = std::atoi(next_value());
    else if (arg == "--tb-k") options.tb_k = std::atoi(next_value());
    else if (arg == "--split-k") options.split_k = std::atoi(next_value());
    else if (arg == "--swizzle") options.swizzle = next_value();
    else if (arg == "--dtype") options.dtype = next_value();
    else if (arg == "--device") options.device = std::atoi(next_value());
  }
  return options;
}

template <typename Swizzle>
__global__ void probe_kernel_log_tile(int swizzle_log_tile, BlockRecord *records) {
  Swizzle swizzle;
  auto tile = swizzle.get_tile_offset(swizzle_log_tile);
  int linear_idx = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  records[linear_idx] = {tile.m(), tile.n(), tile.k()};
}

template <typename Swizzle>
__global__ void probe_kernel_tiled_shape(cutlass::gemm::GemmCoord tiled_shape, BlockRecord *records) {
  Swizzle swizzle;
  auto tile = swizzle.get_tile_offset(tiled_shape);
  int linear_idx = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  records[linear_idx] = {tile.m(), tile.n(), tile.k()};
}

template <typename Swizzle>
static void launch_probe_log_tile(dim3 grid_shape, int swizzle_log_tile, BlockRecord *device_records) {
  probe_kernel_log_tile<Swizzle><<<grid_shape, 1>>>(swizzle_log_tile, device_records);
}

template <typename Swizzle>
static void launch_probe_tiled_shape(
    dim3 grid_shape,
    cutlass::gemm::GemmCoord tiled_shape,
    BlockRecord *device_records) {
  probe_kernel_tiled_shape<Swizzle><<<grid_shape, 1>>>(tiled_shape, device_records);
}

template <typename Swizzle, bool UseLogTile>
static void compute_and_emit(Options const &options) {
  Swizzle swizzle;
  cutlass::gemm::GemmCoord problem_size(options.m, options.n, options.k);
  cutlass::gemm::GemmCoord tile_size(options.tb_m, options.tb_n, options.tb_k);
  cutlass::gemm::GemmCoord grid_tiled_shape = swizzle.get_tiled_shape(problem_size, tile_size, options.split_k);

  int bits = dtype_bits(options.dtype);
  int k_align = std::max(128 / bits, 1);
  int gemm_k_size = round_up(ceil_div(options.k, options.split_k), k_align);
  if (gemm_k_size > 0) {
    grid_tiled_shape = cutlass::gemm::GemmCoord(
        grid_tiled_shape.m(),
        grid_tiled_shape.n(),
        ceil_div(options.k, gemm_k_size));
  }

  int swizzle_log_tile = swizzle.get_log_tile(grid_tiled_shape);
  dim3 grid_shape = swizzle.get_grid_shape(grid_tiled_shape);
  int total_blocks = int(grid_shape.x) * int(grid_shape.y) * int(grid_shape.z);

  BlockRecord *device_records = nullptr;
  cudaMalloc(&device_records, sizeof(BlockRecord) * total_blocks);
  if constexpr (UseLogTile) {
    launch_probe_log_tile<Swizzle>(grid_shape, swizzle_log_tile, device_records);
  } else {
    launch_probe_tiled_shape<Swizzle>(grid_shape, grid_tiled_shape, device_records);
  }
  cudaDeviceSynchronize();

  std::vector<BlockRecord> host_records(total_blocks);
  cudaMemcpy(host_records.data(), device_records, sizeof(BlockRecord) * total_blocks, cudaMemcpyDeviceToHost);
  cudaFree(device_records);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, options.device);

  std::printf("{");
  std::printf("\"device\":%d,", options.device);
  std::printf("\"device_name\":\"%s\",", prop.name);
  std::printf("\"swizzle\":\"%s\",", options.swizzle.c_str());
  std::printf("\"dtype\":\"%s\",", options.dtype.c_str());
  std::printf("\"swizzle_log_tile\":%d,", swizzle_log_tile);
  std::printf("\"grid_shape\":[%u,%u,%u],", grid_shape.x, grid_shape.y, grid_shape.z);
  std::printf("\"grid_tiled_shape\":[%d,%d,%d],", grid_tiled_shape.m(), grid_tiled_shape.n(), grid_tiled_shape.k());
  std::printf("\"gemm_k_size\":%d,", gemm_k_size);
  std::printf("\"tasks\":[");

  bool first = true;
  for (int block_z = 0; block_z < int(grid_shape.z); ++block_z) {
    for (int block_y = 0; block_y < int(grid_shape.y); ++block_y) {
      for (int block_x = 0; block_x < int(grid_shape.x); ++block_x) {
        int linear_idx = (block_z * int(grid_shape.y) + block_y) * int(grid_shape.x) + block_x;
        auto const &record = host_records[linear_idx];

        if (record.tile_m >= grid_tiled_shape.m() || record.tile_n >= grid_tiled_shape.n()) {
          continue;
        }

        int m0 = record.tile_m * options.tb_m;
        int m1 = std::min(options.m, m0 + options.tb_m);
        int n0 = record.tile_n * options.tb_n;
        int n1 = std::min(options.n, n0 + options.tb_n);
        int k0 = record.tile_k * gemm_k_size;
        int k1 = std::min(options.k, (record.tile_k + 1) * gemm_k_size);
        int gemm_k_iterations = ceil_div(k1 - k0, options.tb_k);

        if (!first) {
          std::printf(",");
        }
        first = false;
        std::printf(
            "{"
            "\"block_idx_x\":%d,"
            "\"block_idx_y\":%d,"
            "\"block_idx_z\":%d,"
            "\"tile_idx_m\":%d,"
            "\"tile_idx_n\":%d,"
            "\"tile_idx_k\":%d,"
            "\"m0\":%d,"
            "\"m1\":%d,"
            "\"n0\":%d,"
            "\"n1\":%d,"
            "\"k0\":%d,"
            "\"k1\":%d,"
            "\"m_eff\":%d,"
            "\"n_eff\":%d,"
            "\"k_eff\":%d,"
            "\"gemm_k_iterations\":%d"
            "}",
            block_x,
            block_y,
            block_z,
            record.tile_m,
            record.tile_n,
            record.tile_k,
            m0,
            m1,
            n0,
            n1,
            k0,
            k1,
            m1 - m0,
            n1 - n0,
            k1 - k0,
            gemm_k_iterations);
      }
    }
  }

  std::printf("]}");
}

int main(int argc, char **argv) {
  Options options = parse_args(argc, argv);
  cudaSetDevice(options.device);
  cudaFree(nullptr);

  if (options.swizzle == "Identity") {
    compute_and_emit<cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, true>(options);
  } else if (options.swizzle == "Identity2") {
    compute_and_emit<cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, true>(options);
  } else if (options.swizzle == "Identity4") {
    compute_and_emit<cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, true>(options);
  } else if (options.swizzle == "Identity8") {
    compute_and_emit<cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, true>(options);
  } else if (options.swizzle == "SplitKIdentity") {
    compute_and_emit<cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<1>, true>(options);
  } else if (options.swizzle == "SplitKIdentity2") {
    compute_and_emit<cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<2>, true>(options);
  } else if (options.swizzle == "SplitKIdentity4") {
    compute_and_emit<cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<4>, true>(options);
  } else if (options.swizzle == "SplitKIdentity8") {
    compute_and_emit<cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<8>, true>(options);
  } else if (options.swizzle == "Horizontal") {
    compute_and_emit<cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle, false>(options);
  } else if (options.swizzle == "SplitKHorizontal") {
    compute_and_emit<cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle, true>(options);
  } else {
    std::fprintf(stderr, "unsupported swizzle: %s\n", options.swizzle.c_str());
    return 2;
  }

  return 0;
}
