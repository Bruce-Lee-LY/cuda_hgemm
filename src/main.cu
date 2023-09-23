// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: hgemm main

#include "gflags/gflags.h"
#include "omp.h"
#include "tester.h"

#define HGEMM_FUNC(name) void name(half *A, half *B, half *C, size_t M, size_t N, size_t K)

HGEMM_FUNC(cublasTensorOp);
HGEMM_FUNC(simtNaive);

HGEMM_FUNC(wmmaNaive);
HGEMM_FUNC(wmmaBase);
HGEMM_FUNC(wmmaPadding);
HGEMM_FUNC(wmmaAsync);
HGEMM_FUNC(wmmaAsyncPg2s);
HGEMM_FUNC(wmmaAsyncPg2sPs2r);
HGEMM_FUNC(wmmaAsyncStage2);
HGEMM_FUNC(wmmaAsyncStage3);

HGEMM_FUNC(mmaNaive);
HGEMM_FUNC(mmaBase);
HGEMM_FUNC(mmaPermuted);
HGEMM_FUNC(mmaAsync);
HGEMM_FUNC(mmaAsyncPg2s);
HGEMM_FUNC(mmaAsyncPg2sPs2r);
HGEMM_FUNC(mmaAsyncStage2);
HGEMM_FUNC(mmaAsyncStage3);
HGEMM_FUNC(mmaAsyncStage4);

DEFINE_uint32(M, 512, "M");
DEFINE_uint32(N, 2048, "N");
DEFINE_uint32(K, 1024, "K");
DEFINE_bool(enable_wmma, true, "test WMMA API");
DEFINE_bool(enable_mma, true, "test MMA PTX instruction");
DEFINE_uint32(warmup_iterations, 1, "warmup iteration numbers and average the result");
DEFINE_uint32(profiling_iterations, 10, "profiling iteration numbers and average the result");
DEFINE_uint32(sleep_duration, 100, "sleep_milliseconds between profiling");
DEFINE_bool(enable_check, false, "check the GPU result against the cublas result");
DEFINE_uint32(cpu_procs, omp_get_num_procs(), "processor num used of CPU");
DEFINE_uint32(gpu_rank, 0, "the used GPU rank");

int main(int argc, char *argv[]) {
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    omp_set_num_threads(FLAGS_cpu_procs);
    HGEMM_CHECK_CUDART_ERROR(cudaSetDevice(FLAGS_gpu_rank));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, FLAGS_gpu_rank));
    HLOG("CUDA HGEMM start with %u CPU processes on the %u-th GPU: %s", FLAGS_cpu_procs, FLAGS_gpu_rank, dev_prop.name);

    int driver_version = 0;
    int runtime_version = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaDriverGetVersion(&driver_version));
    HGEMM_CHECK_CUDART_ERROR(cudaRuntimeGetVersion(&runtime_version));
    HLOG("CUDA driver version / runtime version: %d.%d / %d.%d", driver_version / 1000, (driver_version % 100) / 10,
         runtime_version / 1000, (runtime_version % 100) / 10);
    HLOG("CUDA capability major/minor version number: %d.%d", dev_prop.major, dev_prop.minor);
    HLOG("%d multiprocessors, %d CUDA cores/MP: %d CUDA cores", dev_prop.multiProcessorCount,
         convert_SM_to_cores(dev_prop.major, dev_prop.minor),
         convert_SM_to_cores(dev_prop.major, dev_prop.minor) * dev_prop.multiProcessorCount);
    HLOG("GPU max clock rate: %.0f MHz (%0.2f GHz)", static_cast<double>(dev_prop.clockRate) * 1e-3,
         static_cast<double>(dev_prop.clockRate) * 1e-6);
    HLOG("Memory clock rate: %.0f MHz (%0.2f GHz)", static_cast<double>(dev_prop.memoryClockRate) * 1e-3,
         static_cast<double>(dev_prop.memoryClockRate) * 1e-6);
    HLOG("Memory bus width: %d-bit", dev_prop.memoryBusWidth);
    HLOG("Total amount of global memory: %.0f MBytes (%zu Bytes)",
         static_cast<double>(dev_prop.totalGlobalMem) / 1048576, dev_prop.totalGlobalMem);
    HLOG("Total amount of constant memory: %.0f KBytes (%zu Bytes)", static_cast<double>(dev_prop.totalConstMem) / 1024,
         dev_prop.totalConstMem);
    HLOG("Total amount of shared memory per block: %.0f KBytes (%zu Bytes)",
         static_cast<double>(dev_prop.sharedMemPerBlock) / 1024, dev_prop.sharedMemPerBlock);
    HLOG("Total shared memory per multiprocessor: %.0f KBytes (%zu Bytes)",
         static_cast<double>(dev_prop.sharedMemPerMultiprocessor) / 1024, dev_prop.sharedMemPerMultiprocessor);
    HLOG("L2 cache size: %.0f KBytes (%d Bytes)", static_cast<double>(dev_prop.l2CacheSize) / 1024,
         dev_prop.l2CacheSize);
    HLOG("Total number of registers available per block: %d", dev_prop.regsPerBlock);
    HLOG("Warp size: %d", dev_prop.warpSize);
    HLOG("Max number of threads per multiprocessor: %d", dev_prop.maxThreadsPerMultiProcessor);
    HLOG("Max number of threads per block: %d", dev_prop.maxThreadsPerBlock);
    HLOG("Max dimension size of a thread block (x,y,z): (%d, %d, %d)", dev_prop.maxThreadsDim[0],
         dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    HLOG("Max dimension size of a grid size (x,y,z): (%d, %d, %d)", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1],
         dev_prop.maxGridSize[2]);

    HLOG("A (%u x %u) * B (%u x %u) = C (%u x %u)", FLAGS_M, FLAGS_K, FLAGS_K, FLAGS_N, FLAGS_M, FLAGS_N);
    HLOG(
        "Profiling: enable wmma: %d, enable mma: %d, warmup iterations: %u, profiling iterations: %u, sleep duration: "
        "%u ms, enable check: %d",
        FLAGS_enable_wmma, FLAGS_enable_mma, FLAGS_warmup_iterations, FLAGS_profiling_iterations, FLAGS_sleep_duration,
        FLAGS_enable_check);

    Tester tester(FLAGS_M, FLAGS_N, FLAGS_K, FLAGS_warmup_iterations, FLAGS_profiling_iterations, FLAGS_sleep_duration,
                  FLAGS_enable_check);
    tester.evaluate(cublasTensorOp, "Cublas-Tensor-Op");
    // tester.evaluate(simtNaive, "Simt-Naive");

    if (FLAGS_enable_wmma) {
        // tester.evaluate(wmmaNaive, "Wmma-Naive");
        // tester.evaluate(wmmaBase, "Wmma-Base");
        tester.evaluate(wmmaPadding, "Wmma-Padding");
        tester.evaluate(wmmaAsync, "Wmma-Async");
        tester.evaluate(wmmaAsyncPg2s, "Wmma-Async-Pg2s");
        tester.evaluate(wmmaAsyncPg2sPs2r, "Wmma-Async-Pg2s-Ps2r");
        tester.evaluate(wmmaAsyncStage2, "Wmma-Async-Stage2");
        tester.evaluate(wmmaAsyncStage3, "Wmma-Async-Stage3");
    }

    if (FLAGS_enable_mma) {
        // tester.evaluate(mmaNaive, "Mma-Naive");
        // tester.evaluate(mmaBase, "Mma-Base");
        tester.evaluate(mmaPermuted, "Mma-Permuted");
        tester.evaluate(mmaAsync, "Mma-Async");
        tester.evaluate(mmaAsyncPg2s, "Mma-Async-Pg2s");
        tester.evaluate(mmaAsyncPg2sPs2r, "Mma-Async-Pg2s-Ps2r");
        tester.evaluate(mmaAsyncStage2, "Mma-Async-Stage2");
        tester.evaluate(mmaAsyncStage3, "Mma-Async-Stage3");
        tester.evaluate(mmaAsyncStage4, "Mma-Async-Stage4");
    }

    GFLAGS_NAMESPACE::ShutDownCommandLineFlags();

    HLOG("Done");

    return 0;
}
