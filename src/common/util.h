// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: util function

#pragma once

#include "cuda_runtime_api.h"

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Beginning of GPU Architecture definitions
inline int convert_SM_to_cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
                                       {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
                                       {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
                                       {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    HLOG("MapSMtoCores for SM %d.%d is undefined. Default to use %d cores/SM", major, minor,
         nGpuArchCoresPerSM[index - 1].cores);

    return nGpuArchCoresPerSM[index - 1].cores;
}

inline const char *convert_SM_to_arch_name(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine the GPU Arch name)
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        const char *name;
    } sSMtoArchName;

    sSMtoArchName nGpuArchNameSM[] = {{0x30, "Kepler"},  {0x32, "Kepler"},  {0x35, "Kepler"},       {0x37, "Kepler"},
                                      {0x50, "Maxwell"}, {0x52, "Maxwell"}, {0x53, "Maxwell"},      {0x60, "Pascal"},
                                      {0x61, "Pascal"},  {0x62, "Pascal"},  {0x70, "Volta"},        {0x72, "Xavier"},
                                      {0x75, "Turing"},  {0x80, "Ampere"},  {0x86, "Ampere"},       {0x87, "Ampere"},
                                      {0x89, "Ada"},     {0x90, "Hopper"},  {-1, "Graphics Device"}};

    int index = 0;

    while (nGpuArchNameSM[index].SM != -1) {
        if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchNameSM[index].name;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    HLOG("MapSMtoArchName for SM %d.%d is undefined. Default to use %s", major, minor, nGpuArchNameSM[index - 1].name);

    return nGpuArchNameSM[index - 1].name;
}
