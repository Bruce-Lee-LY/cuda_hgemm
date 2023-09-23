# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: run sample script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

rm -rf log ncu && mkdir -p log ncu

# $1: M, $2: N, $3: K
evaluate_hgemm() {
    echo "Evaluating $1 * $2 * $3"
    $WORK_PATH/output/bin/hgemm -M=$1 -N=$2 -K=$3 -enable_wmma=true -enable_mma=true -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > log/hgemm_${1}_${2}_${3}.log 2>&1
    sleep 3
}

# $1: M, $2: N, $3: K
ncu_hgemm() {
    echo "NCU $1 * $2 * $3"
    sudo ncu --set full --target-processes all --force-overwrite -o ncu/hgemm_${1}_${2}_${3} $WORK_PATH/output/bin/hgemm -M=$1 -N=$2 -K=$3 -enable_wmma=true -enable_mma=true -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_hgemm_${1}_${2}_${3}.log 2>&1
    sleep 3
}

benchmark_hgemm() {
    dims=(256 512 768 1024 1536 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384)

    # M == N == K
    for M in ${dims[@]};
    do
        evaluate_hgemm $M $M $M
        # ncu_hgemm $M $M $M
    done

    # M == N > K
    # for M in ${dims[@]:(${#dims[@]} / 2):(${#dims[@]} / 2)};
    # do
    #     for K in ${dims[@]:0:(${#dims[@]} / 2)};
    #     do
    #         evaluate_hgemm $M $M $K
    #         # ncu_hgemm $M $M $K
    #     done
    # done

    # M == N < K
    # for M in ${dims[@]:0:(${#dims[@]} / 2)};
    # do
    #     for K in ${dims[@]:(${#dims[@]} / 2):(${#dims[@]} / 2)};
    #     do
    #         evaluate_hgemm $M $M $K
    #         # ncu_hgemm $M $M $K
    #     done
    # done
}

nohup $WORK_PATH/output/bin/hgemm -M=512 -N=2048 -K=1024 -enable_wmma=true -enable_mma=true -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/hgemm_512_2048_1024.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/hgemm_512_2048_1024 $WORK_PATH/output/bin/hgemm -M=512 -N=2048 -K=1024 -enable_wmma=true -enable_mma=true -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_hgemm_512_2048_1024.log 2>&1

# benchmark_hgemm
