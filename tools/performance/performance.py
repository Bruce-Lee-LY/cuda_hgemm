# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: performance line chart

# !/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
import optparse
import numpy as np
import matplotlib.pyplot as plt


def get_methods(log_file):
    methods = []

    with open(log_file) as fp:
        line = fp.readline()
        while line:
            if 'exit' in line and 'Naive' not in line:
                iterms = line.split(' ')
                methods.append(iterms[6])

            line = fp.readline()

    return methods


def get_dims(log_files):
    dims = []

    for log_file in log_files:
        dims.append(int((log_file.split('.')[0]).split('_')[-1]))

    dims.sort()

    return dims


def read_data(methods, dims, data_path, log_files):
    data_throughput = np.zeros((len(methods), len(dims)), np.float64)
    data_performance = np.zeros((len(methods), len(dims)), np.float64)

    for log_file in log_files:
        dim = int((log_file.split('.')[0]).split('_')[-1])
        with open(data_path + log_file) as fp:
            line = fp.readline()
            while line:
                if 'exit' in line and 'Naive' not in line:
                    iterms = line.split(' ')
                    method = iterms[6]
                    data_throughput[methods.index(
                        method)][dims.index(dim)] = float(iterms[14])
                    data_performance[methods.index(method)][dims.index(dim)] = float(
                        iterms[16].replace('(', '').replace(')', '').replace('%', ''))

                line = fp.readline()

    return data_throughput, data_performance


def draw_line_chart(methods, dims, data, figure_name, y_label, title):
    fig = plt.figure(figsize=(32, 24), dpi=100)

    dims_str = list(map(str, dims))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']

    for i in range(len(methods)):
        plt.plot(dims_str, data[i], color=colors[i % len(colors)],
                 linestyle=linestyles[(i // len(colors)) % len(linestyles)], marker='o', markersize=6)

    # plt.xticks(dims)
    plt.ylim(bottom=0)
    plt.yticks(range(0, round(np.max(np.max(data, axis=0)) + 0.5) + 10, 10))
    plt.tick_params(labelsize=25)

    # plt.hlines(y=100, xmin=dims_str[0], xmax=dims_str[-1], colors='r', linestyles='-.')
    plt.grid(True, linestyle='-.')

    plt.xlabel('Matrix Dimension / M = N = K', fontdict={'size': '30'})
    plt.ylabel(y_label, fontdict={'size': '30'})
    plt.title(title, fontdict={'size': '30'})
    plt.legend(methods, loc='best', prop={'size': '30'})

    plt.savefig(figure_name, dpi=fig.dpi)
    # plt.show()


def analyze_data(data_path):
    log_files = []
    for file_name in os.listdir(data_path):
        if '.log' not in file_name:
            continue

        log_files.append(file_name)

    methods = get_methods(data_path + log_files[0])
    dims = get_dims(log_files)
    data_throughput, data_performance = read_data(
        methods, dims, data_path, log_files)
    draw_line_chart(methods, dims, data_throughput, data_path +
                    'throughput.png', 'Throughput / TFLOPS', 'HGEMM Throughput')
    draw_line_chart(methods, dims, data_performance, data_path + 'performance.png',
                    'Performance Compared with Cublas / %', 'HGEMM Performance')


def main():
    usage = "python3 performance.py -p/--path log/"
    parser = optparse.OptionParser(usage)
    parser.add_option('-p', '--path', dest='path',
                      type='string', help='data path', default='log/')

    options, args = parser.parse_args()
    path = options.path

    analyze_data(path)


if __name__ == "__main__":
    main()
