#!/usr/bin/env python3
"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import argparse
import subprocess
import sys
import threading
import os
import shlex
import time

import torch


class GPU():
    def __init__(self, id, manager):
        self.id = id
        self.manager = manager

    def release(self):
        self.manager.release_gpu(self)

    def __str__(self):
        return str(self.id)


class GPUManager():
    """ Track available GPUs and provide on request
    """
    def __init__(self, available_gpus):
        self.total_gpus = len(available_gpus)
        self.semaphore = threading.BoundedSemaphore(self.total_gpus)
        self.gpu_list_lock = threading.Lock()
        self.available_gpus = list(available_gpus)

    def get_gpu(self):
        self.semaphore.acquire()
        with self.gpu_list_lock:
            gpu = self.available_gpus.pop()
        return GPU(gpu, self)

    def get_gpus(self, num_gpu=1):
        gpu_list = []
        for ii in range(num_gpu):
            gpu_list.append(self.get_gpu())
        return gpu_list

    def release_gpu(self, gpu):
        with self.gpu_list_lock:
            self.available_gpus.append(gpu.id)
            self.semaphore.release()

    def block(self):
        # Block until all gpus are free
        while True:
            with self.gpu_list_lock:
                if len(self.available_gpus) == self.total_gpus:
                    return
            time.sleep(1.)

def run_command_with_gpus(command, gpu_list):
    print(f'GPU {",".join([str(gpu) for gpu in gpu_list])}: {command}')

    def run_and_release(command, gpu_list):
        myenv = os.environ.copy()
        myenv['CUDA_VISIBLE_DEVICES'] = ",".join([str(gpu) for gpu in gpu_list])
        proc = subprocess.Popen(args=command,
                                shell=True,
                                env=myenv)
        proc.wait()
        for gpu in gpu_list:
            gpu.release()

    thread = threading.Thread(target=run_and_release,
                              args=(command, gpu_list))
    thread.start()
    return thread


def run_command_list(manager, command_list, num_gpu):
    for command in command_list:
        if command == "":
            continue
        elif command[0] == '#':
            continue
        elif command.lower() == 'block':
            manager.block()
        else:
            gpu_list = manager.get_gpus(num_gpu=num_gpu)
            run_command_with_gpus(command, gpu_list)


def read_commands(exp_file):
    with open(exp_file, 'r') as f:
        command_list = [line.strip() for line in f]
    return command_list


def expand_repeats(command_list, start_number):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-repeats', type=int, default=None)
    parser.add_argument('--exp-name', type=str, default=None)

    out_commands = []
    for command in command_list:
        args, _ = parser.parse_known_args(shlex.split(command))
        if args.num_repeats is None:
            out_commands.append(command)
        else:
            if args.exp_name is None:
                raise Exception('ERROR: When using repeats, --exp-name is required')
            for repeat_it in range(args.num_repeats):
                new_command = command
                exp_num = repeat_it + start_number
                new_command += " --exp-name " + args.exp_name + "-" + str(exp_num)
                new_command += " --random-seed " + str(exp_num)
                out_commands.append(new_command)
    return out_commands


def main():
    parser = argparse.ArgumentParser(description='Schedule a list of GPU experiments.')
    parser.add_argument('-e', '--exp-txt', type=str, required=True,
                        help='txt file with one line per command, see e.g. exp/example.txt')
    parser.add_argument('-g', '--gpus', nargs='+', type=str, default=[], required=False, help='which GPUs to use. If unset, will use all')
    parser.add_argument('-s', '--start-number', type=int, default=0, help='starting number for random seed and experiment name for multi-run experiments (default 0)')
    parser.add_argument('-n', '--num-gpu', type=int, default=1, help='number of GPUs to use per experiment (default 1)')
    args = parser.parse_args()

    gpus = args.gpus
    if len(gpus) == 0:
        # find all available gpus
        gpus = [str(x) for x in range(torch.cuda.device_count())]

    manager = GPUManager(gpus)
    exp_file = args.exp_txt
    command_list = read_commands(exp_file)
    command_list = expand_repeats(command_list, args.start_number)
    run_command_list(manager, command_list, args.num_gpu)


if __name__ == '__main__':
    main()

