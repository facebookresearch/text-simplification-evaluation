# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
import subprocess

import numpy as np


def run_command(cmd):
    # HACK: This method is not secure at all
    try:
        completed_process = subprocess.run(cmd,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT,
                                           shell=True,
                                           check=True)
    except subprocess.CalledProcessError as e:
        print(e.output.decode('utf-8'))
        raise e
    return completed_process.stdout.decode().strip()


def write_lines(lines, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('w') as f:
        for line in lines:
            f.write(line + '\n')


def hash_numpy_array(x):
    assert type(x) == np.ndarray
    # Note: tobytes() Makes a copy of the array
    return hash(x.tobytes())


def numpy_memoize(f):
    '''Decorator to memoize methods that that a numpy array as input (given that numpy array are not hashable).'''
    # TODO: limit the size of memo
    memo = {}

    def helper(x):
        hashed = hash_numpy_array(x)
        if hashed not in memo:
            memo[hashed] = f(x)
        return memo[hashed]
    return helper
