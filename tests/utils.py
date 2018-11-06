# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys


def mock_fairseq():
    '''Mock sys.stdin.fileno (or fairseq.multiprocessing_pdb will crash when used in pytest)'''
    def mocked_fileno():
        return 1
    sys.stdin.fileno = mocked_fileno
