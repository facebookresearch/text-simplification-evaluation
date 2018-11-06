# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from utils import mock_fairseq
mock_fairseq()  # noqa: E402
from tseval.qats import get_qats_train_data, evaluate_scoring_method_on_qats
from tseval.feature_extraction import get_all_vectorizers


def test_get_qats_train_data():
    sentences, labels = get_qats_train_data(aspect='simplicity')
    assert sentences.shape == (505, 2)
    assert labels.shape == (505,)


def test_evaluate_scoring_method():
    vectorizer = get_all_vectorizers()[0]
    metrics = evaluate_scoring_method_on_qats('simplicity', vectorizer)
    assert abs(metrics['valid_pearson']) > 0.2
