# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pandas as pd

from utils import mock_fairseq
mock_fairseq()  # noqa: E402
from tseval.qats import evaluate_scoring_method_on_qats
from tseval.feature_extraction import get_all_vectorizers


def test_methods_comparison():
    metric_name = 'pearson'
    aspect = 'simplicity'
    df = pd.DataFrame(columns=['team', f'valid_{metric_name}', f'{metric_name}'])
    for vectorizer in get_all_vectorizers():
        df = df.append(evaluate_scoring_method_on_qats(aspect, vectorizer), ignore_index=True)
        df[f'valid_{metric_name}_abs'] = df[f'valid_{metric_name}'].abs()
        df = df.sort_values(by=f'valid_{metric_name}_abs', ascending=False)
    assert 'count_characters_per_sentence' in df.head()['team'].values
