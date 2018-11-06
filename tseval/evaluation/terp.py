# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
import re
import tempfile

import numpy as np

from tseval.qats import get_qats_train_data, get_qats_test_data
from tseval.resources.paths import TERP_PATH, TERP_DIR
from tseval.utils import run_command, write_lines, numpy_memoize


def write_lines_to_trans_format(lines, output_filepath):
    with Path(output_filepath).open('w') as output_file:
        for i, l in enumerate(lines):
            output_file.write(l.rstrip('\n') + f' ([sys][doc][{i}])\n')


def text_to_trans_format(input_filepath, output_filepath):
    def line_generator(input_filepath):
        with open(input_filepath, 'r') as input_file:
            for l in input_file:
                yield l

    write_lines_to_trans_format(line_generator(input_filepath), output_filepath)


def terp(reference_filepath, hypothesis_filepath, output_dir=tempfile.mkdtemp()):
    # Convert files to trans format
    tmp_hypothesis_filepath = Path(output_dir) / 'tmp_hyp_file.txt'
    text_to_trans_format(hypothesis_filepath, tmp_hypothesis_filepath)
    tmp_reference_filepath = Path(output_dir) / 'tmp_ref_file.txt'
    text_to_trans_format(reference_filepath, tmp_reference_filepath)
    # Compute terp
    cmd = f'cd {output_dir}; {TERP_PATH} -r {tmp_reference_filepath} -h {tmp_hypothesis_filepath}; cd -;'
    stdout = run_command(cmd)
    m = re.search(r'Total TER: (\d+\.\d+) \(', stdout)
    return float(m.groups()[0])


terp_features = ['Ins', 'Del', 'Sub', 'Stem', 'Syn', 'Phrase', 'Shft', 'WdSh', 'NumEr', 'NumWd', 'TERp']


def parse_terp_file(filepath):
    '''Parse the output terp.sum file.

    Format:
    ID             | Ins    | Del    | Sub    | Stem   | Syn    | Phrase | Shft   | WdSh   | NumEr    | NumWd    | TERp
    -------------------------------------------------------------------------------------------------------------------------------------------------
    [sys][doc][0]  |      0 |     11 |      4 |      0 |      0 |      0 |      2 |      3 |   17.000 |   19.000 |   89
    [sys][doc][1]  |      0 |     15 |      1 |      0 |      0 |      0 |      0 |      0 |   16.000 |   23.000 |   69
    '''
    with open(filepath, 'r') as f:
        line_id = 0
        features = []
        for line in f:
            m = re.match(r'\[sys\]\[doc\]\[(\d+)\] +\|(.*)', line)
            if m is None:
                continue
            assert line_id == int(m.groups()[0])
            line_id += 1
            features.append([float(val) for val in m.groups()[1].split('|')])
    return np.array(features)


@numpy_memoize
def get_terp_features(sentence_pairs):
    '''Computes intermediary terp features given a numpy array of shape (n_samples, 2) with the input sentences'''
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        complex_filepath = tmp_dir / 'complex.txt'
        simple_filepath = tmp_dir / 'simple.txt'
        write_lines(sentence_pairs[:, 0], complex_filepath)
        write_lines(sentence_pairs[:, 1], simple_filepath)
        terp(complex_filepath, simple_filepath, output_dir=tmp_dir)
        return parse_terp_file(tmp_dir / 'terp.sum')


def get_terp_features_on_qats_pair(complex_sentence, simple_sentence):
    # Computing features on a single sentence pair is as long as computing all sentence pairs in QATS at once
    if 'QATS_TERP_FEATURES' not in globals():
        print('Computing TERp features on all QATS sentence pairs.')
        global QATS_TERP_FEATURES
        train_sentences, _ = get_qats_train_data('simplicity')
        test_sentences, _ = get_qats_test_data('simplicity')
        sentences = np.concatenate([train_sentences, test_sentences])
        sentences = np.concatenate([sentences, np.flip(sentences, axis=1)])
        terp_features = get_terp_features(sentences)
        QATS_TERP_FEATURES = {tuple(sentence_pair): features
                              for sentence_pair, features in zip(sentences, terp_features)}
        print('Done.')
    assert (complex_sentence, simple_sentence) in QATS_TERP_FEATURES, 'Sentence pair is not in QATS.'
    return QATS_TERP_FEATURES[(complex_sentence, simple_sentence)]


def get_terp_vectorizers():
    if not Path(TERP_DIR).exists():
        print(f'In order to use TERp please install it to {TERP_DIR} from https://github.com/snover/terp')
        return []

    def get_scoring_method(i):
        """Necessary to wrap the scoring_method() in get_scoring_method(), in order to set the external variable to
        its current value."""
        def scoring_method(complex_sentence, simple_sentence):
            return get_terp_features_on_qats_pair(complex_sentence, simple_sentence)[i]
        return scoring_method

    vectorizers = []
    for i, terp_feature in enumerate(terp_features):
        vectorizer = get_scoring_method(i)
        vectorizer.__name__ = f'TERp_{terp_feature}'
        vectorizers.append(vectorizer)
    return vectorizers
