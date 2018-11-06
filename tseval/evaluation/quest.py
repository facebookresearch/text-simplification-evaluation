# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Quest features from https://github.com/ghpaetzold/questplusplus
# Lucia Specia, Gustavo Henrique Paetzold and Carolina Scarton (2015):
# Multi-level Translation Quality Prediction with QuEst++.
# In Proceedings of ACL-IJCNLP 2015 System Demonstrations, Beijing, China, pp. 115-120.
import os

import numpy as np
import pandas as pd

from tseval.qats import get_qats_train_data, get_qats_test_data
from tseval.utils import numpy_memoize, write_lines, run_command
from tseval.resources.paths import QUEST_DIR


quest_features = [
    'nb_source_tokens',  # number of tokens in the source sentence
    'nb_target_tokens',  # number of tokens in the target sentence
    'avg_source_token_length',  # average source token length
    'lm_prob_source',  # LM probability of source sentence
    'lm_prob_target',  # LM probability of target sentence
    'type_token_ratio',  # number of occurrences of the target word within the target hypothesis (averaged for all words in the hypothesis - type/token ratio)'  # noqa: E501
    'nb_translations',  # average number of translations per source word in the sentence (as given by IBM 1 table thresholded such that prob(t|s) > 0.2)'  # noqa: E501
    'nb_translations_idf',  # average number of translations per source word in the sentence (as given by IBM 1 table thresholded such that prob(t|s) > 0.01) weighted by the inverse frequency of each word in the source corpus'  # noqa: E501
    'nb_unigram_q1_freq',  # percentage of unigrams in quartile 1 of frequency (lower frequency words) in a corpus of the source language (SMT training corpus)'  # noqa: E501
    'nb_unigram_q4_freq',  # percentage of unigrams in quartile 4 of frequency (higher frequency words) in a corpus of the source language  # noqa: E501
    'nb_bigram_q1_freq',  # percentage of bigrams in quartile 1 of frequency of source words in a corpus of the source language  # noqa: E501
    'nb_bigram_q4_freq',  # percentage of bigrams in quartile 4 of frequency of source words in a corpus of the source language  # noqa: E501
    'nb_trigram_q1_freq',  # percentage of trigrams in quartile 1 of frequency of source words in a corpus of the source language  # noqa: E501
    'nb_trigram_q4_freq',  # percentage of trigrams in quartile 4 of frequency of source words in a corpus of the source language  # noqa: E501
    'nb_source_words_in_corpus',  # percentage of unigrams in the source sentence seen in a corpus (SMT training corpus)
    'nb_source_punct',  # number of punctuation marks in the source sentence
    'nb_target_punct',  # number of punctuation marks in the target sentence
]


@numpy_memoize
def get_quest_features(sentence_pairs):
    # HACK: quick & dirty method
    source_filepath = os.path.join(QUEST_DIR, 'input/source.qats.en')
    target_filepath = os.path.join(QUEST_DIR, 'input/target.qats.en')
    output_filepath = os.path.join(QUEST_DIR, 'output/test/output.txt')
    write_lines(sentence_pairs[:, 0], source_filepath)
    write_lines(sentence_pairs[:, 1], target_filepath)
    cmd = f'cd {QUEST_DIR}'
    cmd += ' && java -cp QuEst++.jar shef.mt.SentenceLevelFeatureExtractor -tok -case true -lang english english'
    cmd += f' -input {source_filepath} {target_filepath} -config config/config.sentence-level.properties'
    # TODO: Fix LM features
    run_command(cmd)
    return np.nan_to_num(pd.read_csv(output_filepath, header=None, sep='\t').values)


def get_quest_features_on_qats_pair(complex_sentence, simple_sentence):
    # Computing features on a single sentence pair is as long as computing all sentence pairs in QATS at once
    if 'QATS_QUEST_FEATURES' not in globals():
        print('Computing QuEst features on all QATS sentence pairs.')
        global QATS_QUEST_FEATURES
        train_sentences, _ = get_qats_train_data('simplicity')
        test_sentences, _ = get_qats_test_data('simplicity')
        sentences = np.concatenate([train_sentences, test_sentences])
        sentences = np.concatenate([sentences, np.flip(sentences, axis=1)])
        quest_features = get_quest_features(sentences)
        QATS_QUEST_FEATURES = {tuple(sentence_pair): features
                               for sentence_pair, features in zip(sentences, quest_features)}
        print('Done.')
    assert (complex_sentence, simple_sentence) in QATS_QUEST_FEATURES, 'Sentence pair is not in QATS.'
    return QATS_QUEST_FEATURES[(complex_sentence, simple_sentence)]


def get_quest_vectorizers():
    if not os.path.exists(QUEST_DIR):
        # Extract the quest archive file to QUEST_DIR:
        url = 'https://www.quest.dcs.shef.ac.uk/quest_files/questplusplus-vanilla.tar.gz'
        print(f'In order to use QuEst please install it to {QUEST_DIR} from {url}')
        return []

    def get_scoring_method(i):
        """Necessary to wrap the scoring_method() in get_scoring_method(), in order to set the external variable to
        its current value."""
        def scoring_method(complex_sentence, simple_sentence):
            return get_quest_features_on_qats_pair(complex_sentence, simple_sentence)[i]
        return scoring_method

    vectorizers = []
    for i, quest_feature in enumerate(quest_features):
        vectorizer = get_scoring_method(i)
        vectorizer.__name__ = f'QuEst_{quest_feature}'
        vectorizers.append(vectorizer)
    return vectorizers
