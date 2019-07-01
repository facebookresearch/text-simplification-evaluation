# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import Counter
import itertools
from functools import lru_cache
import os

import Levenshtein
from nlgeval import NLGEval
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.base import TransformerMixin
import torch
import torch.nn.functional as F

from tseval.embeddings import to_embeddings
from tseval.evaluation.readability import sentence_fre, sentence_fkgl
from tseval.evaluation.terp import get_terp_vectorizers
from tseval.evaluation.quest import get_quest_vectorizers
from tseval.resources.paths import VARIOUS_DIR, FASTTEXT_EMBEDDINGS_PATH
from tseval.resources.prepare import _prepare_fasttext_embeddings
from tseval.text import (count_words, count_sentences, to_words, count_syllables_in_sentence, remove_stopwords,
                         remove_punctuation_tokens)
from tseval.models.language_models import average_sentence_lm_prob, min_sentence_lm_prob
from tseval.utils import yield_lines


@lru_cache(maxsize=1)
def get_word2concreteness():
    concrete_words_path = os.path.join(VARIOUS_DIR, 'concrete_words.tsv')
    df = pd.read_csv(concrete_words_path, sep='\t')
    df = df[df['Bigram'] == 0]  # Remove bigrams
    return {row['Word']: row['Conc.M'] for _, row in df.iterrows()}


@lru_cache(maxsize=1)
def get_word2frequency():
    frequency_table_path = os.path.join(VARIOUS_DIR, 'enwiki_frequency_table.tsv')
    word2frequency = {}
    for line in yield_lines(frequency_table_path):
        word, frequency = line.split('\t')
        word2frequency[word] = int(frequency)
    return word2frequency


@lru_cache(maxsize=1)
def get_word2rank(vocab_size=50000):
    if not FASTTEXT_EMBEDDINGS_PATH.exists():
        _prepare_fasttext_embeddings()
    word2rank = {}
    line_generator = yield_lines(FASTTEXT_EMBEDDINGS_PATH)
    next(line_generator)  # Skip the first line (header)
    for i, line in enumerate(line_generator):
        if (i+1) > vocab_size:
            break
        word = line.split(' ')[0]
        word2rank[word] = i
    return word2rank


def get_concreteness(word):
    # TODO: Default value is arbitrary
    return get_word2concreteness().get(word, 5)


def get_frequency(word):
    return get_word2frequency().get(word, None)


def get_negative_frequency(word):
    return -get_frequency(word)


def get_rank(word):
    return get_word2rank().get(word, len(get_word2rank()))


def get_negative_log_frequency(word):
    return -np.log(1 + get_frequency(word))


def get_log_rank(word):
    return np.log(1 + get_rank(word))


# Single sentence feature extractors with signature method(sentence) -> float
def get_concreteness_scores(sentence):
    return np.log(1 + np.array([get_concreteness(word) for word in to_words(sentence)]))


def get_frequency_table_ranks(sentence):
    return np.log(1 + np.array([get_rank(word) for word in to_words(sentence)]))


def get_wordrank_score(sentence):
    # Computed as the third quartile of log ranks
    words = to_words(remove_stopwords(remove_punctuation_tokens(sentence)))
    words = [word for word in words if word in get_word2rank()]
    if len(words) == 0:
        return np.log(1 + len(get_word2rank()))  # TODO: This is completely arbitrary
    return np.quantile([get_log_rank(word) for word in words], 0.75)


def count_characters(sentence):
    return len(sentence)


def safe_division(a, b):
    if b == 0:
        return b
    return a / b


def count_words_per_sentence(sentence):
    return safe_division(count_words(sentence), count_sentences(sentence))


def count_characters_per_sentence(sentence):
    return safe_division(count_characters(sentence), count_sentences(sentence))


def count_syllables_per_sentence(sentence):
    return safe_division(count_syllables_in_sentence(sentence), count_sentences(sentence))


def count_characters_per_word(sentence):
    return safe_division(count_characters(sentence), count_words(sentence))


def count_syllables_per_word(sentence):
    return safe_division(count_syllables_in_sentence(sentence), count_words(sentence))


def max_pos_in_freq_table(sentence):
    return max(get_frequency_table_ranks(sentence))


def average_pos_in_freq_table(sentence):
    return np.mean(get_frequency_table_ranks(sentence))


def min_concreteness(sentence):
    return min(get_concreteness_scores(sentence))


def average_concreteness(sentence):
    return np.mean(get_concreteness_scores(sentence))


# OPTIMIZE: Optimize feature extractors? A lot of computation is duplicated (e.g. to_words)
sentence_feature_extractors = [
    count_words,
    count_characters,
    count_sentences,
    count_syllables_in_sentence,
    count_words_per_sentence,
    count_characters_per_sentence,
    count_syllables_per_sentence,
    count_characters_per_word,
    count_syllables_per_word,
    max_pos_in_freq_table,
    average_pos_in_freq_table,
    min_concreteness,
    average_concreteness,
    sentence_fre,
    sentence_fkgl,
    average_sentence_lm_prob,
    min_sentence_lm_prob,
]


# Sentence pair feature extractors with signature method(complex_sentence, simple_sentence) -> float
def count_sentence_splits(complex_sentence, simple_sentence):
    return safe_division(count_sentences(complex_sentence), count_sentences(simple_sentence))


def get_compression_ratio(complex_sentence, simple_sentence):
    return safe_division(count_characters(simple_sentence), count_characters(complex_sentence))


def word_intersection(complex_sentence, simple_sentence):
    complex_words = to_words(complex_sentence)
    simple_words = to_words(simple_sentence)
    nb_common_words = len(set(complex_words).intersection(set(simple_words)))
    nb_max_words = max(len(set(complex_words)), len(set(simple_words)))
    return nb_common_words / nb_max_words


@lru_cache(maxsize=10000)
def average_dot(complex_sentence, simple_sentence):
    complex_embeddings = to_embeddings(complex_sentence)
    simple_embeddings = to_embeddings(simple_sentence)
    return float(torch.dot(complex_embeddings.mean(dim=0), simple_embeddings.mean(dim=0)))


@lru_cache(maxsize=10000)
def average_cosine(complex_sentence, simple_sentence):
    complex_embeddings = to_embeddings(complex_sentence)
    simple_embeddings = to_embeddings(simple_sentence)
    return float(F.cosine_similarity(complex_embeddings.mean(dim=0),
                                     simple_embeddings.mean(dim=0),
                                     dim=0))


@lru_cache(maxsize=10000)
def hungarian_dot(complex_sentence, simple_sentence):
    complex_embeddings = to_embeddings(complex_sentence)
    simple_embeddings = to_embeddings(simple_sentence)
    similarity_matrix = torch.mm(complex_embeddings, simple_embeddings.t())
    row_indexes, col_indexes = linear_sum_assignment(-similarity_matrix)
    # TODO: Penalize less deletion of unimportant words
    return float(similarity_matrix[row_indexes, col_indexes].sum() / max(len(complex_sentence), len(simple_sentence)))


@lru_cache(maxsize=10000)
def hungarian_cosine(complex_sentence, simple_sentence):
    complex_embeddings = to_embeddings(complex_sentence)
    simple_embeddings = to_embeddings(simple_sentence)
    similarity_matrix = torch.zeros(len(complex_embeddings), len(simple_embeddings))
    for (i, complex_embedding), (j, simple_embedding) in itertools.product(enumerate(complex_embeddings),
                                                                           enumerate(simple_embeddings)):
        similarity_matrix[i, j] = F.cosine_similarity(complex_embedding, simple_embedding, dim=0)
    row_indexes, col_indexes = linear_sum_assignment(-similarity_matrix)
    # TODO: Penalize less deletion of unimportant words
    return float(similarity_matrix[row_indexes, col_indexes].sum() / max(len(complex_sentence), len(simple_sentence)))


def characters_per_sentence_difference(complex_sentence, simple_sentence):
    return count_characters_per_sentence(complex_sentence) - count_characters_per_sentence(simple_sentence)


def is_exact_match(complex_sentence, simple_sentence):
    return complex_sentence == simple_sentence


def get_levenshtein_similarity(complex_sentence, simple_sentence):
    return Levenshtein.ratio(complex_sentence, simple_sentence)


def get_levenshtein_distance(complex_sentence, simple_sentence):
    return 1 - get_levenshtein_similarity(complex_sentence, simple_sentence)


def get_additions_proportion(complex_sentence, simple_sentence):
    n_additions = sum((Counter(to_words(simple_sentence)) - Counter(to_words(complex_sentence))).values())
    return n_additions / max(count_words(complex_sentence), count_words(simple_sentence))


def get_deletions_proportion(complex_sentence, simple_sentence):
    n_deletions = sum((Counter(to_words(complex_sentence)) - Counter(to_words(simple_sentence))).values())
    return n_deletions / max(count_words(complex_sentence), count_words(simple_sentence))


# Making one call to nlgeval returns all metrics, we therefore cache the results in order to limit the number of calls
@lru_cache(maxsize=10000)
def get_all_nlgeval_metrics(complex_sentence, simple_sentence):
    if 'NLGEVAL' not in globals():
        global NLGEVAL
        print('Loading NLGEval models...')
        # Change False to True if you want to use skipthought or glove
        NLGEVAL = NLGEval(no_skipthoughts=True, no_glove=True)
        print('Done.')
    return NLGEVAL.compute_individual_metrics([complex_sentence], simple_sentence)


def get_nlgeval_methods():
    """Returns all scoring methods from nlgeval package.

    Signature: method(complex_sentence, simple_setence)
    """
    def get_scoring_method(metric_name):
        """Necessary to wrap the scoring_method() in get_scoring_method(), in order to set the external variable to
        its current value."""
        def scoring_method(complex_sentence, simple_sentence):
            return get_all_nlgeval_metrics(complex_sentence, simple_sentence)[metric_name]
        return scoring_method

    nlgeval_metrics = [
        # Fast metrics
        'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr',
        # Slow metrics
        # 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore',
    ]
    methods = []
    for metric_name in nlgeval_metrics:
        scoring_method = get_scoring_method(metric_name)
        scoring_method.__name__ = f'nlgeval_{metric_name}'
        methods.append(scoring_method)
    return methods


def get_nltk_bleu_methods():
    """Returns bleu methods with different smoothings from NLTK.
Signature: scoring_method(complex_sentence, simple_setence)
    """
    def get_scoring_method(smoothing_function):
        """Necessary to wrap the scoring_method() in get_scoring_method(), in order to set the external variable to
        its current value."""
        def scoring_method(complex_sentence, simple_sentence):
            try:
                return sentence_bleu([complex_sentence.split()], simple_sentence.split(),
                                     smoothing_function=smoothing_function)
            except AssertionError as e:
                return 0
        return scoring_method

    methods = []
    for i in range(8):
        smoothing_function = getattr(SmoothingFunction(), f'method{i}')
        scoring_method = get_scoring_method(smoothing_function)
        scoring_method.__name__ = f'nltkBLEU_method{i}'
        methods.append(scoring_method)
    return methods


sentence_pair_feature_extractors = [
    word_intersection,
    characters_per_sentence_difference,
    average_dot,
    average_cosine,
    hungarian_dot,
    hungarian_cosine,
] + get_nlgeval_methods() + get_nltk_bleu_methods() + get_terp_vectorizers() + get_quest_vectorizers()


# Various
def wrap_single_sentence_vectorizer(vectorizer):
    '''Transform a single sentence vectorizer to a sentence pair vectorizer

    Change the signature of the input vectorizer
    Initial signature: method(simple_sentence)
    New signature: method(complex_sentence, simple_sentence)
    '''
    def wrapped(complex_sentence, simple_sentence):
        return vectorizer(simple_sentence)

    wrapped.__name__ = vectorizer.__name__
    return wrapped


def reverse_vectorizer(vectorizer):
    '''Reverse the arguments of a vectorizer'''
    def reversed_vectorizer(complex_sentence, simple_sentence):
        return vectorizer(simple_sentence, complex_sentence)

    reversed_vectorizer.__name__ = vectorizer.__name__ + '_reversed'
    return reversed_vectorizer


def get_all_vectorizers(reversed=False):
    vectorizers = [wrap_single_sentence_vectorizer(vectorizer)
                   for vectorizer in sentence_feature_extractors] + sentence_pair_feature_extractors
    if reversed:
        vectorizers += [reverse_vectorizer(vectorizer) for vectorizer in vectorizers]
    return vectorizers


def concatenate_corpus_vectorizers(vectorizers):
    '''Given a list of corpus vectorizers, create a new single concatenated corpus vectorizer.

    Corpus vectorizer:
    Given a numpy array of shape (n_samples, 2), it will extract features for each sentence pair
    and output a (n_samples, n_features) array.
    '''
    def concatenated(sentence_pairs):
        return np.column_stack([vectorizer(sentence_pairs) for vectorizer in vectorizers])
    return concatenated


class FeatureSkewer(TransformerMixin):
    '''Normalize features that have a skewed distribution'''
    def fit(self, X, y):
        self.skewed_indexes = [i for i in range(X.shape[1]) if skew(X[:, i]) > 0.75]
        return self

    def transform(self, X):
        for i in self.skewed_indexes:
            X[:, i] = boxcox1p(X[:, i], 0)
        return np.nan_to_num(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
