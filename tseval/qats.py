# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from tseval.resources.paths import VARIOUS_DIR, get_dataset_dir
from tseval.text import nist_tokenize


ASPECTS = ['grammaticality', 'meaning_preservation', 'simplicity', 'overall']
QATS_METRICS = [
    'pearson',
    # Drop metrics other than pearson and F-score
    # 'accuracy',
    # 'mean_absolute_error',
    # 'root_mean_squared_error',
    'weighted_f_score',
]


def pearson(y_true, y_pred):
    return scipy.stats.pearsonr(y_true, y_pred)[0]


def pearsonr_with_confidence_interval(x, y, alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''
    r, p = scipy.stats.pearsonr(x, y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(len(x) - 3)
    z = scipy.stats.norm.ppf(1 - alpha/2)
    lo_z, hi_z = r_z - z*se, r_z + z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi


def check_data():
    if not (Path(get_dataset_dir('qats2016')) / 'train.shared-task.tsv').exists():
        from tseval.resources.prepare import prepare_resource
        prepare_resource('qats2016')


def extract_data_from_dataframe(df, aspect, tokenize=True):
    assert aspect in ASPECTS, f'Aspect must be one of: {", ".join(ASPECTS)}'
    aspect = {'grammaticality': 'G', 'meaning_preservation': 'M', 'simplicity': 'S', 'overall': 'Overall'}[aspect]
    label_to_int = {'bad': 0, 'ok': 50, 'good': 100}
    sentences = df[['Original', 'Simplified']].values.astype(str)
    if tokenize:
        sentences = np.vectorize(nist_tokenize)(sentences)
    labels = np.array([label_to_int[label] for label in df[aspect].values])
    return sentences, labels


def get_qats_train_data(aspect, tokenize=True):
    check_data()
    # Data is tab separated, and we ignore quotes
    df = pd.read_csv(os.path.join(get_dataset_dir('qats2016'), 'train.shared-task.tsv'),
                     sep='\t', quoting=csv.QUOTE_NONE)
    return extract_data_from_dataframe(df, aspect, tokenize)


def get_qats_test_data(aspect, tokenize=True):
    check_data()
    df = pd.read_csv(os.path.join(get_dataset_dir('qats2016'), 'test.shared-task.tsv'),
                     sep='\t', quoting=csv.QUOTE_NONE)
    # Join labels
    df = df.join(pd.read_csv(os.path.join(get_dataset_dir('qats2016'), 'test.shared-task.labels.tsv'), sep='\t'))
    return extract_data_from_dataframe(df, aspect, tokenize)


def get_scoring_metrics(test_true_labels, test_pred_scores):
    return {'pearson': round(pearson(test_true_labels, test_pred_scores), 3)}


def get_classification_metrics(test_true_labels, test_pred_labels):
    return {
        # 'accuracy': round(100 * sklearn.metrics.accuracy_score(test_true_labels, test_pred_labels), 2),
        # 'mean_absolute_error': round(sklearn.metrics.mean_absolute_error(test_true_labels, test_pred_labels), 2),
        # TODO: Can't reproduce the task's formula for RMSE
        # 'Root mean squared error': math.sqrt(sklearn.metrics.mean_squared_error(test_true_labels,
        #                                                                         test_pred_labels)),
        'weighted_f_score': round(100 * f1_score(test_true_labels,
                                                 test_pred_labels,
                                                 average='weighted'), 2)
    }


def mean_with_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    m, se = np.mean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return m, h


def evaluate_classification_pipeline_on_qats(aspect, pipeline, method_name):
    train_sentences, train_labels = get_qats_train_data(aspect=aspect)
    valid_scores = cross_val_score(pipeline, train_sentences, train_labels, cv=10, scoring='f1_weighted')
    valid_score, cross_val_confidence_interval = mean_with_confidence_interval(valid_scores, 0.95)
    pipeline.fit(train_sentences, train_labels)
    test_sentences, test_labels = get_qats_test_data(aspect=aspect)
    test_score = f1_score(test_labels, pipeline.predict(test_sentences), average='weighted')
    return {
        'team': method_name,
        'valid_weighted_f_score': valid_score,
        # 95% confidence interval of the cross validation mean: valid_score +- condience_interval
        'valid_conf_int': cross_val_confidence_interval,
        'weighted_f_score': test_score,
    }


def evaluate_regression_pipeline_on_qats(aspect, pipeline, method_name):
    train_sentences, train_labels = get_qats_train_data(aspect=aspect)
    valid_scores = cross_val_score(pipeline, train_sentences, train_labels, cv=10, scoring=make_scorer(pearson))
    valid_score, cross_val_confidence_interval = mean_with_confidence_interval(valid_scores, 0.95)
    pipeline.fit(train_sentences, train_labels)
    test_sentences, test_labels = get_qats_test_data(aspect=aspect)
    test_score = pearson(test_labels, pipeline.predict(test_sentences))
    return {
        'team': method_name,
        'valid_pearson': valid_score,
        # 95% confidence interval of the cross validation mean: valid_score +- condience_interval
        'valid_conf_int': cross_val_confidence_interval,
        'pearson': test_score,
    }


def evaluate_scoring_method_on_qats(aspect, scoring_method):
    '''Evaluates a scoring method with signature method(complex_sentence, simple_sentence)'''
    pipeline = Pipeline([('raw_scoring', FunctionPredictor(row_vectorize(scoring_method)))])
    train_sentences, train_labels = get_qats_train_data(aspect=aspect)
    pred_train_labels = pipeline.predict(train_sentences)
    train_score, p_value, conf_int_low, conf_int_high = pearsonr_with_confidence_interval(train_labels,
                                                                                          pred_train_labels)
    test_sentences, test_labels = get_qats_test_data(aspect=aspect)
    test_score = pearson(test_labels, pipeline.predict(test_sentences))
    return {
        'team': scoring_method.__name__,
        'valid_pearson': train_score,
        'valid_p': p_value,
        # 95% Confidence interval on pearson correlation: r in [valid_conf_int_low, valid_conf_int_high]
        'valid_conf_int_low': conf_int_low,
        'valid_conf_int_high': conf_int_high,
        'pearson': test_score,
    }


def get_qats_results(aspect):
    df = pd.DataFrame(columns=['team'])
    for metric in QATS_METRICS:
        csv_path = os.path.join(VARIOUS_DIR, 'qats2016_results', f'{metric}_{aspect}.csv')
        serie = pd.read_csv(csv_path)
        df = df.merge(serie, how='outer', on='team')
    df = df.dropna(axis=0, how='all')
    return df.sort_values(by='weighted_f_score', ascending=False)


def sort_results_columnwise(df):
    dfs = []
    for metric in QATS_METRICS:
        df_metric = df[['team', metric]]
        df_metric = df_metric.sort_values(by=[metric, 'team'], ascending=[('error' in metric), True])
        df_metric = df_metric.dropna().reset_index()
        df_metric[metric] = df_metric[metric].map(lambda x: f'{x:.2f}   ') + df_metric['team']
        dfs.append(df_metric[metric])
    return pd.concat(dfs, axis=1)


def row_vectorize(func):
    """Similar to np.vectorize but on rows instead of single elements of array."""
    return lambda X: np.apply_along_axis(lambda row: np.array(func(*row)).reshape(-1), 1, X)


# TODO: This class might be better of somewhere else
class FunctionPredictor(BaseEstimator):
    '''Create an sklearn estimator from a function'''

    def __init__(self, func=None):
        self.func = func

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.func(X).flatten()
