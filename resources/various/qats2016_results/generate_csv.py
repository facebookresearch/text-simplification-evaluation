# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import pandas as pd

from ts.utils import get_dataset_dir


path = os.path.join(get_dataset_dir('qats2016'), 'results/raw_scores.csv')
df = pd.read_csv(path, header=None)
metric = 'pearson'
aspect = 'grammaticality'
filepath = os.path.join(get_dataset_dir('qats2016'), 'results', f'{metric}_{aspect}.csv')
df_csv = df[[1, 0]]
df_csv.columns = ['team', metric]
df_csv.to_csv(filepath, index=None)
metric = 'pearson'
aspect = 'meaning_preservation'
filepath = os.path.join(get_dataset_dir('qats2016'), 'results', f'{metric}_{aspect}.csv')
df_csv = df[[3, 2]]
df_csv.columns = ['team', metric]
df_csv.to_csv(filepath, index=None)
metric = 'pearson'
aspect = 'simplicity'
filepath = os.path.join(get_dataset_dir('qats2016'), 'results', f'{metric}_{aspect}.csv')
df_csv = df[[5, 4]]
df_csv.columns = ['team', metric]
df_csv.to_csv(filepath, index=None)
metric = 'pearson'
aspect = 'overall'
filepath = os.path.join(get_dataset_dir('qats2016'), 'results', f'{metric}_{aspect}.csv')
df_csv = df[[7, 6]]
df_csv.columns = ['team', metric]
df_csv.to_csv(filepath, index=None)

for aspect in ['grammaticality', 'meaning_preservation', 'simplicity', 'overall']:
    path = os.path.join(get_dataset_dir('qats2016'), 'results', f'{aspect}_scores.csv')
    df = pd.read_csv(path, header=None)

    metric = 'accuracy'
    filepath = os.path.join(get_dataset_dir('qats2016'), 'results', f'{metric}_{aspect}.csv')
    df_csv = df[[1, 0]]
    df_csv.columns = ['team', metric]
    df_csv.to_csv(filepath, index=None)
    metric = 'mean_absolute_error'
    filepath = os.path.join(get_dataset_dir('qats2016'), 'results', f'{metric}_{aspect}.csv')
    df_csv = df[[3, 2]]
    df_csv.columns = ['team', metric]
    df_csv.to_csv(filepath, index=None)
    metric = 'root_mean_squared_error'
    filepath = os.path.join(get_dataset_dir('qats2016'), 'results', f'{metric}_{aspect}.csv')
    df_csv = df[[5, 4]]
    df_csv.columns = ['team', metric]
    df_csv.to_csv(filepath, index=None)
    metric = 'weighted_f_score'
    filepath = os.path.join(get_dataset_dir('qats2016'), 'results', f'{metric}_{aspect}.csv')
    df_csv = df[[7, 6]]
    df_csv.columns = ['team', metric]
    df_csv.to_csv(filepath, index=None)
