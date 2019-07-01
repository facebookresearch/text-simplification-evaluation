# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os


REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
RESOURCES_DIR = os.path.join(REPO_DIR, 'resources')
DOWNLOAD_DIR = os.path.join(RESOURCES_DIR, 'download')
DATASETS_DIR = os.path.join(RESOURCES_DIR, 'datasets')
VARIOUS_DIR = os.path.join(RESOURCES_DIR, 'various')
MODELS_DIR = os.path.join(RESOURCES_DIR, 'models')
TOOLS_DIR = os.path.join(RESOURCES_DIR, 'tools')
# TODO: Move this to setup or add the folders to the git repo
for dir_path in [DOWNLOAD_DIR, DATASETS_DIR, VARIOUS_DIR, MODELS_DIR, TOOLS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
FASTTEXT_EMBEDDINGS_PATH = os.path.join(VARIOUS_DIR, 'fasttext-vectors', 'cc.en.300.vec')
TERP_DIR = os.path.join(TOOLS_DIR, 'terp')
TERP_PATH = os.path.join(TERP_DIR, 'bin/terp')
QUEST_DIR = os.path.join(TOOLS_DIR, 'questplusplus')
WORDNET_DIR = os.path.join(TOOLS_DIR, 'WordNet-3.0')


def get_dataset_dir(dataset):
    return os.path.join(DATASETS_DIR, dataset)


def get_data_file_path(dataset, phase, language, i=None):
    suffix = ''  # Create suffix e.g. for multiple references
    if i is not None:
        suffix = f'.{i}'
    assert phase in ['train', 'valid', 'test']
    filename = f'{dataset}.{phase}.{language}{suffix}'
    dataset_dir = get_dataset_dir(dataset)
    if not os.path.exists(dataset_dir):
        print(f'Creating {dataset_dir}')
        os.makedirs(dataset_dir)
    return os.path.join(dataset_dir, filename)
