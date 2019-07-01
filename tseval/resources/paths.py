# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent.parent.parent
RESOURCES_DIR = REPO_DIR / 'resources'
DOWNLOAD_DIR = RESOURCES_DIR / 'download'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
VARIOUS_DIR = RESOURCES_DIR / 'various'
MODELS_DIR = RESOURCES_DIR / 'models'
TOOLS_DIR = RESOURCES_DIR / 'tools'
# TODO: Move this to setup or add the folders to the git repo
for dir_path in [DOWNLOAD_DIR, DATASETS_DIR, VARIOUS_DIR, MODELS_DIR, TOOLS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)
FASTTEXT_EMBEDDINGS_PATH = VARIOUS_DIR / 'fasttext-vectors/cc.en.300.vec'
TERP_DIR = TOOLS_DIR / 'terp'
TERP_PATH = TERP_DIR / 'bin/terp'
QUEST_DIR = TOOLS_DIR / 'questplusplus'
WORDNET_DIR = TOOLS_DIR / 'WordNet-3.0'


def get_dataset_dir(dataset):
    return DATASETS_DIR / dataset


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
