# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path
import shutil

from tseval.utils.helpers import run_command
from .resources_utils import download, download_and_extract, move_with_overwrite, git_clone
from .paths import VARIOUS_DIR, MODELS_DIR, FASTTEXT_EMBEDDINGS_PATH, TERP_DIR, WORDNET_DIR, QUEST_DIR, get_dataset_dir


def get_available_resources():
    def is_prepare_method(variable):
        return (callable(variable) and variable.__name__.startswith('_prepare_'))
    return [name.replace('_prepare_', '') for (name, variable) in globals().items()
            if is_prepare_method(variable)]


def prepare_resource(resource_name):
    # General method to prepare a given resource
    method_name = f'_prepare_{resource_name.lower()}'
    if method_name not in globals():
        raise NameError(f'Resouce {resource_name} does not exist.\n'
                        f'Available resources are: {", ".join(get_available_resources())}.')
    return globals()[method_name]()


def _prepare_qats2016():
    qats2016_dir = get_dataset_dir('qats2016')
    if not os.path.exists(qats2016_dir):
        os.makedirs(qats2016_dir)
    url = 'http://qats2016.github.io/qats2016.github.io/train.shared-task.tsv'
    destination_path = os.path.join(qats2016_dir, 'train.shared-task.tsv')
    download(url, destination_path)
    url = 'http://qats2016.github.io/qats2016.github.io/test.shared-task.tsv'
    destination_path = os.path.join(qats2016_dir, 'test.shared-task.tsv')
    download(url, destination_path)
    url = 'http://qats2016.github.io/qats2016.github.io/test.o+g+m+s.human-labels'
    destination_path = os.path.join(qats2016_dir, 'test.shared-task.labels.tsv')
    download(url, destination_path)
    print('Done.')


def _prepare_brysbaert_concrete_words():
    # http://crr.ugent.be/archives/1330
    # The file contains 8 columns:
    # 1. The word
    # 2. Whether it is a single word or a two-word expression
    # 3. The mean concreteness rating
    # 4. The standard deviation of the concreteness ratings
    # 5. The number of persons indicating they did not know the word
    # 6. The total number of persons who rated the word
    # 7. Percentage participants who knew the word
    # 8. The SUBTLEX-US frequency count (on a total of 51 million; Brysbaert & New, 2009)
    url = 'http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt'
    download(url, os.path.join(VARIOUS_DIR, 'concrete_words.tsv'))


def _prepare_fairseq_lm():
    # Download model
    url = 'https://dl.fbaipublicfiles.com/fairseq/models/wiki103_fconv_lm.tar.bz2'
    extracted_paths = download_and_extract(url)
    output_dir = os.path.join(MODELS_DIR, 'language_models/wiki103_fconv_lm')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for extracted_path in extracted_paths:
        move_with_overwrite(extracted_path, output_dir)
    # Download data and dictionary
    url = 'https://dl.fbaipublicfiles.com/fairseq/data/wiki103_test_lm.tar.bz2'
    extracted_paths = download_and_extract(url)
    output_dir = os.path.join(MODELS_DIR, 'language_models/wiki103_test_lm')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for extracted_path in extracted_paths:
        move_with_overwrite(extracted_path, output_dir)


def _prepare_fasttext_embeddings():
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz'
    [extracted_path] = download_and_extract(url)
    Path(FASTTEXT_EMBEDDINGS_PATH).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(extracted_path, FASTTEXT_EMBEDDINGS_PATH)


def _prepare_wordnet():
    extracted_path = download_and_extract('http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz')[0]
    move_with_overwrite(extracted_path, WORDNET_DIR)


def _prepare_terp():
    url = 'https://github.com/snover/terp'
    git_clone(url, TERP_DIR)
    run_command(f'cd {TERP_DIR}; ant clean; ant')
    _prepare_wordnet()
    run_command(f'cd {TERP_DIR}; bin/setup_bin.sh {TERP_DIR} {os.environ["JAVA_HOME"]} {WORDNET_DIR}')


def _prepare_quest():
    url = 'https://github.com/ghpaetzold/questplusplus.git'
    git_clone(url, QUEST_DIR)
    run_command(f'cd {QUEST_DIR}; ant "-Dplatforms.JDK_1.8.home=/usr/lib/jvm/java-8-<<version>>"')
