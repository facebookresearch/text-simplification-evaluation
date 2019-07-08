# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from setuptools import setup, find_packages


setup(
    name='tseval',
    version='1.0',
    description='Reference-less Text Simplification Evaluation Methods',
    author='Louis Martin',
    author_email='louismartin@fb.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.1.15', 'scipy', 'pandas', 'torch', 'sklearn', 'python-Levenshtein', 'gitpython', 'nltk',
        # Commented out because of https://github.com/Maluuba/nlg-eval/issues/77
        # 'nlg-eval@git+https://github.com/Maluuba/nlg-eval.git',
        'fairseq@git+https://github.com/pytorch/fairseq@e286243c68f1589a781488580fc19388714612be',
        ],
)
