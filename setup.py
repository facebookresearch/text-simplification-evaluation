# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


def _post_setup():
    from nltk.downloader import download
    download('stopwords')


# Set up post install actions as per https://stackoverflow.com/a/36902139/1226799
class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        _post_setup()


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        _post_setup()


setup(
    name='tseval',
    version='1.0',
    description='Reference-less Text Simplification Evaluation Methods',
    author='Louis Martin',
    author_email='louismartin@fb.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.1.15', 'scipy', 'pandas', 'torch', 'sklearn', 'python-Levenshtein', 'gitpython',
        'nlg-eval@git+https://github.com/Maluuba/nlg-eval.git',
        'fairseq@git+https://github.com/pytorch/fairseq@e286243c68f1589a781488580fc19388714612be',
        ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
