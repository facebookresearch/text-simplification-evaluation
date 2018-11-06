# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from tseval.text import count_words, count_syllables_in_sentence


# https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
# https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
class ReadabilityScorer:
    def __init__(self):
        self.nb_words = 0
        self.nb_syllables = 0
        self.nb_sentences = 0

    def add_sentence(self, sentence):
        self.nb_words += count_words(sentence, remove_punctuation=True)
        self.nb_syllables += count_syllables_in_sentence(sentence)
        self.nb_sentences += 1

    def score(self):
        if self.nb_sentences == 0 or self.nb_words == 0:
            return (0, 0)
        # Flesch reading-ease
        fre = 206.835 - 1.015 * (self.nb_words / self.nb_sentences) - 84.6 * (self.nb_syllables / self.nb_words)
        # Flesch-Kincaid grade level
        fkgl = 0.39 * (self.nb_words / self.nb_sentences) + 11.8 * (self.nb_syllables / self.nb_words) - 15.59
        return (fre, fkgl)


def sentence_fre(sentence):
    scorer = ReadabilityScorer()
    scorer.add_sentence(sentence)
    fre, fkgl = scorer.score()
    return fre


def sentence_fkgl(sentence):
    scorer = ReadabilityScorer()
    scorer.add_sentence(sentence)
    fre, fkgl = scorer.score()
    return fkgl
