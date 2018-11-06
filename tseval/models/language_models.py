# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import os

from fairseq import options, utils, tasks
import torch

from tseval.resources.paths import MODELS_DIR


def load_fairseq_lm_model_and_dict(checkpoint_path, data_path):
    # Initialize model
    parser = options.get_eval_lm_parser()
    parsed_args = options.parse_args_and_arch(parser, ['--path', checkpoint_path, data_path])
    task = tasks.setup_task(parsed_args)
    models, _ = utils.load_ensemble_for_inference([checkpoint_path], task)
    return models[0], task.dictionary


def init_fairseq_lm_globals():
    print('Loading fairseq language model...')
    fairseq_lm_dir = os.path.join(MODELS_DIR, 'language_models/wiki103_fconv_lm')
    checkpoint_path = os.path.join(fairseq_lm_dir, 'wiki103.pt')
    data_path = os.path.join(MODELS_DIR, 'language_models/wiki103_test_lm')
    if not os.path.exists(data_path):
        from tseval.resources.prepare import prepare_resource
        prepare_resource('fairseq_lm')
    global FAIRSEQ_MODEL, DICTIONARY, DEVICE
    FAIRSEQ_MODEL, DICTIONARY = load_fairseq_lm_model_and_dict(checkpoint_path, data_path)
    FAIRSEQ_MODEL.make_generation_fast_()
    DEVICE = torch.device('cuda')
    FAIRSEQ_MODEL = FAIRSEQ_MODEL.to(DEVICE)
    print('Done.')


def word_probs(sentence, model, dictionary, device, log_probs=True, verbose=False):
    def prepend_eos(indexes):
        return torch.cat([torch.LongTensor([[dictionary.eos()]]).to(device), indexes], dim=1)

    def append_eos(indexes):
        return torch.cat([indexes, torch.LongTensor([[dictionary.eos()]]).to(device)], dim=1)

    # Forward pass
    indexes = torch.LongTensor([[dictionary.index(word) for word in sentence.split()]]).to(device)
    src_lengths = torch.LongTensor([indexes.shape[1] + 1]).to(device)  # src_lengths = [len(words) + 1]
    decoder_out = model(prepend_eos(indexes), src_lengths)

    # Compute sentence log-likelihood
    logprobs = model.get_normalized_probs(decoder_out,
                                          log_probs=log_probs,
                                          sample={'target': append_eos(indexes)}).squeeze()
    if verbose:
        # Print top predictions
        # We must feed None as target to get_normalized_probs
        logprobs = model.get_normalized_probs(decoder_out, log_probs=log_probs, sample={'target': None}).squeeze()
        for i in range(indexes.shape[1] + 1):
            print(i)
            sorted_values, sorted_indexes = logprobs[i].sort(descending=True)
            top_predictions = [f'{dictionary[sorted_indexes[j]]} ({sorted_values[j]:.1f})' for j in range(5)]
            input_words = ' '.join([dictionary[idx] for idx in prepend_eos(indexes).squeeze()[:i+1]])
            print(input_words, '->', '; '.join(top_predictions))

    return logprobs.gather(dim=1, index=append_eos(indexes).reshape(-1, 1)).squeeze().data.cpu().numpy()


@functools.lru_cache(maxsize=10000)
def memoized_word_probs(sentence):
    if 'FAIRSEQ_MODEL' not in globals():
        init_fairseq_lm_globals()
    return word_probs(sentence, FAIRSEQ_MODEL, DICTIONARY, DEVICE)


def average_sentence_lm_prob(sentence):
    return memoized_word_probs(sentence).mean()


def min_sentence_lm_prob(sentence):
    return memoized_word_probs(sentence).min()
