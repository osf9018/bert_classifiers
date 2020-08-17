""" Tools for loading datasets as Classification/SequenceLabelling Examples. """
import os
import logging
from collections import namedtuple

from tqdm import tqdm
from transformers import BasicTokenizer

from utils import retokenize

ClassificationExample = namedtuple(
    'ClassificationExample', ['id', 'tokens_a', 'tokens_b', 'label'])
SequenceLabellingExample = namedtuple(
    'SequenceLabellingExample', ['id', 'token_sequence', 'label_sequence'])


def load_classification_dataset(data_filename, do_lower_case):
    """ Loads classification examples from a dataset. """
    basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    examples = []
    with open(data_filename, 'r', encoding='utf-8') as data_file:
        lines = data_file.readlines()
        for i, line in tqdm(enumerate(lines), desc=f'reading `{os.path.basename(data_filename)}`...'):
            # example: __label__negative I don't like tomatoes.
            splitline = line.strip().split()
            label = splitline[0].split('__label__')[-1]
            tokens = ' '.join(splitline[1:])
            examples.append(
                ClassificationExample(
                    id=i,
                    tokens_a=basic_tokenizer.tokenize(tokens),
                    tokens_b=None,
                    label=label,
                )
            )
    logging.info('Number of `%s` examples: %d', data_filename, len(examples))
    return examples

def load_sequence_labelling_dataset(data_filename, do_lower_case):
    """ Loads sequence labelling examples from a dataset. """
    i = 0
    examples = []
    with open(data_filename, 'r', encoding='utf-8') as data_file:
        lines = data_file.readlines()
        token_sequence = []
        label_sequence = []
        for line in tqdm(lines, desc=f'reading `{os.path.basename(data_filename)}`...'):
            # example:
            #          My O
            #          name O
            #          is O
            #          Hicham B-PER
            #          . O
            splitline = line.strip().split()
            if splitline:
                token, label = splitline
                token_sequence.append(token)
                label_sequence.append(label)
            else:
                if token_sequence:
                    examples.append(
                        SequenceLabellingExample(
                            id=i,
                            token_sequence=token_sequence,
                            label_sequence=label_sequence,
                        )
                    )
                    i += 1
                token_sequence = []
                label_sequence = []

    # Don't forget to add the last example
    if token_sequence:
        examples.append(
            SequenceLabellingExample(
                id=i,
                token_sequence=token_sequence,
                label_sequence=label_sequence,
            )
        )
    retokenize(
        examples,
        tokenization_function=BasicTokenizer(do_lower_case=do_lower_case).tokenize)
    logging.info('Number of `%s` examples: %d', data_filename, len(examples))
    return examples
