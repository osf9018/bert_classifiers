# Adapted from Huggingface's transformers library:
# https://github.com/huggingface/transformers

""" Main script. """
import os
import logging
import argparse
import datetime
from collections import Counter

import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    BertForSequenceClassification
)

from data import load_classification_dataset, load_sequence_labelling_dataset
from utils import set_seed, retokenize, build_and_cache_features
from training_utils import train, evaluate

from download import MODEL_TO_URL
AVAILABLE_MODELS = list(MODEL_TO_URL.keys()) + ['bert-base-uncased']

def parse_args():
    """ Parse command line arguments and initialize experiment. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=['classification', 'sequence_labelling'],
        help="The evaluation task."
    )
    parser.add_argument(
        "--embedding",
        type=str,
        required=True,
        choices=AVAILABLE_MODELS,
        help="The model to use."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Whether to apply lowercasing during tokenization."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size to use for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--validation_ratio",
        default=0.5, type=float, help="Proportion of training set to use as a validation set.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_ratio",
        default=0.1, type=int, help="Linear warmup over warmup_ratio*total_steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Do training & validation."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Do prediction on the test set."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--train",
        type=str,
        help="Training data."
    )
    parser.add_argument(
        "--validation",
        type=str,
        help="Validation data."
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Test data."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory of an already trained model."
    )
    parser.add_argument(
        "--test_output",
        type=str,
        help="Output file for storing the evaluation results on test data."
    )
    parser.add_argument(
        "--train_output_dir",
        type=str,
        help="Output directory for storing the trained model and its evaluation results."
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        help="directory containing a pretrained BERT model."
    )
    parser.add_argument(
        "--do_eval_test_in_training",
        action="store_true",
        help="Perform the evaluation on test data during training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for storing all the outputs"
    )
    args = parser.parse_args()
    return args

def main(args):
    """ Main function. """
    # --------------------------------- INIT ---------------------------------
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO)

    # Check for GPUs
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1  # This script doesn't support multi-gpu
        args.device = torch.device("cuda")
        logging.info("Using GPU (`%s`)", torch.cuda.get_device_name(args.device))
    else:
        args.device = torch.device("cpu")
        logging.info("Using CPU")

    # Set random seed for reproducibility
    set_seed(seed_value=args.seed)

    # --------------------------------- DATA ---------------------------------
    # Tokenizer
    logging.disable(logging.INFO)
    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(args.pretrained_model, args.embedding),
        do_lower_case=args.do_lower_case)
    logging.disable(logging.NOTSET)
    tokenization_function = tokenizer.tokenize

    # Pre-processsing: apply basic tokenization (both) then split into wordpieces (BERT only)
    data = {}
    for data_type in ['train', 'validation', 'test']:
        data_file = args.__dict__.get(data_type)
        if data_file is not None:
            if args.task == 'classification':
                func = load_classification_dataset
            elif args.task == 'sequence_labelling':
                func = load_sequence_labelling_dataset
            else:
                raise NotImplementedError
            current_data = func(data_filename=data_file, do_lower_case=args.do_lower_case)
            retokenize(current_data, tokenization_function)
            data[data_type] = current_data

    if 'validation' not in data:
        logging.info('Splitting training data into train / validation sets...')
        data['validation'] = data['train'][:int(args.validation_ratio * len(data['train']))]
        data['train'] = data['train'][int(args.validation_ratio * len(data['train'])):]
        logging.info('New number of training sequences: %d', len(data['train']))
        logging.info('New number of validation sequences: %d', len(data['validation']))

    # Count target labels or classes
    if args.task == 'classification':
        counter_all = Counter(
            [example.label for example in data['train'] + data['validation'] + data['test']])
        counter = Counter(
            [example.label for example in data['train']])

        # Maximum sequence length is either 512 or maximum token sequence length + 5
        max_seq_length = min(
            512,
            5 + max(
                map(len, [
                    e.tokens_a if e.tokens_b is None else e.tokens_a + e.tokens_b
                    for e in data['train'] + data['validation'] + data['test']
                ])
            )
        )
    elif args.task == 'sequence_labelling':
        counter_all = Counter(
            [label
             for example in data['train'] + data['validation'] + data['test']
             for label in example.label_sequence])
        counter = Counter(
            [label
             for example in data['train']
             for label in example.label_sequence])

        # Maximum sequence length is either 512 or maximum token sequence length + 5
        max_seq_length = min(
            512,
            5 + max(
                map(len, [
                    e.token_sequence
                    for e in data['train'] + data['validation'] + data['test']
                ])
            )
        )
    else:
        raise NotImplementedError
    labels = sorted(counter_all.keys())
    num_labels = len(labels)

    logging.info("Goal: predict the following labels")
    for i, label in enumerate(labels):
        logging.info("* %s: %s (count: %s)", label, i, counter[label])

    # Input features: list[token indices]
    pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_label_id = None
    if args.task == 'sequence_labelling':
        pad_token_label_id = CrossEntropyLoss().ignore_index

    dataset = {}
    logging.info("Maximum sequence lenght: %s", max_seq_length)
    for split in data:
        dataset[split] = build_and_cache_features(
            args,
            split=split,
            tokenizer=tokenizer,
            examples=data[split],
            labels=labels,
            pad_token_id=pad_token_id,
            pad_token_label_id=pad_token_label_id,
            max_seq_length=max_seq_length)
    del data  # Not used anymore

    # ------------------------------ TRAIN / EVAL ------------------------------
    # Training
    if args.do_train:
        args.start_time = datetime.datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss')
        if args.output_dir is None:
            args.output_dir = os.path.join(
                args.train_output_dir,
                args.embedding,
                f'{args.start_time}__seed-{args.seed}')
            # --------------------------------- MODEL ---------------------------------
        # Initialize model
        if args.task == 'classification':
            model = BertForSequenceClassification
        elif args.task == 'sequence_labelling':
            model = BertForTokenClassification
        else:
            raise NotImplementedError

        logging.info('Loading `%s` model...', args.embedding)
        logging.disable(logging.INFO)
        config = BertConfig.from_pretrained(
            os.path.join(args.pretrained_model, args.embedding),
            num_labels=num_labels)
        model = model.from_pretrained(
            os.path.join(args.pretrained_model, args.embedding),
            config=config)
        logging.disable(logging.NOTSET)

        model.to(args.device)
        logging.info('Model:\n%s', model)

        # Log args
        logging.info('Using the following arguments for training:')
        for k, v in vars(args).items():
            logging.info("* %s: %s", k, v)

        # train model
        global_step, train_loss, best_val_metric, best_val_epoch = train(
            args=args,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            labels=labels,
            pad_token_label_id=pad_token_label_id
        )
        logging.info("global_step = %s, average training loss = %s", global_step, train_loss)
        logging.info("Best performance: Epoch=%d, Value=%s", best_val_epoch, best_val_metric)
        with open(os.path.join(args.output_dir, 'validation-performance.txt'), 'w') as f:
            f.write(f'best validation score: {best_val_metric}\n')
            f.write(f'best validation epoch: {best_val_epoch}\n')


    # Evaluation on test data
    if args.do_predict:
        # Load best model
        if args.task == 'classification':
            model = BertForSequenceClassification
        elif args.task == 'sequence_labelling':
            model = BertForTokenClassification
        else:
            raise NotImplementedError
        logging.disable(logging.INFO)
        model = model.from_pretrained(args.output_dir)
        logging.disable(logging.NOTSET)
        model.to(args.device)

        # Compute predictions and metrics
        results, _ = evaluate(
            args=args,
            eval_title='test',
            eval_dataset=dataset['test'],
            model=model, labels=labels,
            pad_token_label_id=pad_token_label_id
        )
        # Save metrics
        with open(args.test_output, 'w') as fd:
            fd.write('--- Performance on test set ---\n')
            for k, v in results.items():
                fd.write(f'{k}: {v}\n')

if __name__ == "__main__":
    main(parse_args())
