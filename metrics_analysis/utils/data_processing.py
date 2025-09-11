"""
Data Processing Module for Multimodal Vision-Language Models
============================================================

This module provides helper functions to preprocess data and collate batches for training 
and evaluation with different multimodal vision-language models, such as LLaMa 3.2 Vision, 
Phi-3 Vision, and PaliGemma. The functions manage sequence masking, image formatting, 
and batching for image captioning tasks in Brazilian Portuguese.

Authors
-------
BSc, Gabriel Mota Bromonschenkel Lima
Email: gabriel.mota.b.lima@gmail.com

PhD, Hilário Tomaz Alves de Oliveira
Email: hilariotomaz@gmail.com

PhD, Thiago Meireles Paixão
Email: thiago.paixao@ifes.edu.br

Functions
---------
load_datasets(data_dir, step, hf_dataset, dataset_from_hub)
    Loads training, validation, and test datasets from either local storage or the Hugging Face Hub.

transform_datasets(train_ds, valid_ds, test_ds, preprocess_fn, step)
    Applies a preprocessing function to the datasets provided based on the specified
    training step.

preprocess(items, question)
    Prepares and preprocesses images and captions for model input, adjusting image mode 
    and duplicating entries as needed.
"""

import os
import base64
import random

from pprint import pprint
from io import BytesIO
from datasets import load_from_disk, load_dataset, Dataset, Image, List, Value, Features
from tqdm import tqdm

tqdm.pandas()


def load_datasets(data_dir, step='train', hf_dataset=None, dataset_from_hub=False):
    """
    Load training, validation, and test datasets from either local storage or the Hugging Face Hub.

    Parameters
    ----------
    data_dir : str
        Directory path to the local dataset files.
    step : str, optional
        Determines the data loading step, either 'train' or 'eval' (default is 'train').
    hf_dataset : str, optional
        Identifier for the Hugging Face Hub dataset, used if `dataset_from_hub` is True.
    dataset_from_hub : bool, optional
        Flag to load datasets from the Hugging Face Hub instead of local storage (default is False).

    Returns
    -------
    tuple
        A tuple containing the training, validation, and test datasets. If `step` is 'eval',
        only the test dataset is returned with the others set to None.

    Raises
    ------
    Exception
        If `step` is not set to 'train' or 'eval'.
    """
    train_ds, valid_ds, test_ds = None, None, None

    if dataset_from_hub:
        dataset = load_dataset(hf_dataset)

    if step=='train':
        train_ds = dataset['train'] if dataset_from_hub \
             else load_from_disk(os.path.join(data_dir, 'train.hf'))
        valid_ds = dataset['validation'] if dataset_from_hub \
             else load_from_disk(os.path.join(data_dir, 'validation.hf'))
        test_ds = dataset['test'] if dataset_from_hub \
             else load_from_disk(os.path.join(data_dir, 'test.hf'))
    elif step=='eval':
        train_ds, valid_ds = None, None
        test_ds = dataset['test'] if dataset_from_hub \
             else load_from_disk(os.path.join(data_dir, 'test.hf'))
    else:
        raise Exception("The parameters `step` needs to be equals to `train` or `eval`")
    
    return train_ds, valid_ds, test_ds


def select_incorrect_data(row, incorrect_sample_size, incorrect_data, replacement=False, reproducible=True, use_control_for_incorrects=False):
    random_state = int(row['img_id']) if reproducible else None
    current_filename = row['filename']
    incorrect_sample = incorrect_data.loc[~incorrect_data.filename.isin([current_filename])].sample(
        n=incorrect_sample_size, 
        replace=replacement,
        random_state=random_state
    )
    row['incorrect_group_filenames'] = incorrect_sample['filename'].values

    if use_control_for_incorrects:
        row['incorrect_group_sentids'] = incorrect_sample['sentids'].values
        row['incorrect_group'] = incorrect_sample['caption'].values
    else:
        row['incorrect_group_sentids'] = incorrect_sample['correct_group_sentids'].values
        row['incorrect_group'] = incorrect_sample['correct_group'].values

    return row


def generate_grouped_dataset(dataset, correct_sample_size, incorrect_sample_size, reproducible=False, use_control_for_incorrects=False):
    df = dataset.to_pandas()

    if reproducible:
        random.seed(correct_sample_size)

    print("\nCorrect group IDs")
    df['correct_group_sentids'] = df['sentids'].progress_apply(
        lambda x: random.sample(x.tolist(), correct_sample_size)
    )

    print("\nCorrect group captions")
    df['correct_group'] = df.progress_apply(
        lambda x: [
            x['caption'].tolist()[i] for i in range(len(x['caption'])) if i in [int(sentid) % len(x['caption']) for sentid in x['correct_group_sentids']]
        ],
        axis=1
    )
    print("\nControl group IDs")
    df['control_group_sentids'] = df.progress_apply(
        lambda x: [i for i in x['sentids'].tolist() if i not in x['correct_group_sentids']],
        axis=1
    )

    print("\nControl group captions")
    df['control_group'] = df.progress_apply(
        lambda x: [
            x['caption'].tolist()[i] for i in range(len(x['caption'])) if i in [int(sentid) % len(x['caption']) for sentid in x['control_group_sentids']]
        ],
        axis=1
    )
    
    incorrect_data = df.explode(['caption', 'sentids']) if use_control_for_incorrects else df.explode(['correct_group', 'correct_group_sentids'])
    
    print("\nIncorrect group IDs and captions")
    df = df.progress_apply(
        lambda row: select_incorrect_data(
            row=row,
            incorrect_sample_size=incorrect_sample_size,
            incorrect_data=incorrect_data,
            replacement=False
        ),
        axis=1
    )
    
    features = Features({
        'image': Image(mode=None, decode=True),
        'caption': List(Value('string')),
        'sentids': List(Value('int32')),
        'split': Value('string'),
        'img_id': Value('string'),
        'filename': Value('string'),
        'correct_group_sentids': List(Value('int32')),
        'correct_group': List(Value('string')),
        'control_group_sentids': List(Value('int32')),
        'control_group': List(Value('string')),
        'incorrect_group_filenames': List(Value('string')),
        'incorrect_group_sentids': List(Value('int32')),
        'incorrect_group': List(Value('string'))
    })
    
    return Dataset.from_pandas(df, features=features)