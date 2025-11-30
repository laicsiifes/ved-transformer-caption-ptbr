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
import numpy as np
import pandas as pd

from pprint import pprint
from io import BytesIO
from datasets import load_from_disk, load_dataset, Dataset, Image, List, Value, Features
from tqdm import tqdm

tqdm.pandas()


def join_datasets(ds_native, ds_translated, dataset):
    df = None
    if dataset:
        df = dataset.to_pandas()
        df['caption'] = df['caption'].apply(lambda x: x.tolist())
    else:
        df_native = ds_native.to_pandas().rename(columns={'caption': 'caption_native'})
        df_translated = ds_translated.to_pandas().rename(columns={'caption': 'caption_translated'})

        df = df_native.set_index('filename').join(
            other=df_translated[['filename', 'caption_translated']].set_index('filename'),
            how='inner'
        ).reset_index()

        # df = df.progress_apply(lambda x: join_lists(x), axis=1)
        df['caption'] = df.apply(
            lambda row: row['caption_native'].tolist() + row['caption_translated'].tolist(),
            axis=1
        )

        df = df.drop(columns=['caption_native', 'caption_translated'])

    return df


def load_datasets(data_dir, step='train', hf_dataset=None):
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
    dataset_native, dataset_translated, dataset = None, None, None

    if isinstance(hf_dataset, dict):
        dataset_native = load_dataset(hf_dataset["dataset_native"])['test']
        dataset_translated = load_dataset(hf_dataset["dataset_translated"])['test']
    else:
        dataset = load_dataset(hf_dataset)['test']

    return dataset_native, dataset_translated, dataset


def select_incorrect_data(row, incorrect_sample_size, incorrect_data, replacement, reproducible, use_control_as_incorrect):
    random_state = int(row['img_id']) if reproducible else None
    current_filename = row['filename']
    incorrect_sample = incorrect_data.loc[incorrect_data['filename'] != current_filename].sample(
        n=incorrect_sample_size,
        replace=replacement,
        random_state=random_state
    )
    row['incorrect_group_filenames'] = incorrect_sample['filename'].values.tolist()

    if use_control_as_incorrect:
        row['incorrect_group'] = incorrect_sample['caption'].values.tolist()
    else:
        row['incorrect_group'] = incorrect_sample['correct_group'].values.tolist()

    return row


def select_correct_data(row, correct_sample_size):
    ids = [i for i in range(len(row['caption']))]
    id_pos = random.sample(ids, correct_sample_size)
    id_cnt = [idx for idx in ids if idx not in id_pos]

    row['correct_group'] = [row['caption'][idx] for idx in id_pos]
    row['control_group'] = [row['caption'][idx] for idx in id_cnt]
    return row

def generate_grouped_dataset(
        dataset_native,
        dataset_translated,
        dataset,
        correct_sample_size,
        incorrect_sample_size,
        reproducible,
        use_control_as_incorrect,
        replacement
    ):
    pd.set_option('display.max_columns', None)
    df = join_datasets(dataset_native, dataset_translated, dataset)

    print('\n', 'Source Dataset')
    print('\t', df.info())
    print('\t', df[['filename', 'caption']].head())
    print('\t', df.caption[0], '\n')

    if reproducible:
        random.seed(correct_sample_size)

    df = df.progress_apply(lambda row: select_correct_data(row=row, correct_sample_size=correct_sample_size), axis=1)

    print('\n', 'Correct/Control Columns')
    print('\t', df.info())
    print('\t', df[['filename', 'caption', 'control_group', 'correct_group']].head())
    print('\t', df.caption[0], '\n')

    incorrect_data = df.explode(['caption']) if use_control_as_incorrect else df.explode(['correct_group'])

    print("\nIncorrect group IDs and captions")
    df = df.progress_apply(
        lambda row: select_incorrect_data(
            row=row,
            incorrect_sample_size=incorrect_sample_size,
            incorrect_data=incorrect_data,
            replacement=replacement,
            reproducible=reproducible,
            use_control_as_incorrect=use_control_as_incorrect
        ),
        axis=1
    )

    print('\n', 'Incorrect/Correct/Control Columns')
    print('\t', df.info())
    print('\t', df[['filename', 'caption', 'control_group', 'correct_group', 'incorrect_group', 'incorrect_group_filenames']].head())
    print('\t', df.caption[0])
    print('\t', df.incorrect_group[0])
    print('\t', df.isnull().any().sum())
    print('\t', type(df.control_group[0]), type(df.correct_group[0]), type(df.incorrect_group[0]), '\n')

    features = Features({
        'image': Image(mode=None, decode=True),
        'caption': List(Value('string')),
        'sentids': List(Value('int32')),
        'split': Value('string'),
        'img_id': Value('string'),
        'filename': Value('string'),
        'correct_group': List(Value('string')),
        'control_group': List(Value('string')),
        'incorrect_group_filenames': List(Value('string')),
        'incorrect_group': List(Value('string'))
    })

    return Dataset.from_pandas(df, features=features)
