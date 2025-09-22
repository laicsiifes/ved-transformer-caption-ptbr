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


def join_datasets(ds_native, ds_translated, dataset):
    df = None
    if dataset:
        df = dataset.to_pandas()
    else:
        df_native = ds_native.to_pandas().rename(columns={'caption': 'caption_native'})
        df_translated = ds_translated.to_pandas().rename(columns={'caption': 'caption_translated'})

        df = df_native.set_index('filename').join(
            other=df_translated[['filename', 'caption_translated']].set_index('filename'),
            how='inner'
        ).reset_index()

        df['caption'] = df['caption_native'] + df['caption_translated']

        df = df.drop(columns=['caption_native', 'caption_translated'])

    return df

    # features = Features({
    #     'image': Image(mode=None, decode=True),
    #     'caption': List(Value('string')),
    #     'sentids': List(Value('int32')),
    #     'split': Value('string'),
    #     'img_id': Value('string'),
    #     'filename': Value('string'),
    #     'correct_group': List(Value('string')),
    #     'control_group': List(Value('string')),
    #     'incorrect_group_filenames': List(Value('string')),
    #     'incorrect_group': List(Value('string'))
    # })

    # return Dataset.from_pandas(df, features=features)


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
        dataset_native = load_dataset(hf_dataset["dataset_native"])
        dataset_translated = load_dataset(hf_dataset["dataset_translated"])
    else:
        dataset = load_dataset(hf_dataset)
    
    return dataset_native['test'], dataset_translated['test'], dataset['test']


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
        row['incorrect_group'] = incorrect_sample['caption'].values
    else:
        row['incorrect_group'] = incorrect_sample['correct_group'].values

    return row


def generate_grouped_dataset(
        dataset_native,
        dataset_translated,
        dataset,
        correct_sample_size,
        incorrect_sample_size,
        reproducible=False,
        use_control_for_incorrects=False
    ):
    df = join_dataset(dataset_native, dataset_translated, dataset)

    if reproducible:
        random.seed(correct_sample_size)

    for index, row in df.iterrows():

        positive_ids = random.sample([i for i in range(10)], correct_sample_size)
        control_ids = [i for i in [i for i in range(10)] if i not in positive_ids]

        captions = df.loc[index, 'caption'].tolist()

        df.loc[index, 'correct_group'] = [captions[i] for i in positive_ids]
        df.loc[index, 'control_group'] = [captions[i] for i in control_ids]
    
    incorrect_data = df.explode(['caption']) if use_control_for_incorrects else df.explode(['correct_group'])
    
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
        'correct_group': List(Value('string')),
        'control_group': List(Value('string')),
        'incorrect_group_filenames': List(Value('string')),
        'incorrect_group': List(Value('string'))
    })
    
    return Dataset.from_pandas(df, features=features)


def generate_grouped_dataset_(dataset, correct_sample_size, incorrect_sample_size, reproducible=False, use_control_for_incorrects=False):
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