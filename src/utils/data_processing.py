"""
This module, `data_processing.py`, is dedicated to handling various data processing tasks
for an image captioning project. It includes functions for preprocessing data, loading datasets,
transforming datasets for different phases of model training and evaluation, handling data batching
through custom collate functions, and configuring data loaders for model input.

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
preprocess(items, tokenizer, image_processor, device, max_length)
    Prepares image and text data by processing through tokenizers and image processors.

load_datasets(data_dir, step)
    Loads datasets from a specified directory, returning separate datasets for training,
    validation, and testing.

transform_datasets(train_ds, valid_ds, test_ds, preprocess_fn, step)
    Applies a preprocessing function to the datasets provided based on the specified
    training step.

collate_fn(batch)
    Custom collate function to be used with PyTorch DataLoader to properly batch examples.

get_data_loader(dataset, collate_fn, batch_size, eval_dataset)
    Configures and returns a DataLoader for either training or evaluation.
"""
import os
import re
import torch

from datasets import load_from_disk
from torch.utils.data import DataLoader


def preprocess(items, tokenizer, image_processor, device, max_length):
    """
    Preprocess images and text data for training or evaluation.

    Parameters
    ----------
    items : dict
        A batch of items from the dataset containing 'image' and 'text' keys.
    tokenizer : PreTrainedTokenizer
        The tokenizer for processing text data.
    image_processor : PreTrainedProcessor
        The processor for image data.
    device : torch.device
        The device to perform computations on.
    max_length : int
        The maximum length of the tokenized text.

    Returns
    -------
    dict
        A dictionary with preprocessed 'pixel_values' and 'labels'.
    """
    image_to_RGB = lambda img: img if img.mode == 'RGB' else img.convert('RGB') # CMYK (4 channels) is not accepted by the models

    if 'text' in items.keys(): # For #PraCegoVer63k dataset
        images = [image_to_RGB(img) for img in items["image"]]
        captions = [sentence for sentence in items["text"]]
    else: # For Flickr30k dataset
        images = [image_to_RGB(img) for img in items["image"] for _ in range(5)] # Flickr30k has 5 captions for each image
        captions = [' '.join(re.findall(r"[\w']+|[.,!?;]", sentence)) for sentences in items["caption"] for sentence in sentences]

    pixel_values = image_processor(
        images, 
        return_tensors="pt"
    ).pixel_values.to(device)

    targets = tokenizer(
        captions,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    return {'pixel_values': pixel_values, 'labels': targets["input_ids"]}


def load_datasets(data_dir, step='train'):
    """
    Load datasets from disk and apply preprocessing function.

    Parameters
    ----------
    data_dir : str
        The directory containing the dataset files.
    preprocess_fn : callable
        The preprocessing function to apply to the datasets.
    step : str, optional
        The step of the pipeline ('train' or 'eval'), by default 'train'.

    Returns
    -------
    tuple
        A tuple of datasets (train_ds, valid_ds, test_ds), depending on the step.
    """
    train_ds, valid_ds, test_ds = None, None, None
    if step=='train':
        train_ds = load_from_disk(os.path.join(data_dir, 'train.hf')) #.select(range(16))
        valid_ds = load_from_disk(os.path.join(data_dir, 'validation.hf')) #.select(range(16))
        test_ds = load_from_disk(os.path.join(data_dir, 'test.hf')) #.select(range(16))
    elif step=='eval':
        train_ds, valid_ds = None, None
        test_ds = load_from_disk(os.path.join(data_dir, 'test.hf')) #.select(range(16))
    else:
        raise Exception("The parameters `step` needs to be equals to `train` or `eval`")
    return train_ds, valid_ds, test_ds


def transform_datasets(train_ds=None, valid_ds=None, test_ds=None, preprocess_fn=None, step='train'):
    """
    Transform datasets with preprocessing function.

    Parameters
    ----------
    train_ds : Dataset, optional
        The training dataset to be transformed.
    valid_ds : Dataset, optional
        The validation dataset to be transformed.
    test_ds : Dataset, optional
        The test dataset to be transformed.
    preprocess_fn : callable
        The preprocessing function to apply to the datasets.
    step : str, optional
        The step of the pipeline ('train' or 'eval'), by default 'train'.

    Returns
    -------
    tuple
        A tuple of datasets (train_ds, valid_ds, test_ds), depending on the step.
    """
    train_dataset, valid_dataset, test_dataset = None, None, None
    if step=='train':
        train_dataset = train_ds.with_transform(preprocess_fn)
        valid_dataset = valid_ds.with_transform(preprocess_fn)
        test_dataset = test_ds.with_transform(preprocess_fn)
    elif step=='eval':
        train_dataset, valid_dataset = None, None
        test_dataset = test_ds.with_transform(preprocess_fn)
    else:
        raise Exception("The parameters `step` needs to be equals to `train` or `eval`")
    return train_dataset, valid_dataset, test_dataset


def collate_fn(batch):
    """
    Custom collate function to batch items together.

    Parameters
    ----------
    batch : list
        A list of preprocessed items.

    Returns
    -------
    dict
        A dictionary with 'pixel_values' and 'labels' tensors batched together.
    """
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }


def get_data_loader(dataset, collate_fn, batch_size, eval_dataset=None):
    """
    Get a DataLoader for evaluation.

    Parameters
    ----------
    dataset : Dataset
        The dataset to load data from.
    collate_fn : callable
        The function to collate data items into batches.
    batch_size : int
        The number of items per batch.
    eval_dataset : Dataset, optional
        The evaluation dataset, if different from the input dataset, by default None.

    Returns
    -------
    DataLoader
        A DataLoader ready for evaluation.
    """
    return DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size)