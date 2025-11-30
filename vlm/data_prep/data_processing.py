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

from io import BytesIO
from datasets import load_from_disk, load_dataset


def image_to_message(image, question):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_url = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    return [{
        "role":"user",
        "content": [{
                "type":"text",
                "text":question
            },{
                "type":"image_url",
                "image_url": {"url": image_url}
        }]
    }]


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
        train_dataset = train_ds.map(
          preprocess_fn, batched=True, remove_columns=train_ds.column_names
        )
        valid_dataset = valid_ds.map(
          preprocess_fn, batched=True, remove_columns=valid_ds.column_names
        )
        test_dataset = test_ds.map(
          preprocess_fn, batched=True, remove_columns=test_ds.column_names
        )
    elif step=='eval':
        train_dataset, valid_dataset = None, None
        test_dataset = test_ds.map(
          preprocess_fn, batched=True, remove_columns=test_ds.column_names
        )
    else:
        raise Exception("The parameters `step` needs to be equals to `train` or `eval`")
        
    return train_dataset, valid_dataset, test_dataset


def preprocess(question, text_per_image, image_column, text_column):
    """
    Prepares and preprocesses images and captions for model input, adjusting image mode 
    and duplicating entries as needed.

    Parameters
    ----------
    question : str
        The question prompt to accompany each caption, typically used in image captioning tasks.
    text_per_image : int
        Number of text captions per image. Used to duplicate images and filenames when 
        multiple captions per image are needed.
    image_column : str
        The key in the input dictionary representing image data.
    text_column : str
        The key in the input dictionary representing text captions or answers.

    Returns
    -------
    function
        A function `map_item` that processes a batch of items.
    """
    def map_item(items):
        """
        Converts image data to RGB format, replicates captions and images based on 
        `text_per_image`, and constructs a list of question prompts for each answer.

        Parameters
        ----------
        items : dict
            A dictionary containing image, text caption, and filename data.

        Returns
        -------
        dict
            A dictionary with processed data.
        """
        image_to_RGB = lambda img: img if img.mode == 'RGB' else img.convert('RGB') # CMYK (4 channels) is not accepted by the models

        images = [image_to_RGB(img) for img in items[image_column]]
        answers = items[text_column]
        filenames = items['filename']

        if text_per_image > 1: # For Flickr30K (5 captions per image)
            filenames = [filename for filename in filenames for _ in range(text_per_image)]
            images = [img for img in images for _ in range(text_per_image)]
            answers = [sentence for sentences in answers for sentence in sentences]
        
        questions = [question for _ in answers]

        return {
            "filename": filenames,
            "image": images,
            "answer": answers,
            "question": questions
        }
    
    return map_item



def preprocess_for_API(question, text_per_image, image_column, text_column):
    def map_item(items):
        image_to_RGB = lambda img: img if img.mode == 'RGB' else img.convert('RGB') # CMYK (4 channels) is not accepted by the models
        
        images = [image_to_RGB(img) for img in items[image_column]]
        answers = items[text_column]
        filenames = items['filename']

        if text_per_image > 1: # For Flickr30K (5 captions per image)
            filenames = [filename for filename in filenames for _ in range(text_per_image)]
            images = [img for img in images for _ in range(text_per_image)]
            answers = [sentence for sentences in answers for sentence in sentences]
        
        questions = [question for _ in answers]
        messages = [image_to_message(img, qst) for img, qst in zip(images, questions)]

        return {
            "filename": filenames,
            "image": images,
            "answer": answers,
            "question": questions,
            "message": messages
        }
    
    return map_item