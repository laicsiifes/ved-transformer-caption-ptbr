"""
Metrics Evaluation Module for Image Captioning
==============================================

This module provides functions to calculate evaluation metrics for image captioning 
models, particularly for multimodal vision-language models. It includes helper 
functions for computing BERTScore, CLIPScore, ROUGE, BLEU, METEOR, and other 
captioning metrics, providing both individual and aggregated scores for 
comprehensive analysis.

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
compute_individual_metrics(predictions, labels, metrics, images_names, images)
    Calculate individual evaluation metrics for image captioning predictions.

compute_all_judges(predictions, labels, images_names, images, text_per_image)
    Evaluate and compute various metrics for image captioning predictions, including BERTScore, 
    CLIPScore, ROUGE, BLEU, and METEOR, with both individual and aggregated scores.

evaluate_predictions(dataset, model, config, collate_fn, processor, generate_args, trainer)
    Generate evaluation results for a model, saving metrics and training history to CSV files.
"""

import evaluate
import os
import re
import pandas as pd
import numpy as np

from tqdm import tqdm

from evaluation.judges import compute_llm_as_a_judge, compute_vlm_as_a_judge




def compute_individual_metrics(predictions, labels, template, images_names, images):
    """
    Calculate individual evaluation metrics for image captioning predictions.

    Parameters
    ----------
    predictions : list of str
        A list of predicted captions for each image.
    labels : list of list of str
        A list of lists, where each sublist contains reference captions for each image.
    metrics : dict
        A dictionary of metric functions, including BERTScore and custom metrics.
    images_names : list of str
        A list of filenames or identifiers for each image.
    images : list of PIL.Image.Image
        A list of images corresponding to each prediction.

    Returns
    -------
    dict
        A dictionary containing individual metric results for each image-caption pair.
    """
    # LLM and VLM as evaluators compute the metrics for each example by default
    llm_as_a_judge_score_result = compute_llm_as_a_judge(predictions, labels, template['llm_as_a_judge'])
    vlm_as_a_judge_score_result = compute_llm_as_a_judge(predictions, images, template['vlm_as_a_judge'])

    return {
        "filename": images_names,
        **llm_as_a_judge_score_result,
        **vlm_as_a_judge_score_result,

    }


def compute_all_judges(predictions, dataset, text_column, image_column, template, config):
    """
    Evaluate and compute various metrics for image captioning predictions, including BERTScore, 
    CLIPScore, ROUGE, BLEU, and METEOR, with both individual and aggregated scores.

    Parameters
    ----------
    predictions : list of str
        A list of predicted captions for each image.
    labels : list of list of str
        A list of lists, where each sublist contains reference captions for each image.
    images_names : list of str
        A list of filenames or identifiers for each image.
    images : list of PIL.Image.Image
        A list of images corresponding to each prediction.
    text_per_image : int
        The number of captions related to each image

    Returns
    -------
    dict
        A dictionary with both individual and aggregated scores.
    """
    images_names = dataset['filename']
    labels = dataset[text_column]
    images = dataset[image_column]

    evaluate.enable_progress_bar()

    print("Computing Metrics Individually")
    individual_metrics = compute_individual_metrics(
        predictions, labels, template, images_names, images
    )

    print("Computing Metrics Total")
    original_metrics = {
        "llm_as_a_judge": np.mean(individual_metrics["llm_as_a_judge_score"]),
    }

    sample_metrics = {}

    print("Computing Metrics Mean/Std")
    for metric in [
        "llm_as_a_judge_score"
    ]:  
        sample_metrics[f"{metric}_mean"] = round(np.mean(individual_metrics[metric]) * 100, 4)
        sample_metrics[f"{metric}_std"] = round(np.std(individual_metrics[metric]) * 100, 4)

    return {
        "individual_metrics": individual_metrics,
        "original_metrics": original_metrics,
        "sample_metrics": sample_metrics
    }


def evaluate_predictions(
        raw_dataset,
        predictions,
        text_column,
        image_column,
        results_dir
    ):
    """
    Generate evaluation results for a model, saving metrics and training history to CSV files.

    Parameters
    ----------
    dataset : Dataset
        The processed dataset for evaluation.
    raw_dataset : Dataset
        The raw dataset used for reference during evaluation.
    model : PreTrainedModel
        The model to be evaluated.
    config : dict
        Configuration dictionary with paths and options for saving results.
    collate_fn : Callable
        The function used to collate and prepare batch data.
    processor : ProcessorMixin
        Processor for handling model inputs and tokenization.
    generate_args : dict
        Arguments for generating predictions during evaluation.
    trainer : Trainer, optional
        Trainer object, used for saving training logs if available (default is None).

    Returns
    -------
    None
        This function saves evaluation results to CSV files in the specified directories.
    """
    results = compute_all_judges(
        predictions=predictions, 
        dataset=raw_dataset,
        text_column=text_column,
        image_column=image_column
    )

    pd.DataFrame(results["individual_metrics"]).to_csv(
        path_or_buf=os.path.join(results_dir, f'individual_eval_metrics.csv'),
        index=False
    )

    pd.DataFrame(results["original_metrics"], index=[0]).to_csv(
        path_or_buf=os.path.join(results_dir, f'original_eval_metrics.csv'),
        index=False
    )

    pd.DataFrame(results["sample_metrics"], index=[0]).to_csv(
        path_or_buf=os.path.join(results_dir, f'sample_eval_metrics.csv'),
        index=False
    )