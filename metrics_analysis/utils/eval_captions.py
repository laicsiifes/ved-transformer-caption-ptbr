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

compute_all_metrics(predictions, labels, images_names, images, text_per_image)
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

from utils.metrics import (
    compute_bert_scores,
    compute_clip_scores,
    compute_rouge_scores,
    compute_meteor_scores,
    compute_bleu_scores,
    compute_cider_scores
)

try:
    from aac_metrics import Evaluate
except:
    print(f"AAC metrics is not available, skipping...")
else:
    print(f"AAC metrics is available, importing...")


def compute_individual_metric(predictions, labels, scorer, name):
    result = {}
    repeat = 1

    # Select the computing metric funtion
    if "rouge" == name:
        compute = compute_rouge_scores
    elif "meteor" == name:
        compute = compute_meteor_scores
    elif "bleu" == name:
        compute = compute_bleu_scores
    elif "cider" == name:
        # Works only if the prediction have at least 2 labels to compare
        compute = compute_cider_scores
        repeat = 2
    else:
        compute = None

    # If there is a compute function, score the predictions 
    if compute:
        # with tqdm(total=len(predictions)) as pbar:
        #     pbar.set_description(f"Eval. {scorer.name.upper()}")
        # Create an empty dict with empty lists to add the by-example scores
        result = {
            k:[] for k in list(compute([''], [''], scorer[name]).keys())
        }

        # Compute the score to each example
        for prediction, label in zip(predictions, labels):
            individual_result = compute([prediction], [label], scorer[name])

            # Append the individual scores to the result dict
            for key in individual_result:
                result[key].append(individual_result[key])
            # pbar.update(1)
    else:
        print("`scorer` parameter is not set correctly, returning empty metric dict.")

    return result


def compute_individual_metrics(control_group, target_group, metrics, images):
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
    # BERTScore and CLIPScore compute the metrics for each example by default
    bertscore_result = compute_bert_scores(target_group, control_group, metrics["bertscore"])
    clipscore_result = compute_clip_scores(target_group, control_group, images)

    # Computing example-by-example results for ROUGE, METEOR and BLEU
    rouge_result  = compute_individual_metric(target_group, control_group, metrics, "rouge")
    meteor_result = compute_individual_metric(target_group, control_group, metrics, "meteor")
    bleu_result   = compute_individual_metric(target_group, control_group, metrics, "bleu")
    # cider_result  = compute_individual_metric(target_group, control_group, metrics, "cider")

    return {
        **bertscore_result,
        **clipscore_result,
        **rouge_result,
        **meteor_result,
        **bleu_result,
        # **cider_result
    }


def compute_metrics_sample(metrics):
    def map_item(item):
        print(item["filename"])
        print(item["correct_group"])
        correct_group_metrics = compute_individual_metrics(
            control_group=[item['control_group']]*len(item['correct_group']),
            target_group=item['correct_group'],
            metrics=metrics,
            images=[item["image"]]*len(item['correct_group'])
        )

        print(item["incorrect_group"])
        incorrect_group_metrics = compute_individual_metrics(
            control_group=[item['control_group']]*len(item['incorrect_group']),
            target_group=item['incorrect_group'],
            metrics=metrics,
            images=[item["image"]]*len(item['incorrect_group'])
        )

        for metric in correct_group_metrics:
            item[f'{metric}_correct'] = correct_group_metrics[metric]

        for metric in incorrect_group_metrics:
            item[f'{metric}_incorrect'] = incorrect_group_metrics[metric]

        return item
    return map_item


def compute_all_metrics(dataset, text_per_image, text_column, cpu_cores):
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
    evaluate.enable_progress_bar()

    metrics = {
        "rouge": evaluate.load("rouge"),
        "bleu": evaluate.load("bleu"),
        "meteor": evaluate.load("meteor"),
        "bertscore": evaluate.load("bertscore"),
    }

    try:
        metrics["cider"] = Evaluate(metrics=["cider_d"])
    except:
        print(f"CIDEr-D metric is not available, skipping...")
        metrics["cider"] = None
    else:
        print(f"CIDEr-D metric is available, importing...")

    print("\nProcessing Metrics")
    return dataset.map(
        compute_metrics_sample(metrics),
        batched=False,
        num_proc=int(os.cpu_count() * eval(str(cpu_cores))) if cpu_cores != "None" else None
    )


def evaluate_captions(
        dataset,
        text_per_image,
        text_column,
        results_dir,
        cpu_cores="0.5"
    ):
    """
    Generate evaluation results for a model, saving metrics and training history to CSV files.

    Parameters
    ----------
    dataset : Dataset
        The processed dataset for evaluation.
    dataset : Dataset
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
    results = compute_all_metrics(
        dataset=dataset,
        text_per_image=text_per_image,
        text_column=text_column,
        cpu_cores=cpu_cores
    )

    pd.DataFrame(results).to_csv(
        path_or_buf=os.path.join(results_dir, f'results.csv'),
        index=False
    )
