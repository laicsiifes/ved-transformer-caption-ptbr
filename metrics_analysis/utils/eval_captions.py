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

from evaluation.metrics import (
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

def compute_invidiual_metric(predictions, labels, scorer):
    result = {}

    # Select the computing metric funtion
    if "rouge" in scorer.name.lower():
        compute = compute_rouge_scores
    elif "meteor" in scorer.name.lower():
        compute = compute_meteor_scores
    elif "bleu" in scorer.name.lower():
        compute = compute_bleu_scores
    else:
        compute = None

    # If there is a compute function, score the predictions 
    if compute:
        with tqdm(total=len(predictions)) as pbar:
            pbar.set_description(f"Eval. {scorer.name.upper()}")

            # Create an empty dict with empty lists to add the by-example scores
            result = {
                k:[] for k in list(compute([''], [''], scorer).keys())
            }

            # Compute the score to each example
            for prediction, label in zip(predictions, labels):
                individual_result = compute([prediction], [label], scorer)

                # Append the individual scores to the result dict
                for key in individual_result:
                    result[key].append(individual_result[key])
                pbar.update(1)
    else:
        print("`scorer` parameter is not set correctly, returning empty metric dict.")
        
    return result


def compute_individual_metrics(predictions, labels, metrics, images_names, dataset):
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
    bertscore_result = compute_bert_scores(predictions, labels, metrics["bertscore"])
    clipscore_result = compute_clip_scores(predictions, labels, dataset)

    # Computing example-by-example results for ROUGE, METEOR and BLEU
    rouge_result  = compute_invidiual_metric(predictions, labels, metrics["rouge"])
    meteor_result = compute_invidiual_metric(predictions, labels, metrics["meteor"])
    bleu_result   = compute_invidiual_metric(predictions, labels, metrics["bleu"])

    return {
        "filename": images_names,
        **bertscore_result,
        **clipscore_result,
        **rouge_result,
        **meteor_result,
        **bleu_result
    }


def compute_all_metrics(predictions, dataset, text_per_image, text_column):
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

    # One image has one predicted caption, but it can hold N reference caption
    # Group them in list of lists
    if not isinstance(labels[0], list):
        labels = [labels[i * text_per_image:(i + 1) * text_per_image] for i, _ in enumerate(images_names)]
    predictions = [predictions[i * text_per_image] for i, _ in enumerate(images_names)]

    print("Computing Metrics Individually")
    individual_metrics = compute_individual_metrics(
        predictions, labels, metrics, images_names, dataset
    )

    print("Computing Metrics Total")
    original_metrics = {
        "bertscore_precision": np.mean(individual_metrics["bertscore_precision"]),
        "bertscore_recall": np.mean(individual_metrics["bertscore_recall"]),
        "bertscore_f1": np.mean(individual_metrics["bertscore_f1"]),
        "clipscore": np.mean(individual_metrics["clipscore"]),
        "ref_clipscore": np.mean(individual_metrics["ref_clipscore"]),
        **compute_rouge_scores(predictions, labels, metrics["rouge"]),
        **compute_bleu_scores(predictions, labels, metrics["bleu"]),
        **compute_meteor_scores(predictions, labels, metrics["meteor"]),
        **compute_cider_scores(
            [re.sub(r"\\.", "", s.encode('unicode_escape').decode()) for s in predictions],
            [[re.sub(r"\\.", "", s.encode('unicode_escape').decode()) for s in label ] for label in labels],
            metrics["cider"]
        )
    }

    sample_metrics = {}

    print("Computing Metrics Mean/Std")
    for metric in [
        "bertscore_precision", "bertscore_recall", "bertscore_f1",
        "clipscore", "ref_clipscore",
        "rouge1", "rouge2", "rougeL", "rougeLsum", "bleu", "meteor"
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
        text_per_image,
        text_column,
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
    results = compute_all_metrics(
        predictions=predictions, 
        dataset=raw_dataset,
        text_per_image=text_per_image,
        text_column=text_column
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