"""
This module `metrics.py` provides various functions to handle metric calculations,
decoding, and result generation for evaluating an image captioning model. The functions
within are designed to work together to facilitate the evaluation of model outputs against
ground truth data, using both model-based and non-model-based metrics.

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
batch_decode_filter(tokens_ids, tokenizer)
    Decodes a batch of token IDs, replacing specific token IDs designated for ignoring
    with the tokenizer's pad token ID.

compute_metrics(eval_pred, tokenizer)
    Computes and returns evaluation metrics based on predictions and actual labels.

no_model_metrics(predictions, labels, metrics)
    Calculates non-model-based metrics (e.g., ROUGE, BLEU, METEOR) for a list of predictions
    against references.

define_no_model_metrics_dict(predictions, labels, metrics)
    Initializes a dictionary to store results of various non-model-based metrics for image
    captioning.

calculate_individual_metrics(predictions, labels, metrics, images_names)
    Calculates individual metrics for each image-caption pair and compiles results including
    BERTScore.

evaluate_metrics(predictions, labels, images_names)
    Aggregates and computes a comprehensive set of metrics including individual and overall
    evaluations.

get_evaluation_metrics(dataset, raw_dataset, model, config, collate_fn, tokenizer, generate_args)
    Orchestrates the evaluation of the model, generating metrics for a provided dataset based on
    the model's predictions.

generate_results(dataset, raw_dataset, model, config, collate_fn, tokenizer, generate_args, trainer)
    Generates final results and metrics for the test dataset, typically after model training and
    evaluation phases.
"""
import evaluate
import os
import re
import gc
import open_clip
import torch

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
# from aac_metrics import Evaluate


def batch_decode_filter(tokens_ids, tokenizer):
    """
    Decodes a batch of token IDs to their corresponding strings, replacing specific token IDs
    (typically used for padding or ignored tokens in loss computation) with the tokenizer's pad token ID
    before decoding.

    Parameters
    ----------
    tokens_ids : ndarray or Tensor
        An array or tensor of token IDs, where certain IDs may be designated as ignored (typically -100).
    tokenizer : PreTrainedTokenizer
        A tokenizer for decoding IDs back to strings.

    Returns
    -------
    list of str
        A list of decoded strings corresponding to the input token IDs, with ignored token IDs replaced by pad tokens.
    """
    return tokenizer.batch_decode(
        np.where(tokens_ids != -100, tokens_ids, tokenizer.pad_token_id)
    )


def compute_metrics(eval_pred, tokenizer):
    """
    Compute evaluation metric ROUGE for predictions.

    Parameters
    ----------
    eval_pred : EvalPrediction
        An object containing the model predictions and corresponding labels.
    tokenizer : PreTrainedTokenizer
        A tokenizer for decoding IDs back to strings.

    Returns
    -------
    dict
        A dictionary containing computed metrics:
        - Various ROUGE scores (e.g., rouge1, rouge2, rougeL, rougeLsum).
    """
    rouge = evaluate.load("rouge")

    predictions = batch_decode_filter(eval_pred.predictions, tokenizer)
    labels = batch_decode_filter(eval_pred.label_ids, tokenizer)

    return rouge.compute(
        predictions=predictions,
        references=labels,
        use_stemmer=False,
        tokenizer=lambda x: x.split()
    )


def clip_score(
        reference,
        candidate,
        kind,
        tokenizer,
        preprocess,
        model,
        device='cuda',
        w=2.5
    ):
    """
    Compute the CLIP-based similarity score between reference and candidate inputs.

    Parameters
    ----------
    reference : PIL.Image.Image or str
        The reference input, which can be an image (for 'img-txt' kind) or a text string.
    candidate : str
        The candidate text to compare with the reference.
    kind : str
        Type of comparison, either 'img-txt' for image-to-text or 'txt-txt' for text-to-text.
    tokenizer : Callable
        The tokenizer function used to preprocess the text input.
    preprocess : Callable
        The preprocessing function for images, used if `kind` is 'img-txt'.
    model : CLIPModel
        The CLIP model used to encode images and text.
    device : str, optional
        The device ('cuda' or 'cpu') for processing inputs (default is 'cuda').
    w : float, optional
        Weight factor for scaling the similarity score (default is 2.5).

    Returns
    -------
    float
        The weighted similarity score between the reference and candidate.
    """
    candidate = tokenizer(candidate).to(device)

    if kind == 'img-txt':
        reference = reference.convert('RGB')
        reference = preprocess(reference).unsqueeze(0).to(device)
    else:
        reference = tokenizer(reference).to(device)

    with torch.no_grad():
        if kind == 'img-txt':
            reference_features = model.encode_image(reference)
            candidate_features = model.encode_text(candidate)
        else:
            reference_features = model.encode_text(reference)
            candidate_features = model.encode_text(candidate)

    reference_features /= reference_features.norm(dim=-1, keepdim=True)
    candidate_features /= candidate_features.norm(dim=-1, keepdim=True)

    similarity = torch.matmul(reference_features, candidate_features.T)
    return w * max(similarity.item(), 0)


def ref_clip_score(image_score, text_scores):
    """
    Calculate the harmonic mean of the image and text CLIP scores.

    Parameters
    ----------
    image_score : float
        The CLIP score for the reference image.
    text_scores : list of float
        A list of CLIP scores for the reference text(s).

    Returns
    -------
    float
        The harmonic mean of the image and the highest text score.
    """
    text_score = max(text_scores)
    return 2 * (image_score * text_score) / (image_score + text_score)


def compute_clip_scores(predictions, labels, dataset):
    """
    Compute CLIP-based similarity scores (CLIPScore and RefCLIPScore) for image-caption pairs.

    Parameters
    ----------
    predictions : list of str
        A list of predicted captions.
    labels : list of list of str
        A list of lists, where each sublist contains reference captions for each image.
    dataset : Dataset
        A list of images corresponding to each prediction.

    Returns
    -------
    dict
        A dictionary with two keys containing CLIPScore and RefCLIPScore similarity scores.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = open_clip.create_model_from_pretrained('hf-hub:hiaac-nlp/CAPIVARA')
    model.to(device)
    tokenizer = open_clip.get_tokenizer('hf-hub:hiaac-nlp/CAPIVARA')
    
    scores = {
        'clipscore': [],
        'ref_clipscore': []
    }
    
    with tqdm(total=len(predictions)) as pbar:
        for prediction, label, batch in zip(predictions, labels, dataset):
            pbar.set_description("Eval. CLIPScore")
            img_score = clip_score(
                reference=batch["image"],
                candidate=prediction,
                kind='img-txt',
                tokenizer=tokenizer,
                preprocess=preprocess,
                model=model
            )
            txt_scores = [
                clip_score(
                    reference=reference,
                    candidate=prediction,
                    kind='txt-txt',
                    tokenizer=tokenizer,
                    preprocess=preprocess,
                    model=model
                ) for reference in label
            ]
            score = ref_clip_score(img_score, txt_scores)
            scores['clipscore'].append(img_score)
            scores['ref_clipscore'].append(score)
            pbar.update(1)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return scores


def no_model_metrics(predictions, labels, metrics):
    """
    Computes non-model-based metrics (ROUGE, BLEU, and METEOR) for a list of predicted and reference captions.
    This function handles exceptions specifically for the BLEU metric calculation, where a `ZeroDivisionError`
    may occur if the prediction is empty or does not align with the expected format.

    Parameters
    ----------
    predictions : list of str
        A list of predicted captions generated by the model.
    labels : list of str
        A list of ground truth captions for the images.
    metrics : dict
        A dictionary containing preloaded metric evaluators:
        - 'rouge': Loaded ROUGE evaluator.
        - 'bleu': Loaded BLEU evaluator.
        - 'meteor': Loaded METEOR evaluator.

    Returns
    -------
    dict
        A dictionary containing the results from the ROUGE, BLEU, and METEOR computations.
        - Keys like 'rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor' store the computed scores.
    """
    rouge = metrics["rouge"]
    bleu = metrics["bleu"]
    meteor = metrics["meteor"]
    ic_metrics = metrics["ic_metrics"] if "ic_metrics" in metrics.keys() else None

    rouge_result = rouge.compute(
        predictions=predictions,
        references=labels,
        use_stemmer=False,
        tokenizer=lambda x: x.split()
    )

    meteor_result = meteor.compute(predictions=predictions, references=labels)

    ic_scores = {k: float(v) for k, v in ic_metrics(predictions, labels)[0].items()} \
                if ic_metrics else {}

    """
        Bleu raised an exception `ZeroDivisionError` during the training of some models.
    """
    try:
        bleu_result = bleu.compute(
            predictions=predictions,
            references=labels,
            tokenizer=lambda x: x.split()
        )
    except ZeroDivisionError:
        # bleu_result = {"bleu": 0.0}
        bleu_result = {
            "bleu": 0.0,
            "precisions": [],
            "brevity_penalty": 0.0,
            "length_ratio": 0.0,
            "translation_length": 0.0,
            "reference_length": 0.0
        }

    return {
        **rouge_result,
        **bleu_result,
        **meteor_result,
        **ic_scores
    }


def define_no_model_metrics_dict(predictions, labels, metrics):
    """
    Initializes a dictionary for storing results of various non-model-based metrics for image captioning.

    Parameters
    ----------
    predictions : list of str
        A list of predicted captions generated by the model.
    labels : list of str
        A list of ground truth captions for the images.
    metrics : dict
        A dictionary of loaded metrics functions.

    Returns
    -------
    dict
        A dictionary where each key represents a metric from the non-model-based evaluations and
        each value is an empty list intended to store results for each image-caption pair.
    """
    individual_results = {}
    for prediction, label in zip(predictions[:1], labels[:1]):
        no_model_metrics_results = no_model_metrics([prediction], [label], metrics)
        for key in no_model_metrics_results.keys():
            individual_results[key] = []
    return individual_results


def calculate_individual_metrics(predictions, labels, metrics, images_names, dataset):
    """
    Calculates individual performance metrics for a list of image caption predictions versus the true labels.
    This function computes BERTScore and other non-model based metrics for each image-caption pair,
    handling situations where multiple predictions may correspond to a single image.

    Parameters
    ----------
    predictions : list of str
        A list of predicted captions generated by the model.
    labels : list of str
        A list of ground truth captions for the images.
    metrics : dict
        A dictionary of loaded metrics functions.
    images_names : list of str
        A list of filenames corresponding to the images. Used for associating results with images.

    Returns
    -------
    dict
        A dictionary containing:
        - 'filename': List of image filenames.
        - Metrics keys (e.g., 'bertscore_precision', 'bertscore_recall', 'bertscore_f1', 'bertscore_hashcode'):
          Values are lists with the computed metric for each image-caption pair.
        - Other metric keys as computed by `no_model_metrics`.
    """
    bertscore = metrics["bertscore"]

    individual_results = define_no_model_metrics_dict(predictions, labels, metrics)

    print("Eval. BERTScore")
    bertscore_result = bertscore.compute(
        predictions=[' '.join(prediction.split()[:200]) for prediction in predictions],
        references=[[' '.join(unit.split()[:200]) for unit in label] for label in labels],
        model_type="neuralmind/bert-base-portuguese-cased",
        num_layers=12
    )
    individual_results["filename"] = images_names
    individual_results["bertscore_precision"] = bertscore_result["precision"]
    individual_results["bertscore_recall"] = bertscore_result["recall"]
    individual_results["bertscore_f1"] = bertscore_result["f1"]
    individual_results["bertscore_hashcode"] = bertscore_result["hashcode"]

    clipscore_results = compute_clip_scores(predictions, labels, dataset)
    individual_results['clipscore'] = clipscore_results['clipscore']
    individual_results['ref_clipscore'] = clipscore_results['ref_clipscore']

    for prediction, label in tqdm(zip(predictions, labels), desc="Eval. No-Model Metrics"):
        no_model_metrics_results = no_model_metrics([prediction], [label], metrics)
        for k, v in no_model_metrics_results.items():
            individual_results[k].append(v)

    return individual_results


def evaluate_metrics(predictions, labels, images_names, dataset):
    """
    Computes a series of evaluation metrics for a set of image captioning predictions
    compared to ground truth labels.

    Parameters
    ----------
    predictions : list of str
        A list of predicted captions generated by the model.
    labels : list of str
        A list of ground truth captions for the images.
    images_names : list of str
        A list of filenames corresponding to the images in the dataset.

    Returns
    -------
    dict
        A dictionary containing three key-value pairs:
        - "individual_metrics": A dictionary of metrics calculated for each image.
        - "original_metrics": Aggregated metrics calculated from individual metrics.
        - "sample_metrics": Mean and standard deviation of each metric across the dataset.
    """
    metrics = {
        "rouge": evaluate.load("rouge"),
        "bleu": evaluate.load("bleu"),
        "meteor": evaluate.load("meteor"),
        "bertscore": evaluate.load("bertscore"),
    }
  
    if len(images_names) != len(labels):
        labels = [labels[i * 5:(i + 1) * 5] for i, _ in enumerate(images_names)]
        predictions = [predictions[i * 5] for i, _ in enumerate(images_names)]

    print('Individual Metrics')
    individual_metrics = calculate_individual_metrics(
        predictions,
        labels,
        metrics,
        images_names,
        dataset
    )

    # metrics["ic_metrics"] = Evaluate(metrics=[
    #     "cider_d",
    #     "spice",
    #     "spider"
    # ])

    print('Total Metrics')
    original_metrics = {
        "bertscore_precision": np.mean(individual_metrics["bertscore_precision"]),
        "bertscore_recall": np.mean(individual_metrics["bertscore_recall"]),
        "bertscore_f1": np.mean(individual_metrics["bertscore_f1"]),
        "clipscore": np.mean(individual_metrics["clipscore"]),
        "ref_clipscore": np.mean(individual_metrics["ref_clipscore"]),
        **no_model_metrics(predictions, labels, metrics)
    }

    sample_metrics = {}

    print('Mean and Standard Deviation')
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


def get_evaluation_metrics(
        dataset,
        raw_dataset,
        model,
        config,
        collate_fn,
        tokenizer,
        generate_args
    ):
    """
    Evaluates an image captioning model by generating predictions for a given dataset and comparing these
    predictions against true labels.

    Parameters
    ----------
    dataset : Dataset
        The test dataset, generally, to evaluate the model on.
    raw_dataset : Dataset
        The raw test dataset, generally, to use the raw information (e.g. images names).
    model : PreTrainedModel
        The model that was evaluated.
    config : dict
        A dictionary containing configuration parameters including directories for saving
        the model and results.
    collate_fn : callable
        The function to collate data items into batches.
    tokenizer : PreTrainedTokenizer
        A tokenizer for decoding IDs back to strings.
    generate_args : dict
        Arguments used for generating captions, such as maximum length and number of beams
        for beam search.

    Returns
    -------
    dict
        A dictionary containing the following key-value pairs:
        - "prediction_text": List of predicted captions.
        - "label_text": List of true captions.
        - "filename": List of filenames corresponding to the images.
        - Evaluation metrics: Additional entries as returned by `evaluate_metrics` function.
    """
    model.eval()
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=config["batch_size"])
    predictions = []

    for batch in tqdm(dataloader, "Testing Split Evaluation"):
        outputs = model.generate(
            pixel_values=batch["pixel_values"],
            **generate_args
        )
        predictions.extend(outputs.tolist())

    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    caption_key = 'text' if 'text' in raw_dataset.features.keys() else 'caption'
    
    # the filenames must appear 5 times each one (flickr)
    images_names = raw_dataset['filename'] if caption_key == 'text' \
        else [filename for filename in raw_dataset['filename'] for _ in range(5)]

    # add space between punctuations (flickr)
    labels = [ 
        ' '.join(re.findall(r"[\w']+|[.,!?;]", sentence)) \
            for sentences in raw_dataset[caption_key] \
                for sentence in sentences
        ] if len(images_names) != len(raw_dataset[caption_key]) \
        else [[el] for el in raw_dataset[caption_key]]

    # Save predictions before evaluation
    pd.DataFrame({
        "prediction_text": predictions,
        "label_text": labels,
        "filename": images_names,
    }).to_json(
        os.path.join(config['results_dir'], f'predictions.json'),
        orient="records"
    )

    return {
        "prediction_text": predictions,
        "label_text": labels,
        "filename": images_names,
        **evaluate_metrics(
            predictions=predictions, 
            labels=labels,
            images_names=raw_dataset['filename'],
            dataset=raw_dataset
        )
    }


def generate_results(
        dataset,
        raw_dataset,
        model,
        config,
        collate_fn,
        tokenizer,
        image_processor,
        generate_args,
        trainer=None
    ):
    """
    Evaluate the model on the test dataset, save the model, and write evaluation metrics and
    predictions to disk.

    Parameters
    ----------
    dataset : Dataset
        The test dataset, generally, to evaluate the model on.
    raw_dataset : Dataset
        The raw test dataset, generally, to use the raw information (e.g. images names).
    model : PreTrainedModel
        The model that was evaluated.
    config : dict
        A dictionary containing configuration parameters including directories for saving the
        model and results.
    collate_fn : callable
        The function to collate data items into batches.
    tokenizer : PreTrainedTokenizer
        A tokenizer for decoding IDs back to strings.
    image_processor : ImageProcessingMixin
        An image processor to prepare input features for vision models and post processing
        their outputs.
    trainer : Trainer
        The Hugging Face Trainer instance used for evaluation.

    Returns
    -------
    None
        This function does not return a value. It saves the model, training log history,
        predictions, and evaluation metrics to the specified directories in the configuration.
    """
    if trainer:
        model.save_pretrained(config["model_dir"])
        tokenizer.save_pretrained(config["model_dir"])
        image_processor.save_pretrained(config["model_dir"])

        pd.DataFrame(trainer.state.log_history).to_csv(
            path_or_buf=os.path.join(config['results_dir'], f'training_log_history.csv'),
            index=False
        )

    results = get_evaluation_metrics(
        dataset=dataset,
        raw_dataset=raw_dataset,
        model=model,
        config=config,
        collate_fn=collate_fn,
        tokenizer=tokenizer,
        generate_args=generate_args
    )

    pd.DataFrame(results["individual_metrics"]).to_csv(
        path_or_buf=os.path.join(config['results_dir'], f'individual_eval_metrics.csv'),
        index=False
    )

    pd.DataFrame(results["original_metrics"]).to_csv(
        path_or_buf=os.path.join(config['results_dir'], f'original_eval_metrics.csv'),
        index=False
    )

    pd.DataFrame(results["sample_metrics"], index=[0]).to_csv(
        path_or_buf=os.path.join(config['results_dir'], f'sample_eval_metrics.csv'),
        index=False
    )