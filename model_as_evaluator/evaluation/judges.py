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
clip_score(reference, candidate, kind, tokenizer, preprocess, model, device, w)
    Compute the CLIP-based similarity score between reference and candidate inputs.

ref_clip_score(image_score, text_scores)
    Calculate the harmonic mean of the image and text CLIP scores.

compute_clip_scores(predictions, labels, images)
    Compute CLIP-based similarity scores (CLIPScore and RefCLIPScore) for image-caption pairs.
"""

import gc
import open_clip
import torch
import ast
import os
import base64

from tqdm import tqdm
from io import BytesIO


def format_llm_message(prediction, label, template):
    joint_labels = " \n".join([f"- {l}" for i, l in enumerate(label)])

    return [{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": template.format(
                    candidate_statements=prediction,
                    target_statements=joint_labels
                )
            }]
        }]


def compute_llm_as_a_judge(predictions, labels, config):
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    outputs = {
        "llm_as_a_judge_score": [],
        "llm_as_a_judge_reason": []
    }

    for prediction, label in zip(predictions, labels):
        output = client.chat.completions.create(
            model=config["model_id"],
            messages=format_message(prediction, label, config["template"]),
            temperature=config["temperature"],
            top_p=config["top_p"]
        )
        output_dict = ast.literal_eval(output)
        outputs["llm_as_a_judge_score"].append(ast.literal_eval(output_dict["score"]))
        outputs["llm_as_a_judge_reason"].append(ast.literal_eval(output_dict["reason"]))

    return outputs


def format_vlm_message(prediction, image, template):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_url = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    return [{
        "role":"user",
        "content": [{
            "type":"text",
            "text":template.format(
                candidate_statements=prediction
            )},{
                "type":"image_url",
                "image_url": {"url": image_url}
        }]
    }]


def compute_vlm_as_a_judge(predictions, images, config):
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    outputs = {
        "vlm_as_a_judge_score": [],
        "vlm_as_a_judge_reason": []
    }

    for prediction, image in zip(predictions, images):
        output = client.chat.completions.create(
            model=config["model_id"],
            messages=format_message(prediction, image, config["template"]),
            temperature=config["temperature"],
            top_p=config["top_p"]
        )
        output_dict = ast.literal_eval(output)
        outputs["vlm_as_a_judge_score"].append(ast.literal_eval(output_dict["score"]))
        outputs["vlm_as_a_judge_reason"].append(ast.literal_eval(output_dict["reason"]))

    return outputs


def compute_rouge_scores(predictions, labels, scorer):
    return scorer.compute(
            predictions=predictions,
            references=labels,
            use_stemmer=False,
            tokenizer=lambda x: x.split()
        )


def compute_meteor_scores(predictions, labels, scorer):
    return scorer.compute(
            predictions=predictions,
            references=labels
        )


def compute_bleu_scores(predictions, labels, scorer):
    results = {}
    """
        Bleu raised an exception `ZeroDivisionError` during the training of some models.
    """
    try:
        results = scorer.compute(
            predictions=predictions,
            references=labels,
            tokenizer=lambda x: x.split()
        )
        results["precisions"] = str(results["precisions"])
    except ZeroDivisionError:
        results = {
            "bleu": 0.0,
            "precisions": "[]",
            "brevity_penalty": 0.0,
            "length_ratio": 0.0,
            "translation_length": 0.0,
            "reference_length": 0.0
        }
    return results


def compute_cider_scores(predictions, labels, scorer):
    return {
        k: float(v) for k, v in scorer(predictions, labels)[0].items()
    } if scorer else {}


def compute_bert_scores(predictions, labels, bertscore):
    results = {}
    
    print("Eval. BERTScore")
    bertscore_result = bertscore.compute(
        predictions=[' '.join(prediction.split()[:200]) for prediction in predictions],
        references=[[' '.join(unit.split()[:200]) for unit in label] for label in labels],
        model_type="neuralmind/bert-base-portuguese-cased",
        num_layers=12
    )

    results["bertscore_precision"] = bertscore_result["precision"]
    results["bertscore_recall"] = bertscore_result["recall"]
    results["bertscore_f1"] = bertscore_result["f1"]
    results["bertscore_hashcode"] = bertscore_result["hashcode"]

    return results


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


def compute_clair_scores(predictions, labels, scorer)()