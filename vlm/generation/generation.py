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
generate_captions(dataset, model, config, collate_fn, tokenizer, generate_args)
    Orchestrates the evaluation of the model, generating metrics for a provided dataset based on
    the model's predictions.
"""

import os
import gc
import re
import ast
import torch
import openai
import time
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm


def batch_generation_from_API(
        dataset,
        config,
        temperature,
        top_p
    ):
    n_keys = int(os.environ.get(f"N_MODEL_API_KEY"))
    API_keys = [os.environ.get(f"MODEL_API_KEY_{i+1}") for i in range(n_keys)]
    API_provider = os.environ.get(f"MODEL_API_PROVIDER")

    current_key = 0
    max_tries = 1
    retry_weight = 5 # the retry_time = n_retry*retry_weight recalculated in each retry

    client = openai.OpenAI(
        api_key=API_keys[current_key],
        base_url=API_provider,
    )

    n_caps = config['text_per_image']
    n_init = 0
    predictions = []

    # Check if there is any saved prediction and continue from the last saved one
    preds_path = os.path.join(config['results_dir'], "predictions.json")
    if os.path.exists(preds_path):
        print("Skipping Already Generated Examples")
        eval_preds = pd.read_json(preds_path, lines=True)
        predictions = eval_preds['prediction_text'].values.tolist()
        n_init = int(len(eval_preds)/n_caps)

    # Proceed only if there are examples that were not generated
    if n_init*n_caps < len(dataset):
        dataset = dataset.select(range(n_init*n_caps, len(dataset)))

        # Progress bar considering batch_size equals to 1
        with tqdm(total=int(len(dataset)/n_caps)) as pbar:
            i = 0

            # Process the dataset to generate the captions of each example
            for batch in dataset:

                # Skip each n reference captions to generate from same image once
                if not i % n_caps:
                    pbar.set_description("Generating Captions")
                    n_retry = 0
                    flag_failure = True

                    # In case of failure, retry 5 times. If the failure persists, rotate API key
                    while flag_failure and n_retry < max_tries+1:
                        try:
                            outputs = client.chat.completions.create(
                                model=config["model_id"],
                                messages=batch["message"],
                                temperature=temperature,
                                top_p=top_p
                            )

                            prediction = [outputs.choices[0].message.content]*n_caps

                            # Save predictions incrementally
                            pd.DataFrame({
                                "prediction_text": prediction,
                                "label_text": dataset["answer"][i:i+n_caps],
                                "prompt": dataset["question"][i:i+n_caps],
                                "filename": dataset['filename'][i:i+n_caps],
                            }).to_json(
                                os.path.join(config['results_dir'], f'predictions.json'),
                                orient="records",
                                lines=True,
                                mode='a'
                            )

                            predictions.extend(prediction)
                            flag_failure = False
                            time.sleep(1)

                        # Retry message with sleep time
                        except Exception as e:
                            n_retry+=1
                            if n_retry==1:
                                print()
                            if n_retry < max_tries+1:
                                match = re.search(r"{.*}", str(e))
                                message = ast.literal_eval(match.group(0))['error']['message'] if match else e
                                retry_time = n_retry*retry_weight
                                print(f"{message}. Retrying in {retry_time}s. Attempt {n_retry}/{max_tries}.")
                                time.sleep(retry_time)
                        
                        # API key rotation for generation
                        if not n_retry < max_tries+1 and flag_failure:
                            current_key = current_key+1 if current_key<n_keys-1 else 0
                            print(f"Rotating API key. Key number {current_key+1}/{n_keys}.")
                            client = openai.OpenAI(
                                api_key=API_keys[current_key],
                                base_url=API_provider,
                            )
                            n_retry = 0
                    pbar.update(1)
                i+=1

    return predictions


def batch_generation(
        dataset,
        model,
        config,
        collate_fn,
        processor,
        generate_args
    ):
    """
    Evaluates an image captioning model by generating predictions for a given dataset and comparing these
    predictions against true labels.

    Parameters
    ----------
    dataset : Dataset
        The test dataset, generally, to evaluate the model on.
    model : PreTrainedModel
        The model that was evaluated.
    config : dict
        A dictionary containing configuration parameters including directories for saving
        the model and results.
    collate_fn : Callable
        The function to collate data items into batches.
    processor : ProcessorMixin
        Processor for handling model inputs and tokenization.
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
    n_caps = config['text_per_image']
    n_init = 0
    predictions = []

    # Check if there is any saved prediction and continue from the last saved one
    preds_path = os.path.join(config['results_dir'], "predictions.json")
    if os.path.exists(preds_path):
        print("Skipping Already Generated Examples")
        eval_preds = pd.read_json(preds_path, lines=True)
        predictions = eval_preds['prediction_text'].values.tolist()
        n_init = int(len(eval_preds)/n_caps)

    # Proceed only if there are examples that were not generated
    if n_init*n_caps < len(dataset):
        dataset = dataset.select(range(n_init*n_caps, len(dataset)))

        dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=config["batch_size"])

        # Progress bar considering batch_size equals to 1
        with tqdm(total=int(len(dataset)/n_caps)) as pbar:
            i = 0

            # Process the dataset to generate the captions of each example
            for batch in dataloader:

                # Skip each n reference captions to generate from same image once
                if not i % n_caps:
                    pbar.set_description("Generating Captions")

                    outputs = model.generate(
                        **batch,
                        **generate_args
                    )

                    prediction = outputs[:, batch['input_ids'].shape[1]:].tolist()
                    prediction = processor.batch_decode(prediction, skip_special_tokens=True)*n_caps

                    # Save predictions incrementally
                    pd.DataFrame({
                        "prediction_text": prediction,
                        "label_text": dataset["answer"][i:i+n_caps],
                        "prompt": dataset["question"][i:i+n_caps],
                        "filename": dataset['filename'][i:i+n_caps],
                    }).to_json(
                        os.path.join(config['results_dir'], f'predictions.json'),
                        orient="records",
                        lines=True,
                        mode='a'
                    )
                    predictions.extend(prediction)
                    pbar.update(1)
                i+=1

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return predictions