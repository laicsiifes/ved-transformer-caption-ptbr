"""
Inference Module for Vision-Language Models
============================================

This module provides functions for evaluating multimodal vision-language models on image captioning tasks. 
It includes capabilities for evaluating models based on live predictions as well as pre-generated prediction 
files, calculating and saving various metrics to assess model performance.

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
evaluate_from_model(config, qlora_args, generate_args)
    Conducts evaluation by generating predictions using the model configured with the specified arguments
    and config settings.

evaluate_from_predictions(config)
    Performs evaluation based on a set of pre-existing predictions, using the specified config settings
    to guide the evaluation process.
"""

import os
import time
import yaml
import pandas as pd

from config.config import config_vars, configure_model_and_processor

from data_prep.data_processing import (
    load_datasets,
    preprocess, 
    preprocess_for_API,
    transform_datasets
)
from data_prep.data_collator import DataCollatorForGeneration

from evaluation.eval_prediction import evaluate_predictions

from generation.generation import batch_generation, batch_generation_from_API

from pprint import pprint
from dotenv import load_dotenv
from huggingface_hub import login


def evaluate_from_model(config, generate_args):
    """
    Inference the model on a test dataset using the given configuration arguments.

    Parameters
    ----------
    config : dict
        A dictionary containing configurations for the model, processor, and evaluation settings.
    generate_args : dict
        Arguments used for generating captions, such as maximum length.

    Returns
    -------
    None
        Writes the evaluation metrics to a CSV file in the specified results directory.

    """
    model, processor = configure_model_and_processor(model_id=config["model_id"])

    _, _, test_ds = load_datasets(
        data_dir=config["data_dir"],
        step='eval',
        hf_dataset=config["hf_dataset"],
        dataset_from_hub=config["dataset_from_hub"]
    )
 
    _, _, test_dataset = transform_datasets(
        test_ds=test_ds,
        preprocess_fn=preprocess(
            question=config["question"],
            text_per_image=config["text_per_image"],
            image_column=config["image_column"],
            text_column=config["text_column"],
        ),
        step='eval'
    )

    evaluate_predictions(
        raw_dataset=test_ds,
        predictions=batch_generation(
            dataset=test_dataset,
            model=model,
            config=config,
            collate_fn=DataCollatorForGeneration(
                model_id=config["model_id"],
                device=config["device"],
                processor=processor,
                max_length=config["max_length"]
            ),
            processor=processor,
            generate_args=generate_args
        ),
        text_per_image=config["text_per_image"],
        text_column=config["text_column"],
        results_dir=config["results_dir"]
    )


def evaluate_from_API(config):
    _, _, test_ds = load_datasets(
        data_dir=config["data_dir"],
        step='eval',
        hf_dataset=config["hf_dataset"],
        dataset_from_hub=config["dataset_from_hub"]
    )
 
    _, _, test_dataset = transform_datasets(
        test_ds=test_ds,
        preprocess_fn=preprocess_for_API(
            question=config["question"],
            text_per_image=config["text_per_image"],
            image_column=config["image_column"],
            text_column=config["text_column"],
        ),
        step='eval'
    )

    evaluate_predictions(
        raw_dataset=test_ds,
        predictions=batch_generation_from_API(
            dataset=test_dataset,
            config=config,
            temperature=0.6, # default value used by original model
            top_p=0.9        # default value used by original model
        ),
        text_per_image=config["text_per_image"],
        text_column=config["text_column"],
        results_dir=config["results_dir"]
    )


def evaluate_from_predictions(config):
    """
    Inference the model based on predictions stored in a JSON file.

    Parameters
    ----------
    config : dict
        A dictionary containing configurations for the evaluation settings, including the directory
        of the results where the predictions JSON file is stored.

    Returns
    -------
    None
        Writes the evaluation metrics to a CSV file in the specified results directory.
    """
    _, _, test_ds = load_datasets(
        data_dir=config["data_dir"],
        step='eval',
        hf_dataset=config["hf_dataset"],
        dataset_from_hub=config["dataset_from_hub"]
    )

    eval_preds = pd.read_json(os.path.join(config['results_dir'], "predictions.json"), lines=True)
    
    evaluate_predictions(
        raw_dataset=test_ds,
        predictions=eval_preds["prediction_text"].values.tolist(),
        text_per_image=config["text_per_image"],
        text_column=config["text_column"],
        results_dir=config["results_dir"]
    )


if __name__ == "__main__":
    """
    Main function that loads configurations from a YAML file and either evaluates a model directly on a test dataset
    or evaluates based on stored predictions, based on the configuration.
    """
    load_dotenv(dotenv_path="../.env")
    login(os.getenv("HF_API_KEY"))

    with open('config/config_inference.yml', 'r') as file:
        setups = config_vars(yaml.safe_load(file))

    print('\nConfiguration:', end='\t')
    pprint(setups)

    if not os.path.exists(setups["config"]['results_dir']):
        os.makedirs(setups["config"]['results_dir'])

    if setups["config"]["evaluate_from_model"] and \
       setups["config"]["model_name"][-3:].upper() == "API":
        evaluate_from_API(config=setups["config"])
    elif setups["config"]["evaluate_from_model"]:
        evaluate_from_model(
            config=setups["config"],
            generate_args=setups["generate_args"]
        )
    else:
        evaluate_from_predictions(config=setups["config"])

    if setups["config"]["turn_off_computer"]:
        print('\nTurning off computer ...')
        time.sleep(2 * 60)
        os.system('shutdown -h now')
