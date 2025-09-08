"""
Module for Vision-Language Models
============================================

This module provides functions for ...
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


def analyze(config, generate_args):
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
    _, _, test_ds = load_datasets(
        data_dir=config["test_data_dir"],
        step='eval',
        hf_dataset=config["hf_test_set"],
        dataset_from_hub=config["dataset_from_hub"]
    )

    evaluate_predictions(
        raw_dataset=test_ds,
        predictions=,
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
