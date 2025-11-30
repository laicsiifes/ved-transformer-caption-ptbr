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

from utils.config import config_vars

from utils.data_processing import (
    load_datasets,
    generate_grouped_dataset
)
from utils.eval_captions import evaluate_captions

from pprint import pprint
from dotenv import load_dotenv
from huggingface_hub import login


def analyze(config):
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
    # Load datasets from HuggingFace Hub
    dataset_native, dataset_translated, dataset = load_datasets(
        data_dir=config["test_data_dir"],
        step='eval',
        hf_dataset=config["hf_dataset"]
    )

    test_dataset = generate_grouped_dataset(
        dataset_native=dataset_native,
        dataset_translated=dataset_translated,
        dataset=dataset,
        correct_sample_size=config["correct_sample_size"],
        incorrect_sample_size=config["incorrect_sample_size"],
        reproducible=config["reproducible"],
        use_control_as_incorrect=config["use_control_as_incorrect"],
        replacement=config["replacement"]
    )

    print('\nDataset')
    if dataset:
        print(f'\tLength: {len(dataset)}\n')
    else:
        print(f'\tNative: {len(dataset_native)}\n')
        print(f'\tTranslated: {len(dataset_translated)}\n')
    print('\nEvaluation Info')
    print(f'\tDataset: {config["hf_dataset"]}')
    print(f'\tProportion: {config["correct_sample_size"]} corrects vs. {config["incorrect_sample_size"]} incorrects\n')

    evaluate_captions(
        dataset=test_dataset,
        text_per_image=config["text_per_image"],
        text_column=config["text_column"],
        results_dir=config["results_dir"],
        cpu_cores=config["cpu_cores"]
    )


if __name__ == "__main__":
    """
    Main function that loads configurations from a YAML file and either evaluates a model directly on a test dataset
    or evaluates based on stored predictions, based on the configuration.
    """
    load_dotenv(dotenv_path="../.env")
    login(os.getenv("HF_API_KEY"))

    with open('config.yaml', 'r') as file:
        setups = config_vars(yaml.safe_load(file))

    print('\nConfiguration:', end='\t')
    pprint(setups)

    if not os.path.exists(setups["config"]['results_dir']):
        os.makedirs(setups["config"]['results_dir'])

    analyze(config=setups["config"])

    if setups["config"]["turn_off_computer"]:
        print('\nTurning off computer ...')
        time.sleep(2 * 60)
        os.system('shutdown -h now')
