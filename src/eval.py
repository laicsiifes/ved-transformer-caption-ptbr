"""
This module `eval.py` is dedicated to evaluating the performance of an image captioning model.
It includes functions for both generating predictions from a model and evaluating directly from 
pre-configured predictions. The module handles various aspects of evaluation, including configuring
the evaluation environment based on user-defined settings and applying evaluation metrics to
assess model accuracy and other performance indicators.

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
evaluate_from_model(config, generate_args)
    Conducts evaluation by generating predictions using the model configured with the specified arguments
    and config settings.

evaluate_from_predictions(config)
    Performs evaluation based on a set of pre-existing predictions, using the specified config settings
    to guide the evaluation process.
"""
import os
import yaml
import pandas as pd

from utils.config import config_vars, configure_model_and_tokenizer
from utils.data_processing import (
    load_datasets,
    preprocess,
    collate_fn,
    transform_datasets
)
from utils.metrics import evaluate_metrics, generate_results
from pprint import pprint


def evaluate_from_model(config, generate_args):
    """
    Evaluate the model on a test dataset using the given configuration and training arguments.

    Parameters
    ----------
    config : dict
        A dictionary containing configurations for the model, tokenizer, image processor, and evaluation settings.
    generate_args : dict
        Arguments used for generating captions, such as maximum length and number of beams for beam search.

    Returns
    -------
    None
        Writes the evaluation metrics to a CSV file in the specified results directory.

    """
    model, tokenizer, image_processor = configure_model_and_tokenizer(
        encoder_model=config["encoder_model"],
        decoder_model=config["decoder_model"],
        model_dir=config["model_dir"],
        device=config["device"],
        max_new_tokens=config["max_length"]
    )

    preprocess_fn = lambda items: preprocess(
        items=items,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=config["device"],
        max_length=config["max_length"]
    )

    _, _, test_ds = load_datasets(config["data_dir"], step='eval')
    _, _, test_dataset = transform_datasets(
        test_ds=test_ds,
        preprocess_fn=preprocess_fn,
        step='eval'
    )

    generate_results(
        dataset=test_dataset,
        raw_dataset=test_ds,
        model=model,
        config=config,
        collate_fn=collate_fn,
        tokenizer=tokenizer,
        image_processor=image_processor,
        generate_args=generate_args
    )


def evaluate_from_predictions(config):
    """
    Evaluate the model based on predictions stored in a JSON file.

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
    eval_preds = pd.read_json(os.path.join(config['results_dir'], "predictions.json"))
    
    if "flickr" in config["data_dir"]:
        filenames = eval_preds['filename'].values.tolist()
        image_names = [filenames[i] for i in range(0, len(filenames), 5)]
    else:
        image_names = eval_preds['filename'].values.tolist()

    results = evaluate_metrics(
        predictions=eval_preds["prediction_text"].values.tolist(),
        labels=eval_preds["label_text"].values.tolist(),
        images_names=image_names
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


if __name__ == "__main__":
    """
    Main function that loads configurations from a YAML file and either evaluates a model directly on a test dataset
    or evaluates based on stored predictions, based on the configuration.
    """
    with open('../config.yml', 'r') as file:
        setups = config_vars(yaml.safe_load(file))

    print('\nConfiguration:', end='\t')
    pprint(setups)

    if not os.path.exists(setups["config"]['results_dir']):
        os.makedirs(setups["config"]['results_dir'])

    if setups["config"]["evaluate_from_model"]:
        evaluate_from_model(config=setups["config"], generate_args=setups["generate_args"])
    else:
        evaluate_from_predictions(config=setups["config"])