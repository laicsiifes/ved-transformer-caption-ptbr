"""
This module `config.py` manages configuration and setup for an image captioning system. It provides
functions to set up environment variables and configure the model and tokenizer for the system,
ensuring that all components are initialized with the correct settings for optimal operation.

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
config_vars(setups)
    Configures and returns environment and operational variables based on a dictionary of setups.

configure_model_and_tokenizer(encoder_model, decoder_model, model_dir, device, max_new_tokens)
    Configures and initializes the model and tokenizer for the image captioning system.
"""
import os
import torch

from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor


def config_vars(setups):
    """
    Generate configuration variables for the encoder-decoder model setup.

    Parameters
    ----------
    setups : dict
        The dictionary containing all configuration variables.

    Returns
    -------
    setups: dict
        A dictionary containing configuration variables such as the names of the encoder and decoder models,
        the computation device, and paths for the output, model, and results directories.
    """
    dataset_name = setups["config"]["dataset"]
    correct_sample_size = setups["config"]["correct_sample_size"]
    incorrect_sample_size = setups["config"]["incorrect_sample_size"]

    if dataset_name == 'both':
        available_datasets = {ds:setups["dataset"][ds]["id"] for ds in setups["dataset"]}
        setups["config"]["hf_dataset"] = available_datasets
        setups["config"]["hf_test_set"] = available_datasets

        setups["config"]["results_dir"] = f"results/both/{correct_sample_size}_vs_{incorrect_sample_size}"
        setups["config"]["image_column"] = "image"
        setups["config"]["text_column"] = "caption"
        setups["config"]["text_per_image"] = 5
        setups["config"]["turn_off_computer"] = setups["config"]["turn_off_computer"]
        setups["config"]["data_dir"] = os.path.join("../data", "both")
        setups["config"]["test_data_dir"] = os.path.join("../data", "both")
        setups["config"]["cpu_cores"] = str(setups["config"]["cpu_cores"])
    else:
        setups["config"]["results_dir"] = f"results/{dataset_name}/{correct_sample_size}_vs_{incorrect_sample_size}"
        setups["config"]["hf_dataset"] = setups["dataset"][dataset_name]["id"]
        setups["config"]["hf_test_set"] = setups["dataset"][dataset_name]["id"]
        setups["config"]["image_column"] = setups["dataset"][dataset_name]["image_column"]
        setups["config"]["text_column"] = setups["dataset"][dataset_name]["text_column"]
        setups["config"]["text_per_image"] = setups["dataset"][dataset_name]["text_per_image"]
        setups["config"]["turn_off_computer"] = setups["config"]["turn_off_computer"]
        setups["config"]["data_dir"] = os.path.join("../data", dataset_name)
        setups["config"]["test_data_dir"] = os.path.join("../data", dataset_name)
        setups["config"]["cpu_cores"] = str(setups["config"]["cpu_cores"])

    return setups