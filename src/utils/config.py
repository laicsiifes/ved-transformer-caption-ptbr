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
    encoder_decoder_model = f'{setups["config"]["encoder"]}-{setups["config"]["decoder"]}'
    dataset = setups["config"]["dataset"]

    setups["config"] = {
        "encoder_decoder_model": encoder_decoder_model,
        "encoder_model": setups["encoder"][setups["config"]["encoder"]],
        "decoder_model": setups["decoder"][setups["config"]["decoder"]],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_dir": f"../models/{encoder_decoder_model}/{dataset}/model/",
        "results_dir": f"../models/{encoder_decoder_model}/{dataset}/results/",
        "max_length": setups["config"]["max_length"],
        "batch_size": setups["config"]["batch_size"],
        "evaluate_from_model": setups["config"]["evaluate_from_model"],
        "turn_off_computer": setups["config"]["turn_off_computer"]
    }
    setups["config"]["data_dir"] = os.path.join("../data", dataset)
    setups["training_args"]["output_dir"] = f"../models/{encoder_decoder_model}/artifacts/"

    return setups


def configure_model_and_tokenizer(
        encoder_model=None,
        decoder_model=None,
        model_dir=None,
        device="cpu",
        max_new_tokens=32
    ):
    """
    Configure and initialize the vision encoder-decoder model, tokenizer, and image processor.

    Parameters
    ----------
    encoder_model : str, optional
        The name or path of the encoder model. If not provided, a default encoder is used.
    decoder_model : str, optional
        The name or path of the decoder model. Required for initializing the tokenizer.
    model_dir : str, optional
        The directory from which to load a pre-trained model. If specified, `encoder_model` and `decoder_model` are ignored.
    device : str
        The device to run the model on, either "cpu" or "cuda".
    max_new_tokens : int
        The maximum number of new tokens to generate in the model's generation phase.

    Returns
    -------
    tuple
        A tuple containing the initialized model, tokenizer, and image processor.
    """
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device) \
         if model_dir \
       else VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model).to(device)

    tokenizer = AutoTokenizer.from_pretrained(decoder_model)
    image_processor = ViTImageProcessor.from_pretrained(encoder_model)

    if "gpt2" in decoder_model:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
    else:
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.max_new_tokens = max_new_tokens

    return model, tokenizer, image_processor