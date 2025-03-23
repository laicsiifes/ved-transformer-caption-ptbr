"""
Configuration and Model Setup Module
====================================

This module `config.py` manages configuration and setup for an image captioning system. It provides
functions to set up environment variables and configure the model and processor for the system,
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

create_lora_config(model_id, rank, alpha_to_rank_ratio, dropout, freeze_vision_model, is_all_linear)
    Creates a LoRA (Low-Rank Adaptation) configuration based on the specified model 
    and parameters, adjusting for model-specific requirements.

configure_model_and_processor(model_id, use_qlora, use_flash_attention)
    Configures and initializes the model and processor for the image captioning system, 
    with optional support for QLoRA quantization and Flash Attention.
"""

import torch

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
    IntervalStrategy
)
from peft import LoraConfig


def config_vars(setups):
    """
    Configure and update the project setup dictionary with model, dataset, and training parameters.

    Parameters
    ----------
    setups : dict
        A dictionary containing configuration details for the project, 
        including model and dataset specifications.

    Returns
    -------
    dict
        The updated configuration dictionary with additional model, 
        dataset, and training parameters.
    """
    model_name = setups["config"]["model_name"]
    model_id = setups["mllm"][model_name]["id"]
    question = setups["mllm"][model_name]["question"]
    batch_size = setups["mllm"][model_name]["batch_size"]

    # To save/load outputs, models, and results with suffix -ft
    model_name_save = f'{model_name}-ft' if setups["config"]["use_adapters"] else model_name

    dataset_name = setups["config"]["dataset"]
    max_length = setups["dataset"][dataset_name]["max_length"]

    # Operational configs
    setups["config"]["model_id"] = model_id
    setups["config"]["model_name"] = model_name
    setups["config"]["question"] = question
    setups["config"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    setups["config"]["model_dir"] = f"../models/{model_name_save}/{dataset_name}/model/"
    setups["config"]["results_dir"] = f"../models/{model_name_save}/{dataset_name}/results/"
    setups["config"]["hf_dataset"] = setups["dataset"][dataset_name]["id"]
    setups["config"]["data_dir"] = f"../data/{dataset_name}/"
    setups["config"]["max_length"] = max_length
    setups["config"]["batch_size"] = batch_size
    setups["config"]["image_column"] = setups["dataset"][dataset_name]["image_column"]
    setups["config"]["text_column"] = setups["dataset"][dataset_name]["text_column"]
    setups["config"]["text_per_image"] = setups["dataset"][dataset_name]["text_per_image"]
    setups["config"]["use_flash_attention"] = setups["mllm"][model_name]["use_flash_attention"]

    # QLoRa config
    if "qlora_args" in setups["mllm"][model_name]:
        setups["qlora_args"] = setups["mllm"][model_name]["qlora_args"]

    # Model generative config
    setups["generate_args"]["max_new_tokens"] = max_length

     # Training config
    if "training_args" in setups["mllm"][model_name]:
        # Overall config
        setups["training_args"] = setups["mllm"][model_name]["training_args"]
        setups["training_args"]["output_dir"] = f"../models/{model_name_save}/{dataset_name}/artifacts/"
        setups["training_args"]["per_device_train_batch_size"] = batch_size

        # Interval strategies (e.g. "epoch", "steps" or "no")
        if setups["config"]["interval_strategy"] == IntervalStrategy.EPOCH:
            interval_strategy = IntervalStrategy.EPOCH
        elif setups["config"]["interval_strategy"] == IntervalStrategy.STEPS:
            interval_strategy = IntervalStrategy.STEPS
        elif setups["config"]["interval_strategy"] == IntervalStrategy.NO:
            interval_strategy = IntervalStrategy.NO
        else:
            print("No interval strategy option selected, setting to `STEPS`.")
            interval_strategy = IntervalStrategy.STEPS

        monitoring_steps = setups["config"]["monitoring_steps"]
        
        setups["training_args"]["logging_strategy"] = interval_strategy
        setups["training_args"]["logging_steps"] = monitoring_steps
        setups["training_args"]["save_strategy"] = interval_strategy
        setups["training_args"]["save_steps"] = monitoring_steps

        # Evaluation config
        if setups["training_args"]["do_eval"]:
            setups["training_args"]["per_device_eval_batch_size"] = batch_size
            setups["training_args"]["eval_strategy"] = interval_strategy
            setups["training_args"]["eval_steps"] = monitoring_steps

    return setups


def create_lora_config(
        model_id,
        rank,
        linear_modules,
        alpha_to_rank_ratio=2.0,
        dropout=0.0,
        is_all_linear=False
    ):
    """
    Create a LoRA (Low-Rank Adaptation) configuration for the specified model, 
    adjusting parameters for different model architectures.

    Parameters
    ----------
    model_id : str
        Identifier of the model for which the LoRA configuration is created.
    rank : int
        Rank of the LoRA matrices, determining the compression factor.
    alpha_to_rank_ratio : float, optional
        Multiplier for determining the LoRA alpha parameter (default is 2.0).
    dropout : float, optional
        Dropout rate for LoRA layers (default is 0.0).
    is_all_linear : bool, optional
        If True, applies LoRA to all linear layers; otherwise, applies to selected layers (default is False).

    Returns
    -------
    LoraConfig
        The configuration object for LoRA with specified settings.
    """
    lora_config = None
    
    kwargs = {
        "r": rank,
        "lora_alpha": round(rank * alpha_to_rank_ratio),
        "lora_dropout": dropout,
        "target_modules": 'all-linear' if is_all_linear else linear_modules,
        "task_type": 'CAUSAL_LM'
    }

    if 'phi-3' in model_id.lower():
        kwargs['init_lora_weights'] = 'gaussian'

    lora_config = LoraConfig(**kwargs)

    return lora_config


def configure_model_and_processor(
        model_id=None,
        use_bnb=True,
        use_flash_attention=False
    ):
    """
    Configure and load the model and processor for a specified multimodal language model.

    Parameters
    ----------
    model_id : str, optional
        Identifier of the model to be loaded (default is None).
    use_bnb : bool, optional
        Whether to apply bnb quantization for model compression (default is False).
    use_flash_attention : bool, optional
        Whether to enable Flash Attention for improved memory efficiency (default is False).

    Returns
    -------
    model : PreTrainedModel
        The loaded model configured for generation or causal language modeling.
    processor : PreTrainedProcessor
        The processor for handling input data for the specified model.
    """
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, # if use_flash_attention else torch.float16,
        )
        if use_bnb
        else None
    )

    if 'paligemma' in model_id.lower():
        processor = PaliGemmaProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={'':0}
        )
    elif 'llama-3' in model_id.lower():
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained( 
            model_id,
            torch_dtype=torch.bfloat16, # if use_flash_attention else torch.float32,
            trust_remote_code=True,
            quantization_config=bnb_config
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained( 
            model_id,
            torch_dtype=torch.bfloat16, # if use_flash_attention else torch.float32,
            trust_remote_code=True,
            _attn_implementation='flash_attention_2' if use_flash_attention else 'eager',
            quantization_config=bnb_config
        )

    return model, processor
