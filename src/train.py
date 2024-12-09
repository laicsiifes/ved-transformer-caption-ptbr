"""
This module `train.py` is specifically designed for orchestrating the training process of an image
captioning model. It provides a high-level function that configures training parameters, sets up callbacks,
and initiates the training sequence using provided configurations and arguments.

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
train_model(config, training_args, generate_args, callbacks)
    Sets up and executes the training process for an image captioning model, handling configurations,
    training arguments, and execution callbacks.
"""
import os
import yaml
import time
import wandb

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from utils.config import config_vars, configure_model_and_tokenizer
from utils.data_processing import (
    load_datasets,
    preprocess,
    collate_fn,
    get_data_loader,
    transform_datasets
)
from utils.metrics import compute_metrics, generate_results
from torch.utils.data import DataLoader
from pprint import pprint
from dotenv import load_dotenv


def train_model(config, training_args, generate_args, callbacks):
    """
    Trains an image captioning model using the specified configurations, training arguments, 
    and callbacks. The function sets up the model, tokenizer, and image processor, loads datasets,
    and prepares the training environment. Training is managed by a Seq2SeqTrainer and metrics are computed
    during training. Additionally, the function handles evaluation and test data loading, and logs training
    progress to Weights & Biases (wandb).

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model, device, data directory, batch size, and maximum token length settings.
    training_args : dict
        Training arguments for the Seq2SeqTrainer including output directory and potentially other Hugging Face trainer settings.
    generate_args : dict
        Arguments used for generating captions, such as maximum length and number of beams for beam search.
    callbacks : dict
        Dictionary containing callback configurations such as early stopping conditions.

    Returns
    -------
    None
        This function does not return any value but trains the model according to the specified configurations and arguments,
        and generates results on the test dataset.
    """
    model, tokenizer, image_processor = configure_model_and_tokenizer(
        encoder_model=config["encoder_model"],
        decoder_model=config["decoder_model"],
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

    train_ds, valid_ds, test_ds = load_datasets(config["data_dir"])

    print('\nDataset')
    print(f'\tTrain: {len(train_ds)}')
    print(f'\tVal: {len(valid_ds)}')
    print(f'\tTest: {len(test_ds)}\n')

    train_dataset, valid_dataset, test_dataset = transform_datasets(
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        preprocess_fn=preprocess_fn
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=image_processor,
        args=Seq2SeqTrainingArguments(**training_args),
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=callbacks["early_stopping"]["patience"],
                early_stopping_threshold=callbacks["early_stopping"]["threshold"]
            )
        ],
    )

    trainer.get_train_dataloader = lambda: DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=config["batch_size"]
    )

    trainer.get_eval_dataloader = lambda eval_dataset: get_data_loader(
        dataset=valid_dataset,
        collate_fn=collate_fn,
        batch_size=config["batch_size"],
        eval_dataset=eval_dataset
    )

    trainer.get_test_dataloader = lambda eval_dataset: get_data_loader(
        dataset=test_dataset,
        collate_fn=collate_fn,
        batch_size=config["batch_size"],
        eval_dataset=eval_dataset
    )

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    with wandb.init(project=os.getenv("WANDB_PROJECT_NAME")) as run:
        run.name = f'{config["encoder_decoder_model"]}_ft'
        trainer.train()

    generate_results(
        dataset=test_dataset,
        raw_dataset=test_ds,
        model=trainer.model,
        config=config,
        collate_fn=collate_fn,
        tokenizer=tokenizer,
        image_processor=image_processor,
        generate_args=generate_args,
        trainer=trainer
    )


if __name__ == "__main__":
    """
    Main function to run the training process based on configurations specified in a YAML file.
    """
    load_dotenv(dotenv_path="../.env")

    with open("../config.yml", "r") as file:
        setups = config_vars(yaml.safe_load(file))

    print("\nConfiguration:", end="\t")
    pprint(setups)

    if not os.path.exists(setups["config"]['results_dir']):
        os.makedirs(setups["config"]['results_dir'])

    train_model(
        config=setups["config"],
        training_args=setups["training_args"],
        generate_args=setups["generate_args"],
        callbacks=setups["callbacks"]
    )

    if setups["config"]["turn_off_computer"]:
        print('\nTurning off computer ...')
        time.sleep(2 * 60)
        os.system('shutdown -h now')
