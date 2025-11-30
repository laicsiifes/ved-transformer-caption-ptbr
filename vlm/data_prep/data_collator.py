"""
Data Collators for Training and Generation with Vision-Language Models
======================================================================

This module defines data collator classes to prepare input batches for training and evaluation 
with various multimodal vision-language models, including LLaMa 3.2 Vision, Phi-3 Vision, 
and PaliGemma. These classes support customized batching of images and text, ensuring compatibility 
with model-specific requirements.

Authors
-------
BSc, Gabriel Mota Bromonschenkel Lima
Email: gabriel.mota.b.lima@gmail.com

PhD, Hilário Tomaz Alves de Oliveira
Email: hilariotomaz@gmail.com

PhD, Thiago Meireles Paixão
Email: thiago.paixao@ifes.edu.br

Classes
-------
DataCollatorForTraining(processor, model_id, device, max_length)
    Selects and applies appropriate collator functions to prepare input data into batches 
    for training with the specified model.

DataCollatorForGeneration(processor, model_id, device, max_length)
    Selects and applies appropriate collator functions to prepare input data into batches 
    for evaluation or generation with the specified model.
"""

import torch
from dataclasses import dataclass
from transformers import ProcessorMixin


@dataclass
class DataCollatorForGeneration:
    """
    Data collator for preparing batches for generation with vision-language models.

    This class selects and applies the appropriate collator function based on the model ID to format 
    input data into batches compatible with the model's generation requirements. Supports models 
    like LLaMa 3.2 Vision, Phi-3 Vision, and PaliGemma.

    Parameters
    ----------
    processor : ProcessorMixin
        Processor instance to handle tokenization and data formatting.
    model_id : str
        Model identifier to determine which collator function to apply (e.g., "llama-3", "phi-3", "paligemma").
    device : str
        Device on which the tensors should be stored (e.g., "cpu" or "cuda").
    max_length : int
        Maximum length of the sequence for tokenized inputs.

    Methods
    -------
    __call__(examples)
        Selects and applies the appropriate collator function to prepare input data into batches.
    llama3_collator_for_generation(example)
        Prepares data batches for generation with the LLaMa 3.2 Vision model.
    phi3_collator_for_generation(example)
        Prepares data batches for generation with the Phi-3 Vision model.
    paligemma_collator_for_generation(example)
        Prepares data batches for generation with the PaliGemma model.
    """
    processor: ProcessorMixin
    model_id: str
    device: str
    max_length: int


    def __call__(self, examples):
        """
        Select and apply the appropriate collator function based on the specified model ID to prepare 
        input data into batches for generation.

        Parameters
        ----------
        examples : list of dict
            A list containing preprocessed image, question, and answer data for batching.

        Returns
        -------
        dict
            A batch dictionary with processed input data for the specified model.
        """
        batch = None

        mini_batches = []

        for example in examples:
            if "llama-3" in self.model_id.lower():
                tmp = self.llama3_collator_for_generation(example)
            elif "phi-3" in self.model_id.lower():
                tmp = self.phi3_collator_for_generation(example)
            else:
                tmp = self.paligemma_collator_for_generation(example)

            mini_batches.append(tmp)

        if len(examples)==1:
            batch = tmp
        else:
            batch = {}
            for key in tmp:
                batch[key] = torch.stack([mini_batch[key] for mini_batch in mini_batches])

        return batch


    def llama3_collator_for_generation(self, example):
        """
        Prepare and collate input data into batches for generation with the LLaMa 3.2 Vision model.

        Parameters
        ----------
        example : dict
            A dictionary containing preprocessed image, question, and answers.

        Returns
        -------
        dict
            A batch dictionary with processed input data for the model.
        """
        text_prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' \
                    f'<|image|>{example["question"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

        return self.processor(
            text=text_prompt,
            images=example["image"],
            return_tensors='pt'
        ).to(self.device)


    def phi3_collator_for_generation(self, example):
        """
        Prepare and collate input data into batches for generation with the Phi-3 Vision model.

        Parameters
        ----------
        example : dict
            A dictionary containing preprocessed image, question, and answer.

        Returns
        -------
        dict
            A batch dictionary with processed input data for the model.
        """
        return self.processor(
            text=f'<|user|>\n<|image_1|>\n{example["question"]}<|end|>\n<|assistant|>\n',
            images=[example["image"]],
            return_tensors='pt'
        ).to(self.device)


    def paligemma_collator_for_generation(self, example):
        """
        Prepare and collate input data into batches for generation with the PaliGemma model.

        Parameters
        ----------
        example : dict
            A dictionary containing preprocessed image, question, and answer.

        Returns
        -------
        dict
            A batch dictionary with processed input data for the model.
        """
        return self.processor(
            text=example["question"],
            images=example["image"],
            return_tensors='pt'
        ).to(self.device)
