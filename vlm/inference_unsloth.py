import yaml
import os
import json
import sys

from dotenv import load_dotenv
from utils.config import config_vars
from datasets import load_dataset
from unsloth import FastVisionModel
from  tqdm import tqdm


if __name__ == "__main__":

    load_dotenv(dotenv_path="../.env")

    outputs_dir = "../data/captions/"
    images_dir = "../images/"

    model_tuple = ("llamavision_32_11b", "unsloth/Llama-3.2-11B-Vision-Instruct")

    with open(file="../config.yml", mode="r") as file:
        setups = config_vars(yaml.safe_load(file))

    config = setups["config"]

    outputs_dir = os.path.join(outputs_dir, model_tuple[0])

    os.makedirs(outputs_dir, exist_ok=True)

    max_length = config["max_length"]

    prompt = f"Escreva uma descrição em português do Brasil para a imagem com no máximo {max_length} palavras."

    dataset_hub = config["hf_test_set"]

    print(f"\nLoading {dataset_hub}")

    test_dataset = load_dataset(dataset_hub, split="test")

    print(f"\n\tTotal of Examples: {len(test_dataset)}")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_tuple[1],
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    print(f'\nGenerating Captions Using {model_tuple[0]}\n')

    outputs_file_path = os.path.join(outputs_dir, f'{model_tuple[0]}.json')

    dict_images_processed = {}

    if os.path.exists(outputs_file_path):
        with open(file=outputs_file_path, mode='r', encoding='utf-8') as json_file:
            output_data = json.load(json_file)
            for example in output_data:
                dict_images_processed[example['img_id']] = example

    list_generated_captions = []

    temp_image_path = f"{images_dir}/image.jpeg"

    with tqdm(total=len(test_dataset), colour='green', file=sys.stdout,
              desc='\t\tGenerating Captions') as pbar:

        for example in test_dataset:

            image = example['image']
            img_id = example['img_id']
            file_name = example['filename']
            reference_captions = example['caption']

            if img_id in dict_images_processed:
                data = dict_images_processed[img_id]
                list_generated_captions.append(data)
                pbar.update(1)
                continue

            FastVisionModel.for_inference(model)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image"
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ],
                }
            ]

            input_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

            inputs = tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(config["device"])

            output = model.generate(
                **inputs,
                max_new_tokens=25,
                use_cache=True,
                temperature=0.1
            )

            generated_caption = tokenizer.decode(output[0])

            print(generated_caption)

            list_generated_captions.append(
                {
                    "img_id": img_id,
                    "file_name": file_name,
                    "reference_captions": reference_captions,
                    "generated_caption": generated_caption
                }
            )

            with open(file=outputs_file_path, mode='w', encoding='utf-8') as json_file:
                json.dump(list_generated_captions, json_file, indent=4)

            pbar.update(1)

            break
