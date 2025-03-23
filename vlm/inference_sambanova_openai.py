import yaml
import os
import openai
import base64
import json
import time
import sys

from dotenv import load_dotenv
from utils.config import config_vars
from datasets import load_dataset
from tqdm import tqdm
from io import BytesIO


def encode_image(image_path):
    with open(file=image_path, mode="rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":

    load_dotenv(dotenv_path="../.env")

    with open(file="../config.yml", mode="r") as file:
        setups = config_vars(yaml.safe_load(file))

    outputs_dir = "../data/captions/"
    images_dir = "../images/"

    config = setups["config"]

    SAMBA_NOVA_APIKEY = os.getenv("SAMBA_NOVA_API_KEY_1")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # model_tuple = ("llamavision_32_11b", "Llama-3.2-11B-Vision-Instruct")
    # model_tuple = ("llamavision_32_90b", "Llama-3.2-90B-Vision-Instruct")
    # model_tuple = ("gpt4o_mini", "gpt-4o-mini")
    model_tuple = ("gpt4o", "gpt-4o")

    outputs_dir = os.path.join(outputs_dir, model_tuple[0])

    os.makedirs(outputs_dir, exist_ok=True)

    max_length = config["max_length"]

    prompt = f"Escreva uma descrição em português do Brasil para a imagem com no máximo {max_length} palavras."

    dataset_hub = config["hf_test_set"]

    print(f"\nLoading {dataset_hub}")

    test_dataset = load_dataset(dataset_hub, split="test")

    print(f"\n\tTotal of Examples: {len(test_dataset)}")

    if "gpt" in model_tuple[0]:
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
        )
        time_sleep = 0
    else:
        time_sleep = 3
        client = openai.OpenAI(
            api_key=SAMBA_NOVA_APIKEY,
            base_url="https://api.sambanova.ai/v1",
        )

    im_file = BytesIO()

    print(f'\nGenerating Captions Using {model_tuple[0]}\n')

    outputs_file_path = os.path.join(outputs_dir, f'{model_tuple[0]}.json')

    dict_images_processed = {}

    if os.path.exists(outputs_file_path):
        with open(file=outputs_file_path, mode='r', encoding='utf-8') as json_file:
            output_data = json.load(json_file)
            for example in output_data:
                dict_images_processed[example['img_id']] = example

    list_generated_captions = []

    temp_image_path = f'{images_dir}/image.jpeg'

    with tqdm(total=len(test_dataset), colour='green', file=sys.stdout,
              desc='\tGenerating Captions') as pbar:

        for example in test_dataset:

            # print(example)

            image = example['image']
            img_id = example['img_id']
            file_name = example['filename']
            reference_captions = example['caption']

            if img_id in dict_images_processed:
                data = dict_images_processed[img_id]
                list_generated_captions.append(data)
                pbar.update(1)
                continue

            image.save(temp_image_path, format='JPEG')

            base64_image = encode_image(temp_image_path)

            messages = [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': prompt,
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/jpeg;base64,{base64_image}'
                            }
                        }
                    ]
                }
            ]

            response = client.chat.completions.create(
                model=model_tuple[1],
                messages=messages,
                max_tokens=max_length,
                temperature=0.1,
            )

            generated_caption = response.choices[0].message.content

            list_generated_captions.append(
                {
                    'img_id': img_id,
                    'file_name': file_name,
                    'reference_captions': reference_captions,
                    'generated_caption': generated_caption
                }
            )

            with open(file=outputs_file_path, mode='w', encoding='utf-8') as json_file:
                json.dump(list_generated_captions, json_file, indent=4)

            time.sleep(time_sleep)

            pbar.update(1)

    os.remove(temp_image_path)
