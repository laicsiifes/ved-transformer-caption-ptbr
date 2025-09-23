"""
Metrics Evaluation Module for Image Captioning
==============================================

This module provides functions to calculate evaluation metrics for image captioning 
models, particularly for multimodal vision-language models. It includes helper 
functions for computing BERTScore, CLIPScore, ROUGE, BLEU, METEOR, and other 
captioning metrics, providing both individual and aggregated scores for 
comprehensive analysis.

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
clip_score(reference, candidate, kind, tokenizer, preprocess, model, device, w)
    Compute the CLIP-based similarity score between reference and candidate inputs.

ref_clip_score(image_score, text_scores)
    Calculate the harmonic mean of the image and text CLIP scores.

compute_clip_scores(predictions, labels, images)
    Compute CLIP-based similarity scores (CLIPScore and RefCLIPScore) for image-caption pairs.
"""

import gc
import open_clip
import torch
from tqdm import tqdm


def teste():
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




def compute_llm_as_a_judge(predictions, labels, scorer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = openai.OpenAI(
        api_key=OPENAI_API_KEY,
    )
    time_sleep = 0
    tokenizer = open_clip.get_tokenizer('hf-hub:hiaac-nlp/CAPIVARA')
    
    scores = {
        'clipscore': [],
        'ref_clipscore': []
    }
    
    with tqdm(total=len(predictions)) as pbar:
        for prediction, label, batch in zip(predictions, labels, dataset):
            pbar.set_description("Eval. LLM-as-a-Judge")
            txt_scores = [
                clip_score(
                    reference=reference,
                    candidate=prediction,
                    kind='txt-txt',
                    tokenizer=tokenizer,
                    preprocess=preprocess,
                    model=model
                ) for reference in label
            ]
            score = ref_clip_score(img_score, txt_scores)
            scores['clipscore'].append(img_score)
            scores['ref_clipscore'].append(score)
            pbar.update(1)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return scores

    result = {}

    # Select the computing metric funtion
    if "rouge" in scorer.name.lower():
        compute = compute_rouge_scores
    elif "meteor" in scorer.name.lower():
        compute = compute_meteor_scores
    elif "bleu" in scorer.name.lower():
        compute = compute_bleu_scores
    else:
        compute = None

    # If there is a compute function, score the predictions 
    if compute:
        with tqdm(total=len(predictions)) as pbar:
            pbar.set_description(f"Eval. {scorer.name.upper()}")

            # Create an empty dict with empty lists to add the by-example scores
            result = {
                k:[] for k in list(compute([''], [''], scorer).keys())
            }

            # Compute the score to each example
            for prediction, label in zip(predictions, labels):
                individual_result = compute([prediction], [label], scorer)

                # Append the individual scores to the result dict
                for key in individual_result:
                    result[key].append(individual_result[key])
                pbar.update(1)
    else:
        print("`scorer` parameter is not set correctly, returning empty metric dict.")
        
    return result


def compute_rouge_scores(predictions, labels, scorer):
    return scorer.compute(
            predictions=predictions,
            references=labels,
            use_stemmer=False,
            tokenizer=lambda x: x.split()
        )


def compute_meteor_scores(predictions, labels, scorer):
    return scorer.compute(
            predictions=predictions,
            references=labels
        )


def compute_bleu_scores(predictions, labels, scorer):
    results = {}
    """
        Bleu raised an exception `ZeroDivisionError` during the training of some models.
    """
    try:
        results = scorer.compute(
            predictions=predictions,
            references=labels,
            tokenizer=lambda x: x.split()
        )
        results["precisions"] = str(results["precisions"])
    except ZeroDivisionError:
        results = {
            "bleu": 0.0,
            "precisions": "[]",
            "brevity_penalty": 0.0,
            "length_ratio": 0.0,
            "translation_length": 0.0,
            "reference_length": 0.0
        }
    return results


def compute_cider_scores(predictions, labels, scorer):
    return {
        k: float(v) for k, v in scorer(predictions, labels)[0].items()
    } if scorer else {}


def compute_bert_scores(predictions, labels, bertscore):
    results = {}
    
    print("Eval. BERTScore")
    bertscore_result = bertscore.compute(
        predictions=[' '.join(prediction.split()[:200]) for prediction in predictions],
        references=[[' '.join(unit.split()[:200]) for unit in label] for label in labels],
        model_type="neuralmind/bert-base-portuguese-cased",
        num_layers=12
    )

    results["bertscore_precision"] = bertscore_result["precision"]
    results["bertscore_recall"] = bertscore_result["recall"]
    results["bertscore_f1"] = bertscore_result["f1"]
    results["bertscore_hashcode"] = bertscore_result["hashcode"]

    return results


def clip_score(
        reference,
        candidate,
        kind,
        tokenizer,
        preprocess,
        model,
        device='cuda',
        w=2.5
    ):
    """
    Compute the CLIP-based similarity score between reference and candidate inputs.

    Parameters
    ----------
    reference : PIL.Image.Image or str
        The reference input, which can be an image (for 'img-txt' kind) or a text string.
    candidate : str
        The candidate text to compare with the reference.
    kind : str
        Type of comparison, either 'img-txt' for image-to-text or 'txt-txt' for text-to-text.
    tokenizer : Callable
        The tokenizer function used to preprocess the text input.
    preprocess : Callable
        The preprocessing function for images, used if `kind` is 'img-txt'.
    model : CLIPModel
        The CLIP model used to encode images and text.
    device : str, optional
        The device ('cuda' or 'cpu') for processing inputs (default is 'cuda').
    w : float, optional
        Weight factor for scaling the similarity score (default is 2.5).

    Returns
    -------
    float
        The weighted similarity score between the reference and candidate.
    """
    candidate = tokenizer(candidate).to(device)

    if kind == 'img-txt':
        reference = reference.convert('RGB')
        reference = preprocess(reference).unsqueeze(0).to(device)
    else:
        reference = tokenizer(reference).to(device)

    with torch.no_grad():
        if kind == 'img-txt':
            reference_features = model.encode_image(reference)
            candidate_features = model.encode_text(candidate)
        else:
            reference_features = model.encode_text(reference)
            candidate_features = model.encode_text(candidate)

    reference_features /= reference_features.norm(dim=-1, keepdim=True)
    candidate_features /= candidate_features.norm(dim=-1, keepdim=True)

    similarity = torch.matmul(reference_features, candidate_features.T)
    return w * max(similarity.item(), 0)


def ref_clip_score(image_score, text_scores):
    """
    Calculate the harmonic mean of the image and text CLIP scores.

    Parameters
    ----------
    image_score : float
        The CLIP score for the reference image.
    text_scores : list of float
        A list of CLIP scores for the reference text(s).

    Returns
    -------
    float
        The harmonic mean of the image and the highest text score.
    """
    text_score = max(text_scores)
    return 2 * (image_score * text_score) / (image_score + text_score)


def compute_clip_scores(predictions, labels, dataset):
    """
    Compute CLIP-based similarity scores (CLIPScore and RefCLIPScore) for image-caption pairs.

    Parameters
    ----------
    predictions : list of str
        A list of predicted captions.
    labels : list of list of str
        A list of lists, where each sublist contains reference captions for each image.
    dataset : Dataset
        A list of images corresponding to each prediction.

    Returns
    -------
    dict
        A dictionary with two keys containing CLIPScore and RefCLIPScore similarity scores.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = open_clip.create_model_from_pretrained('hf-hub:hiaac-nlp/CAPIVARA')
    model.to(device)
    tokenizer = open_clip.get_tokenizer('hf-hub:hiaac-nlp/CAPIVARA')
    
    scores = {
        'clipscore': [],
        'ref_clipscore': []
    }
    
    with tqdm(total=len(predictions)) as pbar:
        for prediction, label, batch in zip(predictions, labels, dataset):
            pbar.set_description("Eval. CLIPScore")
            img_score = clip_score(
                reference=batch["image"],
                candidate=prediction,
                kind='img-txt',
                tokenizer=tokenizer,
                preprocess=preprocess,
                model=model
            )
            txt_scores = [
                clip_score(
                    reference=reference,
                    candidate=prediction,
                    kind='txt-txt',
                    tokenizer=tokenizer,
                    preprocess=preprocess,
                    model=model
                ) for reference in label
            ]
            score = ref_clip_score(img_score, txt_scores)
            scores['clipscore'].append(img_score)
            scores['ref_clipscore'].append(score)
            pbar.update(1)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return scores


def compute_clair_scores(predictions, labels, scorer)()