<div align="center">
  <h1> A Comparative Evaluation of Transformer-Based Vision Encoder-Decoder Models for Brazilian Portuguese Image Captioning </h1>
  <!--- ## By Computational Intelligence and Information Systems Laboratory (LAICSI-IFES) --->
  <p>Gabriel Bromonschenkel, Hilário Oliveira, Thiago M. Paixão</p>
</div>

<div align="center">
  <h1>SIBGRAPI 2024</h1>
 <img src='https://github.com/gabrielmotablima/ppcomp-image-captioning/assets/31813682/53c1301b-fd6e-41c0-be07-18d83d5a7b68' width='800'>
</div>

<br>

**Open-stuff available in**
- :floppy_disk: [Flickr30K Portuguese dataset (translated with Google Translator API)](https://huggingface.co/datasets/laicsiifes/flickr30k-pt-br)
- :1st_place_medal: [Swin-DistilBERTimbau (1st place model in Flickr30K Portuguese)](https://huggingface.co/laicsiifes/swin-distilbertimbau)
- :2nd_place_medal: [Swin-GPorTuguese-2 (2nd place model in Flickr30K Portuguese)](https://huggingface.co/laicsiifes/swin-gportuguese-2)

**or access our** [public available collection](https://huggingface.co/collections/laicsiifes/vision-encoder-decoder-brazilian-portuguese-image-captioning-66d6280c9e7dbd3be32d2770).
 
### :wrench: To set up the envionment, use:
```bash
$ chmod +x setup.sh
$ ./setup.sh
```

### :gear: To run the complete train and evaluate, use:
```bash
$ python train.py
```

### :gear: To run only the evaluation, use:
```bash
$ python eval.py
```

### :wrench: Don't forget of setting up the training/model attributes in ```config.yml```. An example:
```yaml
config:
  encoder: "deit-base-224"
  decoder: "gpt2-small"
  dataset: "pracegover_63k"
  max_length: 25
  batch_size: 16
  evaluate_from_model: False
  turn_off_computer: False

generate_args:
  num_beams: 1
  no_repeat_ngram_size: 0
  early_stopping: False

training_args:
  predict_with_generate: True
  num_train_epochs: 1
  eval_steps: 200
  logging_steps: 200
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  learning_rate: 5.6e-5
  weight_decay: 0.01
  save_total_limit: 1
  logging_strategy: "epoch"
  evaluation_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: True
  metric_for_best_model: "rougeL"
  greater_is_better: True
  push_to_hub: False
  fp16: True

callbacks:
  early_stopping:
    patience: 1
    threshold: 0.0

encoder: # available options to select in config.encoder at the top of this document
  vit-base-224: "google/vit-base-patch16-224"
  vit-base-224-21k: "google/vit-base-patch16-224-in21k"
  vit-base-384: "google/vit-base-patch16-384"
  vit-large-384: "google/vit-large-patch16-384"
  vit-huge-224-21k: "google/vit-huge-patch14-224-in21k"
  swin-base-224: "microsoft/swin-base-patch4-window7-224"
  swin-base-224-22k: "microsoft/swin-base-patch4-window7-224-in22k"
  swin-base-384: "microsoft/swin-base-patch4-window12-384"
  swin-large-384-22k: "microsoft/swin-large-patch4-window12-384-in22k"
  beit-base-224: "microsoft/beit-base-patch16-224"
  beit-base-224-22k: "microsoft/beit-base-patch16-224-pt22k"
  beit-large-224-22k: "microsoft/beit-large-patch16-224-pt22k-ft22k"
  beit-large-512: "microsoft/beit-large-patch16-512"
  beit-large-640: "microsoft/beit-large-finetuned-ade-640-640"
  deit-base-224: "facebook/deit-base-patch16-224"
  deit-base-distil-224: "facebook/deit-base-distilled-patch16-224"
  deit-base-384: "facebook/deit-base-patch16-384"
  deit-base-distil-384: "facebook/deit-base-distilled-patch16-384"

decoder: # available options to select in config.decoder at the top of this document
  bert-base: "neuralmind/bert-base-portuguese-cased"
  bert-large: "neuralmind/bert-large-portuguese-cased"
  roberta-small: "josu/roberta-pt-br"
  distilbert-base: "adalbertojunior/distilbert-portuguese-cased"
  gpt2-small: "pierreguillou/gpt2-small-portuguese"
  bart-base: "adalbertojunior/bart-base-portuguese"
```

### 🗃️ Directory structure:
```
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── pracegover_63k     <- Dataset #PraCegoVer 63k
│       ├── test.hf        <- Data for testing split.
│       ├── train.hf       <- Data for training split.
│       └── validation.hf  <- Data for validation split.
│
├── docs               <- A default HTML for docs.
│
├── models             <- The models and its artifacts will be saved here.
│
├── requirements.txt   <- The requirements file for reproducing the training and evaluation pipelines.
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── utils          <- Modularization for configuration, splits processing and evaluation metrics.
    │   ├── config.py
    │   ├── data_processing.py
    │   └── metrics.py
    │
    ├── eval.py
    └── train.py
```

### 📋 BibTex entry and citation info

```bibtex
@inproceedings{bromonschenkel2024comparative,
                title = "A Comparative Evaluation of Transformer-Based Vision 
                         Encoder-Decoder Models for Brazilian Portuguese Image Captioning",
               author = "Bromonschenkel, Gabriel and Oliveira, Hil{\'a}rio and 
                         Paix{\~a}o, Thiago M.",
            booktitle = "Proceedings...",
         organization = "Conference on Graphics, Patterns and Images, 37. (SIBGRAPI)",
                 year = "2024"
}
```
