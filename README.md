# Transferable Image Adversarial Attacks on Vision Language Models

Hi! Welcome to our repository. In this work, we train image adversarial attacks and study their transferability. 

## Models

| Model                                               | Size  | Parameter Count |
|------------------------------------------------------|-------|------------------|
| ViLT: Vision-and-Language Transformer                |    Small   | 118M             |
| GIT: Generative Image-to-Text Transformer            | Small | 177M             |
| Pix2Struct                                           |Small| 282M             |
| Bridgetower (Base)                                   |Small| 366M             |
| Bridgetower (Large)                                  |Large| 1B               |
| BLIP-2 (OPT)                                         | Large | 2.7B             |
| BLIP-2 (Flan-T5-XL)                                  |Large| 3B               |
| LLaVA: Large Language and Vision Assistant           |Large| 7B               |


## Experiments

`Small Model Training` directory has the training and analysis code for Small VLMs (i.e.; $1B$) parameters. `Big Model Training` directory has the training and analysis code for large VLMs (i.e.; $> 1B$) parameters. 

Run the notebook `Small Model Training/adversary training.ipynb` to training adversaries, and `Small Model Training/adversary transfer.ipynb` to analyse transferability.

### Paper Link
Add the link.
