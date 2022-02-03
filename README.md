# Factorising Meaning and Form for Intent-Preserving Paraphrasing

This repo contains the code for the paper "Factorising Meaning and Form for Intent-Preserving Paraphrasing", Tom Hosking & Mirella Lapata (ACL 2021).

![Model diagram](/images/pipeline.png?raw=true)


## Installing

First, install [TorchSeq](https://github.com/tomhosking/torchseq/releases/tag/separator-v1.2) and other dependencies using the following command:

```
python -m pip install -r requirements.txt
```

<a href="https://tomho.sk/models/separator/data_paralex.zip" download>Download our split of Paralex</a>

<a href="https://tomho.sk/models/separator/data_qqp.zip" download>Download our split of QQP</a>

<a href="https://tomho.sk/models/separator/separator_paralex.zip" download>Download a pretrained checkpoint for Paralex</a>

<a href="https://tomho.sk/models/separator/separator_qqp.zip" download>Download a pretrained checkpoint for QQP</a>

Model zip files should be unzipped into `./models`, eg `./models/separator-qqp-v1.2`. Data zip files should be unzipped into `./data/`.

Note: Paralex was originally scraped from WikiAnswers, so many of the Paralex models and datasets are labelled as 'wa' or WikiAnswers.

## Replicating our results

This is pretty straightforward, just run:

`torchseq --load ./models/separator-qqp-v1.2 --test`
or `torchseq --load ./models/separator-wa-v1.2 --test`

Replace `--test` with `--validate` to get results on the dev set.

Once it has finished, check the output folder in `./runs` - the `sepae_codepred_bleu/sepae_codepred_selfbleu` scores in `metrics.json` will allow you to calculate the iBLEU scores.

## Run inference over a custom dataset

Here's a snippet to run Separator on a custom dataset:

```
import json
from torchseq.agents.para_agent import ParaphraseAgent
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config

import torch

# Which checkpoint should we load?
path_to_model = '../models/separator-wa-v1.2/'

# Define the data
examples = [
    {'input': 'What is the income for a soccer player?'},
    {'input': 'What do soccer players earn?'}
]


# Change the config to use the custom dataset
with open(path_to_model + "/config.json") as f:
    cfg_dict = json.load(f)
cfg_dict["env"]["data_path"] = "../data/"
cfg_dict["dataset"] = "json"
cfg_dict["json_dataset"] = {
    "path": None,
    "field_map": [
        {"type": "copy", "from": "input", "to": "s2"},
        {"type": "copy", "from": "input", "to": "s1"},
    ],
}

# Enable the code predictor
cfg_dict["bottleneck"]["code_predictor"]["infer_codes"] = True

# Create the dataset and model
config = Config(cfg_dict)
data_loader = JsonDataLoader(config, test_samples=examples)
checkpoint_path = path_to_model + "/model/checkpoint.pt"
instance = ParaphraseAgent(config=config, run_id=None, output_path=None, silent=True, verbose=False, training_mode=False)

# Load the checkpoint
instance.load_checkpoint(checkpoint_path)
instance.model.eval()
    
# Finally, run inference
_, _, (pred_output, _, _), _ = instance.inference(data_loader.test_loader)

print(pred_output)
```
> ['what is the salary for a soccer player?', 'what do soccer players earn?']


There are more examples in `examples/`.

## Training a model on QQP/Paralex

You can train a predefined model using:

`torchseq --train --config ./configs/separator_wa.json`

or 

`torchseq --train --config ./configs/separator_qqp.json`

## Training a model on a custom dataset

To use a different dataset, you will need to generate a total of 4 datasets. These should be folders in `./data`, containing `{train,dev,test}.jsonl` files.

#### A cluster dataset, that is a list of the paraphrase clusters

```
{"qs": ["What are some good science documentaries?", "What is a good documentary on science?", "What is the best science documentary you have ever watched?", "Can you recommend some good documentaries in science?", "What the best science documentaries?"]}
{"qs": ["What do we use water for?", "Why do we, as human beings, use water for?"]}
...
```

#### A flattened dataset, that is just a list of all the paraphrases

The sentences must be in the same order as in the cluster dataset!

```
{"q": "Can you recommend some good documentaries in science?"}
{"q": "What the best science documentaries?"}
{"q": "What do we use water for?"}
...
```

#### The training triples

Generate this using the following command:

```
python3 ./scripts/generate_3way_wikianswers.py  --use_diff_templ_for_sem --rate 1.0 --sample_size 5 --extended_stopwords  --real_exemplars --template_dropout 0.3 --resample --dataset qqp-clusters
```

Replace qqp-clusters with the path to your dataset in "cluster" format.


#### A dataset to use for evaluation

For each cluster, select a single sentence to use as the input (assigned to `sem_input`) and add all the other references to `paras`. `tgt` and `syn_input` should be set to one of references.

```
{"tgt": "What are some good science documentaries?", "syn_input": "What are some good science documentaries?", "sem_input": "Can you recommend some good documentaries in science?", "paras": ["What are some good science documentaries?", "What the best science documentaries?", "What is the best science documentary you have ever watched?", "What is a good documentary on science?"]}
{"tgt": "What do we use water for?", "syn_input": "What do we use water for?", "sem_input": "Why do we, as human beings, use water for?", "paras": ["What do we use water for?"]}
...
```

#### Train the model

Have a look at the patches, eg `configs/patches/qqp.json`, and create a patch that points to your dataset, then run:

`torchseq --train --config ./configs/separator-wa.json --patch ./config/patches/qqp.json`


## Citation

```
@inproceedings{hosking-lapata-2021-factorising,
    title = "Factorising Meaning and Form for Intent-Preserving Paraphrasing",
    author = "Hosking, Tom  and
      Lapata, Mirella",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.112",
    pages = "1405--1418",
}
```
