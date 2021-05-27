# Separator
## Factorising Meaning and Form for Intent-Preserving Paraphrasing - Tom Hosking & Mirella Lapata



![Model diagram](/images/pipeline.png?raw=true)


## Installing

First, install [TorchSeq](https://github.com/tomhosking/torchseq/tree/separator-v1) and download the models/data:

```
python -m pip install -r requirements.txt
```

[Download our split of Paralex](http://tomho.sk/models/separator/data_paralex.zip)
[Download our split of QQP](http://tomho.sk/models/separator/data_qqp.zip)
[Download a pretrained checkpoint for Paralex](http://tomho.sk/models/separator/separator_paralex.zip)
[Download a pretrained checkpoint for QQP](http://tomho.sk/models/separator/separator_qqp.zip)

## Replicating our results

This is pretty straightforward, just run:

`torchseq --load ./models/separator-qqp-v1.2 --test`
or `torchseq --load ./models/separator-qqp-v1.2 --test`

Replace `--test` with `-validate` to get results on the dev set.

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

`torchseq --train --config ./configs/separator-wa.json`

## Training a model on a custom dataset

To use a different dataset, you will need to generate a total of 4 datasets. These should be folders in `./data`, containing `{train,dev,test}.jsonl` files.

*A cluster dataset, that is a list of the paraphrase clusters*

```
[
    ["",""],
    ["",""],
    ...
]
```

*A flattened dataset, that is just a list of all the paraphrases*

```
[
    "Paraphrase 1",
    "Paraphrase 2",
    ...
]
```

*The training triples*

Generate this using the following command:

```
python3 ./scripts/generate_3way_wikianswers.py  --use_diff_templ_for_sem --rate 1.0 --sample_size 5 --extended_stopwords  --real_exemplars --template_dropout 0.3 --resample --dataset qqp-clusters
```

Replace qqp-clusters with the path to your dataset in "cluster" format.


*A dataset to use for evaluation*

Each paraphrase cluster should be split so that there is an input sentence, and reference paraphrases of that input

```
```

*Train the model*

Have a look at the patches, eg `configs/patches/qqp.json`, and create a patch that points to your dataset, then run:

`torchseq --train --config ./configs/separator-wa.json --patch ./config/patches/qqp.json`


## Citation

TBC