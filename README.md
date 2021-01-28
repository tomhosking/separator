# Separator
## Factorising Meaning and Form for Intent-preserving Paraphrasing - Tom Hosking & Mirella Lapata

## Installing

First, install [TorchSeq](https://github.com/tomhosking/torchseq/tree/separator-v1) and download the models/data:

```
python -m pip install -r requirements.txt
python -m nltk.downloader punkt
```

## Running the demo

Here's a snippet to run Separator on a new dataset:

```
import json
from torchseq.agents.para_agent import ParaphraseAgent
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config
import torch

path_to_model = '../models/separator-wa/'

examples = [
    {'input': 'How heavy is a moose in kilograms?', 'exemplar': 'How many pounds is a surgeons income?'},
    {'input': 'How heavy is a moose?', 'exemplar': 'What is the income of a surgeon?'}
]

with open(path_to_model + "/config.json") as f:
    cfg_dict = json.load(f)
cfg_dict["env"]["data_path"] = "../data/"
cfg_dict["dataset"] = "json"
cfg_dict["json_dataset"] = {
    "path": None,
    "field_map": [
        {"type": "copy", "from": "input", "to": "s2"},
        {"type": "copy", "from": "exemplar", "to": "template"},
        {"type": "copy", "from": "input", "to": "s1"},
    ],
}

config = Config(cfg_dict)
    
data_loader = JsonDataLoader(config, test_samples=examples)

checkpoint_path = path_to_model + "/model/checkpoint.pt"

instance = ParaphraseAgent(config=config, run_id=None, output_path="./runs/parademo/", silent=True, verbose=False)

instance.load_checkpoint(checkpoint_path)
instance.model.eval()
    

loss, metrics, (pred_output, gold_output, gold_input), memory_values_to_return = instance.inference(data_loader.test_loader)

print(pred_output)
```
> ['how many kilograms is a moose?', 'what is the weight of a moose?']



## Training a model

`torchseq --train --config ./configs/model_config.json`

## Citation

TBC