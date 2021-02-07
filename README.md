# Separator
## Factorising Meaning and Form for Intent-preserving Paraphrasing - Tom Hosking & Mirella Lapata

## Installing

First, install [TorchSeq](https://github.com/tomhosking/torchseq/tree/separator-v1) and download the models/data:

```
python -m pip install -r requirements.txt
```

## Example

Here's a snippet to run Separator with oracle exemplars:

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

There are more examples in `examples/`.

## Training a model

You can train a predefined model using:

`torchseq --train --config ./configs/separator-wa.json`

To use a different dataset, have a look at the patches, eg `configs/patches/qqp.json`, then run:

`torchseq --train --config ./configs/separator-wa.json --patch ./config/patches/qqp.json`


## Dataset generation commands

Build the training triples:

```
python3 ./scripts/generate_3way_wikianswers.py  --use_diff_templ_for_sem --rate 1.0 --sample_size 5 --extended_stopwords  --real_exemplars --template_dropout 0.3 --resample --dataset qqp-clusters
python3 ./scripts/generate_3way_wikianswers.py  --use_diff_templ_for_sem --rate 1.0 --sample_size 5 --extended_stopwords  --real_exemplars --template_dropout 0.3 --resample --dataset wa-triples
```

Train the code prediction MLP:

```
python3 ./scripts/train_vq_code_predictor.py --codebook_size 256 --train --dataset wikianswers --model_path ./models/separator-wa/
python3 ./scripts/train_vq_code_predictor.py --codebook_size 256 --train --dataset qqp --model_path ./models/separator-qqp/
```

## Citation

TBC