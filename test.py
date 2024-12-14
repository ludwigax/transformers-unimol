from model.modeling_unimol import UnimolForSequenceClassification
from model.configuration_unimol import UnimolConfig
from model.tokenization_unimol import UnimolTokenizer
import yaml

from utils.data_collator import UnimolDataCollator

import os
import torch

# # Initialize the model
# config = UnimolConfig()
# config.save_pretrained('pretrained/hg_unimol')

# model = UnimolForSequenceClassification(config)


# # Moving pretrained model
# state_dict = torch.load("moved_weight.pt")
# model.load_state_dict(state_dict, strict=False)
# model.save_pretrained('pretrained/hg_unimol')


# Load from the pretrained model
model = UnimolForSequenceClassification.from_pretrained('pretrained/hg_unimol')


# Suppose 'vocab.txt' is a file where each line is a token, ordered from id 0 to n
# with special tokens included at the beginning.

# Initialize the tokenizer
# tokenizer = UnimolTokenizer(vocab_file='pretrained/dptech_unimol/mol.dict.txt')

# Save the tokenizer for future use
# tokenizer.save_pretrained('pretrained/hg_unimol')

tokenizer = UnimolTokenizer.from_pretrained('pretrained/hg_unimol')

# Encode a SMILES string
smiles_list = [
    'CCO', 'c1ccccc1', 'CC(=O)O', 'CCC', 'CCN'
]
batch = []
for smiles in smiles_list:
    encoded = tokenizer.encode(smiles, add_special_tokens=True, use_3d=True)
    batch.append(encoded)

collate_fn = UnimolDataCollator(device="cuda")

encoded = collate_fn(batch)

# Test the inference of the model
model.to("cuda")
logits = model(**encoded)
print(logits)

# for key, value in model.state_dict().items():
#     print(f"{key}:\t{value.shape}\t{value.dtype}")

# caches = torch.load("pretrained/dptech_unimol/mol_pre_all_h_220816.pt")
# for key, value in caches['model'].items():
#     print(f"{key}:\t{value.shape}\t{value.dtype}")