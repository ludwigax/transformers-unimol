from model.modeling_unimol import UnimolForSequenceClassification
from model.tokenization_unimol import UnimolTokenizer
from utils.trainer import Trainer, CustomTest
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

tokenizer = UnimolTokenizer.from_pretrained('pretrained/hg_unimol')
model = UnimolForSequenceClassification.from_pretrained('pretrained/hg_unimol')
model = UnimolForSequenceClassification(model.config)

params = {
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'warmup_ratio': 0.1,
    'patience': 10,
    'batch_size': 16,
    'split_seed': 42,
    'split_ratio': 0.8,
    'do_eval': True,
    'device': 'cuda'
}

df = pd.read_excel("qm9conj_w.xlsx")
smiles_list = df["SMILES"].tolist()
labels = df["W value"].tolist()

dataset = []
for smiles, label in tqdm(zip(smiles_list, labels), desc="Encoding SMILES"):
    try:
        encoded = tokenizer.encode(smiles, add_special_tokens=True, use_3d=True)
        encoded['labels'] = label
        dataset.append(encoded)
    except:
        continue

print("Length of dataset:", len(dataset))

trainer = Trainer(model, dataset, params)
trainer.train()

# Log the results
test = CustomTest(trainer)
test.run()