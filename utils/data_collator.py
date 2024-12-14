import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from typing import List, Dict

class UnimolDataCollator:
    def __init__(self, max_size=512, padding_value=0, device=None):
        self.max_size = max_size
        self.padding_value = padding_value
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        output = {}
        max_seq_len = self.max_size

        for key in batch[0].keys():
            if np.isscalar(batch[0][key]):
                output[key] = torch.tensor([item[key] for item in batch], dtype=torch.float32)
                continue

            shape = torch.tensor([item[key].shape for item in batch]).max(dim=0).values
            shape = torch.minimum(shape, torch.full_like(shape, self.max_size))
            if key == "input_ids":
                max_seq_len = shape[0]
                seq_lens = torch.tensor([item[key].shape[0] for item in batch])
        
            paddeds = []
            for item in batch:
                data = item[key]
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                if data.dtype == torch.float64:
                    data = data.to(torch.float32)

                data_shape = torch.tensor(data.shape)
                padded_sequence = []
                for s in shape - data_shape:
                    padded_sequence.append(0)
                    padded_sequence.append(s)
                
                padded_data = F.pad(data, padded_sequence, value=self.padding_value)
                paddeds.append(padded_data)
            
            output[key] = torch.stack(paddeds)

        bsz = output["input_ids"].shape[0]
        attention_mask = torch.zeros((bsz, max_seq_len, max_seq_len), dtype=torch.float32)
        for i in range(bsz):
            attention_mask[i, :seq_lens[i], :seq_lens[i]] = 1.
        output["attention_mask"] = attention_mask

        for key, value in output.items():
            output[key] = value.to(self.device)
        return output
