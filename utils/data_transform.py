from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Union
import numpy as np
import torch

class UnimolTransform:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, data: List[Union[Dict, float]]):
        if isinstance(data[0], Dict):
            data = [item['labels'] for item in data]
            data = np.array(data).reshape(-1, 1)
        elif isinstance(data[0], np.ndarray | torch.Tensor):
            bsz = len(data)
            data = np.ndarray(data).squeeze().reshape(bsz, -1)
        else:
            data = np.array(data).reshape(-1, 1)
        self.scaler.fit(data)
        self.is_fitted = True

    def transform(self, data: List[Union[Dict, float]]) -> List[Union[Dict, float]]:
        if not self.is_fitted:
            raise RuntimeError("fit() must be called before transform().")
        
        if isinstance(data[0], Dict):
            return [{
                **item,
                'labels': self.scaler.transform([[item['labels']]])[0][0],
                # 'original_labels': item['labels']
            } for item in data]
        else:
            data = np.array(data).reshape(-1, 1)
            return self.scaler.transform(data)

    def inverse_transform(self, data: List[Union[Dict, float]]) -> List[Union[Dict, float]]:
        if not self.is_fitted:
            raise RuntimeError("fit() must be called before inverse_transform().")
        
        if isinstance(data[0], Dict):
            return [{
                **item,
                'labels': self.scaler.inverse_transform([[item['labels']]])[0][0]
            } for item in data]
        else:
            data = np.array(data).reshape(-1, 1)
            return self.scaler.inverse_transform(data)

    def fit_transform(self, data: List[Union[Dict, float]]) -> List[Union[Dict, float]]:
        self.fit(data)
        return self.transform(data)

