import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from transformers import get_scheduler
from data_collator import UnimolDataCollator
from data_transform import UnimolTransform
from tqdm import tqdm

from typing import Dict, List, Union

class Trainer:
    def __init__(self, model: nn.Module, dataset: Union[Dataset, List], params: Dict):
        self.model = model
        self.dataset = dataset
        self.params = params

        self.device = params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = params.get('num_epochs', 50)
        self.learning_rate = params.get('learning_rate', 1e-4)
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.patience = params.get('patience', 10)
        self.batch_size = params.get('batch_size', 16)
        self.split_seed = params.get('split_seed', 42)
        self.split_ratio = params.get('split_ratio', 0.8)
        self.do_eval = params.get('do_eval', False)

        self.model.train()
        self.model.to(self.device)

        self.transform = UnimolTransform()
        self.dataset = self.transform.fit_transform(self.dataset)

        self.collate_fn = UnimolDataCollator(device=self.device)
        self._create_dataloader(self.dataset, self.collate_fn)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-6)

        num_training_steps = len(self.train_dataloader) * self.num_epochs
        num_warmup_steps = int(self.warmup_ratio * num_training_steps)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
    def _create_dataloader(self, dataset: Union[Dataset, List], collate_fn: callable) -> DataLoader:
        dataset_size = len(dataset)
        if self.do_eval:
            train_size = int(dataset_size * self.split_ratio)
            eval_size = dataset_size - train_size

            if self.split_ratio >= 1.0:
                raise ValueError("split_ratio cannot be 1.0 when do_eval is True.")
            if eval_size <= 0:
                raise ValueError("Invalid split_ratio. No samples left for evaluation.")

            self.train_dataset, self.eval_dataset = random_split(dataset, [train_size, eval_size],
                                                                generator=torch.Generator().manual_seed(self.split_seed))
            self.train_dataloader = DataLoader(self.train_dataset, collate_fn=collate_fn, batch_size=self.batch_size, shuffle=True)
            self.eval_dataloader = DataLoader(self.eval_dataset, collate_fn=collate_fn, batch_size=self.batch_size, shuffle=False)
        else:
            self.train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=self.batch_size, shuffle=True)
            self.eval_dataloader = None

    def train(self):
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            self.model.train()

            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            tqdm.write(f"Epoch {epoch + 1} - Loss: {avg_epoch_loss:.4f}")

            if not self.do_eval:
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

            elif self.do_eval:
                eval_loss = self.evaluate()
                tqdm.write(f"Evaluation Loss after Epoch {epoch + 1}: {eval_loss:.4f}")

                if eval_loss < best_loss:
                    best_loss = eval_loss
                    patience_counter = 0
                else:
                    patience_counter += 1


            if patience_counter >= self.patience:
                tqdm.write(f"Early stopping triggered ({'Evaluating' if self.do_eval else 'Training'}) - No improvement in loss for {self.patience} epochs.")
                break

        print("Training finished.")

    def evaluate(self) -> float:
        if not self.do_eval:
            raise ValueError("Cannot evaluate when do_eval is False.")

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(self.eval_dataloader)
        return avg_loss
    

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class CustomTest:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.device = trainer.device
        self.model = trainer.model
        self.train_dataloader = trainer.train_dataloader
        self.eval_dataloader = trainer.eval_dataloader
        self.transform = trainer.transform

    def evaluate_metrics(self, dataloader: DataLoader):
        """Calculate evaluation metrics for the model."""
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                y_true.append(batch['labels'].cpu().numpy())
                y_pred.append(outputs.logits[:, 0, :].cpu().numpy())

        y_true = np.concatenate(y_true, axis=0).reshape(-1, 1)
        y_pred = np.concatenate(y_pred, axis=0).reshape(-1, 1)
        y_true = self.transform.inverse_transform(y_true).squeeze()
        y_pred = self.transform.inverse_transform(y_pred).squeeze()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return mse, mae, r2, y_true, y_pred

    def plot_metrics(self, y_true_train, y_pred_train, y_true_eval, y_pred_eval):
        """Plot evaluation metrics for the model."""
        residuals_train = y_true_train - y_pred_train
        residuals_eval = y_true_eval - y_pred_eval
        errors_train = y_pred_train - y_true_train
        errors_eval = y_pred_eval - y_true_eval

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].scatter(y_true_train, y_pred_train, color='blue', alpha=0.5, label='Train')
        axs[0].scatter(y_true_eval, y_pred_eval, color='red', alpha=0.5, label='Eval')
        axs[0].plot([min(y_true_train), max(y_true_train)], [min(y_true_train), max(y_true_train)], color='black', linestyle='--')
        axs[0].set_xlabel("True Values")
        axs[0].set_ylabel("Predicted Values")
        axs[0].set_title("True vs Predicted")
        axs[0].legend()

        axs[1].scatter(y_pred_train, residuals_train, color='blue', alpha=0.5, label='Train')
        axs[1].scatter(y_pred_eval, residuals_eval, color='red', alpha=0.5, label='Eval')
        axs[1].axhline(0, color='black', linestyle='--')
        axs[1].set_xlabel("Predicted Values")
        axs[1].set_ylabel("Residuals")
        axs[1].set_title("Residuals Plot")
        axs[1].legend()

        axs[2].hist(errors_train, bins=30, color='blue', alpha=0.7, label='Train')
        axs[2].hist(errors_eval, bins=30, color='red', alpha=0.7, label='Eval')
        axs[2].set_xlabel("Prediction Error")
        axs[2].set_ylabel("Frequency")
        axs[2].set_title("Prediction Error Histogram")
        axs[2].legend()

        plt.tight_layout()
        plt.savefig('evaluation_metrics.png')
        # plt.show()

    def save_metrics(self, mse_train, mae_train, r2_train, mse_eval, mae_eval, r2_eval):
        """Save evaluation metrics to an Excel file."""
        metrics = {
            'Metric': ['Train MSE', 'Train MAE', 'Train R²', 'Eval MSE', 'Eval MAE', 'Eval R²'],
            'Value': [mse_train, mae_train, r2_train, mse_eval, mae_eval, r2_eval]
        }
        df = pd.DataFrame(metrics)
        df.to_excel('evaluation_metrics.xlsx', index=False)

    def run(self):
        """Run the evaluation pipeline."""
        if self.eval_dataloader is None:
            raise ValueError("Evaluation dataloader is not available. Ensure `do_eval=True` in Trainer.")

        mse_train, mae_train, r2_train, y_true_train, y_pred_train = self.evaluate_metrics(self.train_dataloader)
        mse_eval, mae_eval, r2_eval, y_true_eval, y_pred_eval = self.evaluate_metrics(self.eval_dataloader)
        self.plot_metrics(y_true_train, y_pred_train, y_true_eval, y_pred_eval)
        self.save_metrics(mse_train, mae_train, r2_train, mse_eval, mae_eval, r2_eval)

        print(f"Metrics saved to 'evaluation_metrics.xlsx' and plots saved to 'evaluation_metrics.png'.")



if __name__ == "__main__":
    
    params = {
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'warmup_ratio': 0.1,
        'patience': 10,
        'batch_size': 16,
        'split_seed': 42,
        'split_ratio': 0.8,
        'do_eval': True
    }

    # model = ...  # transformers模型
    # dataloader = DataLoader(...)  # 你的dataloader
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)

    # 调用训练函数
    # train_model(model, dataloader, device, params)
