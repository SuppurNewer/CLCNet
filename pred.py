#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pred.py
Evaluate trained CLCNet models using data_selected outputs.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import CLCNet
import time
import argparse

# ------------------------------
# Utilities
# ------------------------------
def format_time(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{int(h)}h {int(m)}m {int(s)}s"

# ------------------------------
# Dataset class
# ------------------------------
class MyDatasetVal(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx].unsqueeze(0)

    def __len__(self):
        return self.X.shape[0]

# ------------------------------
# Evaluator class
# ------------------------------
class CLCNetEvaluator:
    def __init__(self, input_path, model_path, GSTP_NAME, traits, batch_size=16, device=None):
        self.input_path = input_path
        self.model_path = model_path
        self.GSTP_NAME = GSTP_NAME
        self.traits = traits if isinstance(traits, list) else [traits]
        self.batch_size = batch_size
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def load_selected_data(self, trait):
        """Load selected features and phenotype labels from data_selected"""
        file_path = os.path.join(self.input_path, self.GSTP_NAME, trait, f"data_selected_{trait}.npz")
        data = np.load(file_path, allow_pickle=True)
        X = data['X'] if 'X' in data else data['gene_train']  # support older naming
        y = data['y'] if 'y' in data else data['pheno_train']
        return X, y

    def evaluate(self):
        all_results = []

        for trait in self.traits:
            print(f"=== Evaluating Trait: {trait} ===")
            X, y = self.load_selected_data(trait)

            # Here, split validation (last 20% as example)
            num_samples = X.shape[0]
            split_idx = int(0.8 * num_samples)
            X_val, y_val = X[split_idx:], y[split_idx:]

            dataset_val = MyDatasetVal(X_val, y_val)
            dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False)

            # Initialize model
            input_dim = X_val.shape[1]
            model = CLCNet(input_dim=input_dim, shared_dim=[4096, 2048, 1024]).to(self.device)

            # Load trained model
            model_file = os.path.join(self.model_path, f"{self.GSTP_NAME}_{trait}_local_aware.pth")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}")

            checkpoint = torch.load(model_file, map_location=self.device)
            model.load_state_dict(checkpoint['net'])
            model.eval()

            # Evaluation loop
            preds, trues = [], []
            start_time = time.time()
            for X_batch, y_batch in tqdm(dataloader_val, desc=f"{trait} evaluation"):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                with torch.no_grad():
                    output, _ = model(X_batch)
                preds.append(output.view(-1).cpu().numpy())
                trues.append(y_batch.view(-1).cpu().numpy())

            preds = np.concatenate(preds)
            trues = np.concatenate(trues)

            mse = np.mean((preds - trues) ** 2)
            pcc = np.corrcoef(preds, trues)[0, 1]

            print(f"Trait: {trait}, MSE: {mse:.4f}, PCC: {100*pcc:.2f}%")
            print("Evaluation time:", format_time(time.time() - start_time))

            # Save per-trait results
            results_df = pd.DataFrame({
                'Trait': [trait]*len(trues),
                'True': trues,
                'Pred': preds
            })
            results_file = f"{self.GSTP_NAME}_{trait}_pred_results.xlsx"
            results_df.to_excel(results_file, index=False)
            print(f"Results saved to {results_file}\n")

            all_results.append(results_df)

        # Optionally combine all traits
        combined_file = f"{self.GSTP_NAME}_CLCNet_all_traits_pred.xlsx"
        pd.concat(all_results, ignore_index=True).to_excel(combined_file, index=False)
        print(f"Combined results saved to {combined_file}")

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="CLCNet Prediction on Selected Features")
    parser.add_argument('--input_path', type=str, required=True, help="Path to data_selected folder")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model weights")
    parser.add_argument('--GSTP_NAME', type=str, required=True, help="Dataset/GSTP name")
    parser.add_argument('--traits', nargs='+', required=True, help="Phenotype traits to evaluate")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for evaluation")
    args = parser.parse_args()

    evaluator = CLCNetEvaluator(
        input_path=args.input_path,
        model_path=args.model_path,
        GSTP_NAME=args.GSTP_NAME,
        traits=args.traits,
        batch_size=args.batch_size
    )
    evaluator.evaluate()

if __name__ == "__main__":
    main()
