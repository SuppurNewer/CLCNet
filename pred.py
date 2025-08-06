import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import CLCNet_origin
import time
import argparse

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

class MyDataset_val(Dataset):
    def __init__(self, gene_data, pheno_data):
        self.data = torch.tensor(gene_data, dtype=torch.float32)
        self.pheno_data = torch.tensor(np.array(pheno_data, dtype=np.float32), dtype=torch.float32)

    def __getitem__(self, index):
        feature1 = self.data[index]
        label1 = self.pheno_data[index]
        return feature1, torch.unsqueeze(label1, 0)

    def __len__(self):
        return self.data.shape[0]

class CLCNetEvaluator:
    def __init__(self, input_path, model_path, GSTP_NAME, traits, batch_size=16, device=None):
        self.input_path = input_path
        self.model_path = model_path
        self.GSTP_NAME = GSTP_NAME
        self.traits = traits if isinstance(traits, list) else [traits]
        self.batch_size = batch_size
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def load_local_data(self, trait, fold):
        train_local_path = os.path.join(self.input_path, self.GSTP_NAME, f"data_chromo_combined_fold_{fold}_{trait}_train.npz")
        test_local_path = os.path.join(self.input_path, self.GSTP_NAME, f"data_chromo_combined_fold_{fold}_{trait}_test.npz")

        train_local_data = np.load(train_local_path, allow_pickle=True, mmap_mode='r')
        test_local_data = np.load(test_local_path, allow_pickle=True, mmap_mode='r')

        gene_train_local = train_local_data['gene_train']
        pheno_train_local = train_local_data['pheno_train']

        gene_test_local = test_local_data['gene_test']
        pheno_test_local = test_local_data['pheno_test']

        return gene_train_local, pheno_train_local, gene_test_local, pheno_test_local

    def evaluate(self):
        results_values = []

        for trait in self.traits:
            print(f"Evaluating Trait: {trait}")
            for fold in range(10):
                gene_train, pheno_train, gene_test, pheno_test = self.load_local_data(trait, fold)

                dataset_val = MyDataset_val(gene_test, pheno_test)
                dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, drop_last=False)

                # 获取输入维度
                for inputs, _ in dataloader_val:
                    x_cat = inputs
                    break

                model_file = os.path.join(self.model_path, f'{self.GSTP_NAME}_{trait}_fold_{fold}_local_aware.pth')
                model = CLCNet_origin(shuffle=False, input_dim=x_cat.shape[1], shared_dim=[4096, 2048, 1024]).to(self.device)

                ckpt = torch.load(model_file, map_location=self.device)
                model.load_state_dict(ckpt['net'])
                model.eval()

                with torch.no_grad():
                    start_time = time.time()
                    output_val_list_1 = []
                    target_list_1 = []
                    loop_val = tqdm(enumerate(dataloader_val), total=len(dataloader_val))
                    for batch_idx, (data1, target1) in loop_val:
                        data1, target1 = data1.to(self.device), target1.to(self.device)
                        output_pri1, _ = model(data1)

                        mse_val1 = torch.mean((output_pri1 - target1) ** 2)
                        rmse_val1 = torch.sqrt(mse_val1)

                        loop_val.set_postfix(rmse_val1=rmse_val1.item(),
                                             mse_val1=mse_val1.item())

                        output_val_list_1.append(output_pri1.view(-1).cpu().numpy())
                        target_list_1.append(target1.view(-1).cpu().numpy())

                    output_val_list_cpu_1 = np.concatenate(output_val_list_1)
                    target_list_cpu_1 = np.concatenate(target_list_1)
                    mse_vall_all = np.mean((output_val_list_cpu_1 - target_list_cpu_1) ** 2)
                    correlation_matrix_1 = np.corrcoef(output_val_list_cpu_1, target_list_cpu_1)
                    pearson_coefficient_1 = correlation_matrix_1[0, 1]

                    print(f"Pheno: {trait}, Fold: {fold}")
                    print(f"Val PCC : {100 * pearson_coefficient_1:.4f} %, MSE: {mse_vall_all:.4f}")

                    process_time = time.time() - start_time
                    print("Process time:", format_time(process_time))
                    print()

                    results_values.extend([[trait, fold, true_val, pred_val]
                                           for true_val, pred_val in zip(target_list_cpu_1, output_val_list_cpu_1)])

        df_results = pd.DataFrame(results_values, columns=['Pheno', 'Fold', 'True', 'Pred'])
        output_excel = f"{self.GSTP_NAME}_CLCNet_local_aware_eval.xlsx"
        df_results.to_excel(output_excel, sheet_name='Results', index=False)
        print(f"Results saved to {output_excel}")

def main():
    parser = argparse.ArgumentParser(description="CLCNet Model Evaluation")
    parser.add_argument('--input_path', type=str, required=True, help="Path to input data folder")
    parser.add_argument('--model_path', type=str, required=True, help="Path to saved model weights folder")
    parser.add_argument('--GSTP_NAME', type=str, required=True, help="Dataset/GSTP name")
    parser.add_argument('--traits', nargs='+', required=True, help="Phenotype traits to evaluate, space separated")
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

if __name__ == '__main__':
    import argparse
    main()
