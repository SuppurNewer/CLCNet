import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import DNNGP
import torch.nn as nn
import torch.optim as optim
import time
import argparse

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def load_checkpoint(model, optimizer, filepath):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']})")
    else:
        start_epoch = 0
    return start_epoch

def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

class MyDataset_train(Dataset):
    def __init__(self, gene_data, pheno_data):
        self.gene_data = torch.tensor(np.array(gene_data, dtype=np.float32), dtype=torch.float32)
        self.pheno_data = torch.tensor(np.array(pheno_data, dtype=np.float32), dtype=torch.float32)

    def __getitem__(self, index):
        feature1 = self.gene_data[index]
        label1 = self.pheno_data[index]
        return torch.unsqueeze(feature1, 0), torch.unsqueeze(label1, 0)

    def __len__(self):
        return len(self.gene_data)

class MyDataset_val(Dataset):
    def __init__(self, gene_data, pheno_data):
        self.gene_data = torch.tensor(np.array(gene_data, dtype=np.float32), dtype=torch.float32)
        self.pheno_data = torch.tensor(np.array(pheno_data, dtype=np.float32), dtype=torch.float32)

    def __getitem__(self, index):
        feature1 = self.gene_data[index]
        label1 = self.pheno_data[index]
        return torch.unsqueeze(feature1, 0), torch.unsqueeze(label1, 0)

    def __len__(self):
        return len(self.gene_data)

class DNNGPTrainer:
    def __init__(self, data_path, save_path, GSTP_NAME, pheno_list, batch_size=16, epochs=100, lr=1e-3, weight_decay=1e-4):
        self.data_path = data_path
        self.save_path = save_path
        self.GSTP_NAME = GSTP_NAME
        self.pheno_list = pheno_list if isinstance(pheno_list, list) else [pheno_list]
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        for trait in self.pheno_list:
            for fold in range(10):
                gene_file = os.path.join(self.data_path, self.GSTP_NAME, trait, f"gene_data_fold_{fold}.npz")
                pheno_file = os.path.join(self.data_path, self.GSTP_NAME, trait, f"pheno_data_fold_{fold}.npz")

                gene_data = np.load(gene_file)
                pheno_data = np.load(pheno_file)

                X_train_fold = gene_data['X_train_fold']
                X_val_fold = gene_data['X_val_fold']
                y_train_fold = pheno_data['y_train_fold']
                y_val_fold = pheno_data['y_val_fold']

                dataset_train = MyDataset_train(X_train_fold, y_train_fold)
                dataset_val = MyDataset_val(X_val_fold, y_val_fold)

                dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False)
                dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, drop_last=False)

                # 取一批数据确定输入维度
                for inputs, _ in dataloader_val:
                    x_cat = inputs
                    break

                Geno_model = DNNGP(x_cat.shape[2]).to(self.device)
                initialize_weights(Geno_model)

                criterion = nn.MSELoss()
                optimizer = optim.Adam(Geno_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

                checkpoint_path = os.path.join(self.save_path, f'{self.GSTP_NAME}_{trait}_fold_{fold}_dnngp.pth')
                start_epoch = load_checkpoint(Geno_model, optimizer, checkpoint_path)

                for epoch in range(start_epoch, self.epochs):
                    start_time = time.time()
                    Geno_model.train()

                    output_train_list_1 = []
                    target_train_list_1 = []

                    loop = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
                    for idx, (input1, label1) in loop:
                        input1, label1 = input1.to(self.device), label1.to(self.device)
                        optimizer.zero_grad()
                        train_pri1 = Geno_model(input1)
                        train_loss = criterion(train_pri1, label1)
                        train_loss.backward()
                        optimizer.step()

                        mse_train1 = torch.mean((train_pri1 - label1) ** 2)
                        rmse_train1 = torch.sqrt(mse_train1)

                        loop.set_description(f'Train Epoch [{epoch}]')
                        loop.set_postfix(rmse_train1=rmse_train1.item())

                        output_train_list_1.append(train_pri1.view(-1).detach().cpu().numpy())
                        target_train_list_1.append(label1.view(-1).detach().cpu().numpy())

                    output_train_list_cpu_1 = np.concatenate(output_train_list_1)
                    target_train_list_cpu_1 = np.concatenate(target_train_list_1)
                    mse_train_1all = np.mean((output_train_list_cpu_1 - target_train_list_cpu_1) ** 2)
                    correlation_matrix_train_1 = np.corrcoef(output_train_list_cpu_1, target_train_list_cpu_1)
                    pearson_coefficient_train_1 = correlation_matrix_train_1[0, 1]

                    print(f'Train PCC : {100 * pearson_coefficient_train_1:.4f} %, MSE 1: {mse_train_1all:.4f}')

                    Geno_model.eval()
                    with torch.no_grad():
                        output_val_list_1 = []
                        target_list_1 = []
                        loop_val = tqdm(enumerate(dataloader_val), total=len(dataloader_val))
                        for batch_idx, (data1, target1) in loop_val:
                            data1, target1 = data1.to(self.device), target1.to(self.device)
                            output_pri1 = Geno_model(data1)

                            mse_val1 = torch.mean((output_pri1 - target1) ** 2)
                            rmse_val1 = torch.sqrt(mse_val1)

                            loop_val.set_description(f'Val Epoch [{epoch}]')
                            loop_val.set_postfix(rmse_val1=rmse_val1.item(),
                                                mse_val1=mse_val1.item())

                            output_val_list_1.append(output_pri1.view(-1).detach().cpu().numpy())
                            target_list_1.append(target1.view(-1).detach().cpu().numpy())

                        output_val_list_cpu_1 = np.concatenate(output_val_list_1)
                        target_list_cpu_1 = np.concatenate(target_list_1)
                        mse_vall_all = np.mean((output_val_list_cpu_1 - target_list_cpu_1) ** 2)
                        correlation_matrix_1 = np.corrcoef(output_val_list_cpu_1, target_list_cpu_1)
                        pearson_coefficient_1 = correlation_matrix_1[0, 1]

                        print(f'Val PCC : {100 * pearson_coefficient_1:.4f} %, MSE: {mse_vall_all:.4f}')

                        process_time = time.time() - start_time
                        print("Process:", format_time(process_time))
                        print()

                        state = {
                            'net': Geno_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch
                        }
                        torch.save(state, checkpoint_path)

def main():
    parser = argparse.ArgumentParser(description="Train DNNGP Model")
    parser.add_argument('--data_path', type=str, required=True, help='Input data directory')
    parser.add_argument('--save_path', type=str, required=True, help='Model saving directory')
    parser.add_argument('--GSTP_NAME', type=str, required=True, help='Dataset name')
    parser.add_argument('--traits', nargs='+', required=True, help='Phenotype trait list')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()

    trainer = DNNGPTrainer(
        data_path=args.data_path,
        save_path=args.save_path,
        GSTP_NAME=args.GSTP_NAME,
        pheno_list=args.traits,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    trainer.train()

if __name__ == '__main__':
    main()
