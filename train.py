import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, phenotype_diff):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((euclidean_distance - phenotype_diff)**2)
        return loss

class MyDatasetTrain(Dataset):
    def __init__(self, gene_data, pheno_data):
        self.data = torch.tensor(gene_data, dtype=torch.float32)
        self.pheno_data = torch.tensor(np.array(pheno_data, dtype=np.float32), dtype=torch.float32)

    def __getitem__(self, index):
        idx1 = index
        idx2 = torch.randint(0, len(self.data), (1,)).item()
        feature1 = self.data[idx1]
        feature2 = self.data[idx2]
        label1 = self.pheno_data[idx1]
        label2 = self.pheno_data[idx2]
        label_diff = abs(label1 - label2)**2
        return feature1, feature2, label1.unsqueeze(0), label2.unsqueeze(0), label_diff

    def __len__(self):
        return self.data.shape[0]

class MyDatasetVal(Dataset):
    def __init__(self, gene_data, pheno_data):
        self.data = torch.tensor(gene_data, dtype=torch.float32)
        self.pheno_data = torch.tensor(pheno_data, dtype=torch.float32)

    def __getitem__(self, index):
        return self.data[index], self.pheno_data[index].unsqueeze(0)

    def __len__(self):
        return self.data.shape[0]

class ContrastiveTrainer:
    def __init__(self, model_class, input_path, output_path, GSTP_NAME, pheno_list, epochs=100):
        self.model_class = model_class
        self.input_path = input_path
        self.output_path = output_path
        self.GSTP_NAME = GSTP_NAME
        self.pheno_list = pheno_list
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self, trait, fold):
        train_path = os.path.join(self.input_path, self.GSTP_NAME, f"data_chromo_combined_fold_{fold}_{trait}_train.npz")
        test_path = os.path.join(self.input_path, self.GSTP_NAME, f"data_chromo_combined_fold_{fold}_{trait}_test.npz")
        train_data = np.load(train_path, allow_pickle=True)
        test_data = np.load(test_path, allow_pickle=True)
        return train_data['gene_train'], train_data['pheno_train'], test_data['gene_test'], test_data['pheno_test']

    def train(self):
        for trait in self.pheno_list:
            for fold in range(10):
                gene_train, pheno_train, gene_test, pheno_test = self.load_data(trait, fold)
                train_loader = DataLoader(MyDatasetTrain(gene_train, pheno_train), batch_size=32, shuffle=True)
                val_loader = DataLoader(MyDatasetVal(gene_test, pheno_test), batch_size=32)

                input_dim = gene_train.shape[1]
                model = self.model_class(shuffle=False, input_dim=input_dim, shared_dim=[4096,2048,1024]).to(self.device)
                optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-6)
                criterion = nn.MSELoss()
                contrastive_loss = ContrastiveLoss()

                save_name = f"{self.GSTP_NAME}_{trait}_fold_{fold}_local_aware.pth"
                save_path = os.path.join(self.output_path, save_name)
                start_epoch = self.load_checkpoint(model, optimizer, save_path)

                for epoch in range(start_epoch, self.epochs):
                    start_time = time.time()
                    model.train()
                    train_loss = self.run_epoch(model, optimizer, train_loader, criterion, contrastive_loss, epoch, fold, trait)
                    scheduler.step(train_loss)
                    self.evaluate(model, val_loader, criterion, epoch, fold, trait)
                    self.save_checkpoint(model, optimizer, epoch, save_path)
                    print("Time elapsed:", self.format_time(time.time() - start_time))

    def run_epoch(self, model, optimizer, loader, criterion, contrastive_loss, epoch, fold, trait):
        total_loss = 0
        for inputs1, inputs2, labels1, labels2, diffs in tqdm(loader, desc=f"{trait} Fold {fold} Epoch {epoch}"):
            input1, input2 = inputs1.to(self.device), inputs2.to(self.device)
            label1, label2, diff = labels1.to(self.device), labels2.to(self.device), diffs.to(self.device)
            optimizer.zero_grad()
            out1, aux1 = model(input1)
            out2, aux2 = model(input2)
            loss = criterion(out1, label1) + criterion(out2, label2) + contrastive_loss(aux1, aux2, diff)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(self, model, loader, criterion, epoch, fold, trait):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred, _ = model(x)
                preds.append(pred.view(-1).cpu().numpy())
                targets.append(y.view(-1).cpu().numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        mse = np.mean((preds - targets) ** 2)
        pcc = np.corrcoef(preds, targets)[0, 1]
        print(f"Validation MSE: {mse:.4f}, PCC: {100 * pcc:.2f}%")

    def load_checkpoint(self, model, optimizer, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            return checkpoint['epoch'] + 1
        return 0

    def save_checkpoint(self, model, optimizer, epoch, path):
        state = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, path)

    @staticmethod
    def format_time(seconds):
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"


# Example usage
if __name__ == '__main__':
    import argparse
    from model import CLCNet

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to preprocessed data')
    parser.add_argument('--output_path', type=str, default='weight', help='Path to save models')
    parser.add_argument('--GSTP_NAME', type=str, required=True, help='Dataset name')
    parser.add_argument('--traits', nargs='+', required=True, help='List of phenotype names')
    args = parser.parse_args()

    trainer = ContrastiveTrainer(
        model_class=CLCNet,
        input_path=args.input_path,
        output_path=args.output_path,
        GSTP_NAME=args.GSTP_NAME,
        pheno_list=args.traits
    )
    trainer.train()
