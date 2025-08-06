import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from tqdm import tqdm
from pandas_plink import read_plink1_bin

class GenomicFeatureProcessor:
    def __init__(self, gstp_name, data_dir, pheno_list):
        self.gstp_name = gstp_name
        self.data_dir = data_dir
        self.pheno_list = pheno_list

        self.bed_path = os.path.join(data_dir, gstp_name, f"{gstp_name}.bed", f"{gstp_name}.bed")
        self.bim_path = os.path.join(data_dir, gstp_name, f"{gstp_name}.bed", f"{gstp_name}.bim")
        self.fam_path = os.path.join(data_dir, gstp_name, f"{gstp_name}.bed", f"{gstp_name}.fam")
        self.pheno_path = os.path.join(data_dir, gstp_name, f"{gstp_name}.pheno")
        self.output_dir = os.path.join("data_preprocess", gstp_name)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def delete_NA(gene_data, pheno_data, sample_name):
        gene_data = np.array(gene_data, dtype=np.int8)
        gene_data = np.nan_to_num(gene_data, nan=3)
        pheno_data = np.array([np.nan if x == 'NA' else float(x) for x in pheno_data])
        na_mask = np.isnan(pheno_data)
        return gene_data[~na_mask], pheno_data[~na_mask], sample_name[~na_mask]

    @staticmethod
    def split_by_chromosome(data, chrom_names):
        unique_chromosomes = np.unique(chrom_names)
        return {chrom: data[:, chrom_names == chrom] for chrom in unique_chromosomes}

    @staticmethod
    def feature_selection(X, y, num_features=2500):
        X, y = X.astype(np.float32), y.astype(np.float32)
        train_data = lgb.Dataset(X, label=y)
        params = {'objective': 'regression', 'metric': 'mse', 'device': 'cpu', 'force_col_wise': True}
        bst = lgb.train(params, train_set=train_data, num_boost_round=100)
        importances = bst.feature_importance(importance_type='gain')
        return np.sort(np.argsort(importances)[::-1][:num_features])

    def feature_selection_save(self, X, y, path, trait, fold, chrom=None):
        X, y = X.astype(np.float32), y.astype(np.float32)
        bst = lgb.train({'objective': 'regression', 'metric': 'mse', 'device': 'cpu', 'force_col_wise': True},
                        train_set=lgb.Dataset(X, label=y), num_boost_round=100)
        importances = bst.feature_importance(importance_type='gain')
        indices = np.sort(np.where(importances > 0)[0])
        df = pd.DataFrame({'feature': np.arange(len(importances)), 'importance': importances})
        tag = f"chromo_{chrom}_" if chrom is not None else "all_"
        df.to_csv(os.path.join(path, f"{tag}{trait}_fold_{fold}_feature_importances.csv"), index=False)
        bst.save_model(os.path.join(path, f"{tag}{trait}_fold_{fold}_lgb_model.txt"))
        return indices

    def process(self):
        geno, sample_names, chrom_names, _ = read_plink1_bin(self.bed_path, self.bim_path, self.fam_path, verbose=True).values

        with open(self.pheno_path, 'r') as f:
            pheno_data = [line.strip().split() for line in f]
        pheno_data = np.array(pheno_data)

        for idx, name in enumerate(pheno_data[0, 1:]):
            if name in self.pheno_list:
                out_path = os.path.join(self.output_dir, name)
                os.makedirs(out_path, exist_ok=True)

                X, y, s = self.delete_NA(geno, pheno_data[1:, idx + 1], sample_names)
                kf = KFold(n_splits=10, shuffle=True, random_state=42)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    s_train, s_val = s[train_idx], s[val_idx]

                    np.savetxt(os.path.join(out_path, f'sample_data_fold_{fold}_train.txt'), s_train, fmt='%s')
                    np.savetxt(os.path.join(out_path, f'sample_data_fold_{fold}_val.txt'), s_val, fmt='%s')

                    y_train_mean, y_train_std = np.mean(y_train), np.std(y_train)
                    y_train_scaled = (y_train - y_train_mean) / y_train_std
                    y_val_scaled = (y_val - y_train_mean) / y_train_std

                    np.savez_compressed(os.path.join(out_path, f'gene_data_fold_{fold}.npz'),
                                        X_train_fold=X_train, X_val_fold=X_val)
                    np.savez_compressed(os.path.join(out_path, f'pheno_data_fold_{fold}.npz'),
                                        y_train_fold=y_train_scaled, y_val_fold=y_val_scaled)

                    self.feature_selection_save(X_train, y_train_scaled, out_path, name, fold)

                    X_train_chrom = self.split_by_chromosome(X_train, chrom_names)
                    for chrom, data in X_train_chrom.items():
                        self.feature_selection_save(data, y_train_scaled, out_path, name, fold, chrom)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gstp_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='CropGS_hub_datasets')
    parser.add_argument('--traits', nargs='+', required=True)
    args = parser.parse_args()

    processor = GenomicFeatureProcessor(args.gstp_name, args.data_dir, args.traits)
    processor.process()
