import os
import numpy as np
import pandas as pd
import argparse

class ChromosomeAwareSelected:
    def __init__(self, gstp_name, pheno_list, chr_name, 
                 data_path='data_preprocess', save_path='data_selected'):
        self.gstp_name = gstp_name
        self.pheno_list = pheno_list
        self.chr_name = chr_name
        self.data_path = data_path
        self.save_path = save_path
        self.data_save_path = os.path.join(data_path, gstp_name)

    def load_and_select_features(self, importance_df, threshold=0, num_features=None):
        feature_importances = importance_df['importance'].values
        valid_indices = np.where(feature_importances > threshold)[0]
        valid_importances = feature_importances[valid_indices]

        if num_features is not None:
            top_indices = np.argsort(valid_importances)[::-1][:num_features]
            selected_indices = np.sort(valid_indices[top_indices])
        else:
            selected_indices = np.sort(valid_indices)
        return selected_indices

    def process(self):
        for trait in self.pheno_list:
            trait_path = os.path.join(self.save_path, self.gstp_name)
            os.makedirs(trait_path, exist_ok=True)
            print(f"Processing trait: {trait}")

            selected_indices_global_num = []
            selected_indices_local_num = []

            for fold in range(10):
                print(f"  Fold: {fold+1}")
                gene_data_path = os.path.join(self.data_save_path, trait, f"gene_data_fold_{fold}.npz")
                pheno_data_path = os.path.join(self.data_save_path, trait, f"pheno_data_fold_{fold}.npz")
                gene_data = np.load(gene_data_path, allow_pickle=True, mmap_mode='r')
                pheno_data = np.load(pheno_data_path, allow_pickle=True, mmap_mode='r')

                X_train = gene_data['X_train_fold']
                X_val = gene_data['X_val_fold']
                y_train = pheno_data['y_train_fold']
                y_val = pheno_data['y_val_fold']

                csv_global_path = os.path.join(self.data_save_path, trait, f"all_{trait}_fold_{fold}_feature_importances.csv")
                csv_global_df = pd.read_csv(csv_global_path)
                selected_global = self.load_and_select_features(csv_global_df)
                selected_indices_global_num.append(len(selected_global))
                combined_indices = set(selected_global)

                start_index = 0
                for chr in self.chr_name:
                    csv_local_path = os.path.join(self.data_save_path, trait, f"chromo_{chr}_{trait}_fold_{fold}_feature_importances.csv")
                    csv_local_df = pd.read_csv(csv_local_path)
                    selected_local = self.load_and_select_features(csv_local_df)
                    selected_indices_local_num.append(len(selected_local))
                    combined_indices.update(selected_local + start_index)
                    start_index += len(csv_local_df)

                combined_indices = np.array(sorted(combined_indices))

                np.savez_compressed(
                    os.path.join(trait_path, f"data_chromo_combined_fold_{fold}_{trait}_train.npz"),
                    gene_train=X_train[:, combined_indices],
                    pheno_train=y_train
                )
                np.savez_compressed(
                    os.path.join(trait_path, f"data_chromo_combined_fold_{fold}_{trait}_test.npz"),
                    gene_test=X_val[:, combined_indices],
                    pheno_test=y_val
                )
                np.savetxt(
                    os.path.join(trait_path, f"selected_feature_indices_fold_{fold}_{trait}.txt"),
                    combined_indices,
                    fmt="%d"
                )

            print(f"    Min global selected features: {min(selected_indices_global_num)}")
            print(f"    Min local selected features: {min(selected_indices_local_num)}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gstp_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--traits', nargs='+', required=True, help='List of phenotype traits')
    parser.add_argument('--chr_name_file', type=str, required=True, help='Path to txt file containing chromosome names')
    parser.add_argument('--data_path', type=str, default='data_preprocess')
    parser.add_argument('--save_path', type=str, default='data_selected')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.chr_name_file, 'r') as f:
        chr_list = [line.strip() for line in f if line.strip()]

    processor = ChromosomeAwareSelected(
        gstp_name=args.gstp_name,
        pheno_list=args.traits,
        chr_name=chr_list,
        data_path=args.data_path,
        save_path=args.save_path
    )
    processor.process()


if __name__ == '__main__':
    main()
