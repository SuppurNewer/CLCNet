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
        self.data_path = data_path        # input: data_preprocess/<gstp_name>/<trait>/
        self.save_path = save_path        # output: data_selected/<gstp_name>/<trait>/
        self.data_save_path = os.path.join(data_path, gstp_name)

    def load_and_select_features(self, importance_df, threshold=0, num_features=None):
        """Select feature indices based on importance"""
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
            trait_input_path = os.path.join(self.data_save_path, trait)
            trait_output_path = os.path.join(self.save_path, self.gstp_name, trait)
            os.makedirs(trait_output_path, exist_ok=True)
            print(f"Processing trait: {trait}")

            # Load full dataset
            data_full_path = os.path.join(trait_input_path, f"data_full_{trait}.npz")
            data_full = np.load(data_full_path, allow_pickle=True)
            X_full = data_full['X']
            y_full = data_full['y']
            samples = data_full['samples']

            # Select global features
            csv_global_path = os.path.join(trait_input_path, f"all_{trait}_feature_importance.csv")
            csv_global_df = pd.read_csv(csv_global_path)
            selected_global = self.load_and_select_features(csv_global_df)
            print(f"  Global features selected: {len(selected_global)}")

            # Select chromosome-aware features
            combined_indices = set(selected_global)
            start_index = 0
            for chr in self.chr_name:
                csv_local_path = os.path.join(trait_input_path, f"chr{chr}_{trait}_feature_importance.csv")
                if not os.path.exists(csv_local_path):
                    print(f"  Warning: chromosome file not found: {csv_local_path}, skipped")
                    continue
                csv_local_df = pd.read_csv(csv_local_path)
                selected_local = self.load_and_select_features(csv_local_df)
                combined_indices.update(selected_local + start_index)
                start_index += len(csv_local_df)
                print(f"  Chromosome {chr} features selected: {len(selected_local)}")

            combined_indices = np.array(sorted(combined_indices))
            print(f"  Total combined features: {len(combined_indices)}")

            # Save processed dataset
            np.savez_compressed(
                os.path.join(trait_output_path, f"data_selected_{trait}.npz"),
                X=X_full[:, combined_indices],
                y=y_full,
                samples=samples
            )
            np.savetxt(
                os.path.join(trait_output_path, f"selected_feature_indices_{trait}.txt"),
                combined_indices,
                fmt="%d"
            )
            np.savetxt(
                os.path.join(trait_output_path, f"samples_{trait}.txt"),
                samples,
                fmt="%s"
            )

            print(f"  Saved processed data and selected features for trait {trait}\n")


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
