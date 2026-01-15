import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from pandas_plink import read_plink1_bin


class ChromosomeAwareProcessor:
    def __init__(self, gstp_name, data_dir, pheno_list):
        """
        data_dir/
        └── example/
            ├── geno/
            │   ├── example.bed
            │   ├── example.bim
            │   └── example.fam
            └── pheno/
                └── example.pheno
        """
        self.gstp_name = gstp_name
        self.data_dir = data_dir
        self.pheno_list = pheno_list

        # ---------- PLINK paths ----------
        self.bed_path = os.path.join(data_dir, f"{gstp_name}.bed")
        self.bim_path = os.path.join(data_dir, f"{gstp_name}.bim")
        self.fam_path = os.path.join(data_dir, f"{gstp_name}.fam")

        # ---------- phenotype path ----------
        self.pheno_path = os.path.join(data_dir, f"{gstp_name}.pheno")

        # ---------- output ----------
        self.output_dir = os.path.join("data_preprocess", gstp_name)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def delete_NA(gene_data, pheno_data, sample_name):
        gene_data = np.asarray(gene_data, dtype=np.int8)
        gene_data = np.nan_to_num(gene_data, nan=3)

        pheno_data = np.array(
            [np.nan if x == 'NA' else float(x) for x in pheno_data],
            dtype=np.float32
        )

        na_mask = np.isnan(pheno_data)
        return gene_data[~na_mask], pheno_data[~na_mask], sample_name[~na_mask]

    @staticmethod
    def split_by_chromosome(data, chrom_names):
        unique_chromosomes = np.unique(chrom_names)
        return {chrom: data[:, chrom_names == chrom] for chrom in unique_chromosomes}

    def feature_selection_save(self, X, y, path, trait, chrom=None):
        X, y = X.astype(np.float32), y.astype(np.float32)

        train_data = lgb.Dataset(X, label=y)
        params = {
            "objective": "regression",
            "metric": "mse",
            "force_col_wise": True,
            "device": "cpu"
        }

        bst = lgb.train(params, train_data, num_boost_round=100)
        importances = bst.feature_importance(importance_type="gain")

        indices = np.where(importances > 0)[0]

        df = pd.DataFrame({
            "feature_index": np.arange(len(importances)),
            "importance": importances
        })

        prefix = f"chr{chrom}_" if chrom is not None else "all_"
        df.to_csv(
            os.path.join(path, f"{prefix}{trait}_feature_importance.csv"),
            index=False
        )

        bst.save_model(
            os.path.join(path, f"{prefix}{trait}_lgb.txt")
        )

        return indices

    def process(self):
        # ---------- load PLINK ----------
        geno, sample_names, chrom_names, _ = read_plink1_bin(
            self.bed_path,
            self.bim_path,
            self.fam_path,
            verbose=True
        ).values

        # ---------- load phenotype ----------
        pheno_df = pd.read_csv(self.pheno_path, sep=r"\s+")

        for trait in self.pheno_list:
            if trait not in pheno_df.columns:
                raise ValueError(f"Trait {trait} not found in phenotype file.")

            out_path = os.path.join(self.output_dir, trait)
            os.makedirs(out_path, exist_ok=True)

            # ---------- delete NA ----------
            X, y, s = self.delete_NA(
                geno,
                pheno_df[trait].values,
                sample_names
            )

            # ---------- normalize phenotype ----------
            y_mean, y_std = y.mean(), y.std()
            y_scaled = (y - y_mean) / y_std

            # ---------- save full dataset ----------
            np.savez_compressed(
                os.path.join(out_path, f"data_full_{trait}.npz"),
                X=X,
                y=y_scaled,
                samples=s
            )
            np.savetxt(
                os.path.join(out_path, f"samples_{trait}.txt"),
                s,
                fmt="%s"
            )

            # ---------- global feature selection ----------
            self.feature_selection_save(X, y_scaled, out_path, trait)

            # ---------- chromosome-aware feature selection ----------
            chrom_dict = self.split_by_chromosome(X, chrom_names)
            for chrom, data in chrom_dict.items():
                self.feature_selection_save(
                    data, y_scaled, out_path, trait, chrom
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gstp_name", type=str, required=True,
                        help="Dataset name (e.g., example)")
    parser.add_argument("--data_dir", type=str, default="example",
                        help="Path to example directory")
    parser.add_argument("--traits", nargs="+", required=True)

    args = parser.parse_args()

    processor = ChromosomeAwareProcessor(
        args.gstp_name,
        args.data_dir,
        args.traits
    )
    processor.process()
