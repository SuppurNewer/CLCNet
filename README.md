# CLCNet
**CLCNet: A Contrastive Learning and Chromosome-aware Network for Genomic Prediction in Plants**

## Model Architecture
We propose **CLCNet**, a novel deep learning framework for genomic prediction that integrates contrastive learning with a multi-task predictive architecture and a chromosome-aware feature selection strategy.
The framework is specifically designed to address the challenges of high-dimensional genomic data and limited sample sizes in plant breeding applications.
<p align="center">
  <img src="fig/Figure 1.png" width="600" alt="CLCNet architecture"/>
</p>

CLCNet consists of two key components:
(i) a **chromosome-aware (CA) module** for structured SNP representation, and (ii) a **contrastive learning module** to enhance representation robustness and generalization.

## Key Features
- **Chromosome-aware feature selection**  
  Explicitly models chromosomal structures of SNPs, including linkage disequilibrium (LD) patterns and epistatic interactions.

- **Contrastive learning for genomic representation**  
  Improves robustness and generalization under high-dimensional and low-sample-size settings (e.g.,>100,000 features vs. <10,000 samples).

- **Multi-task prediction framework**  
  Reinforces representation learning by explicitly modeling inter-sample differences in
  genotype–phenotype relationships.

## 1.0 Installation
```bash
conda create -n CLCNet python=3.10.13
conda activate CLCNet
pip install -r requirements.txt
```
## 2.0 Data Preparation
Example genotype data in PLINK format (.bed, .bim, .fam) and phenotype data (.pheno) are provided in the example/ directory.
To perform chromosome-aware feature selection, run:
```bash
python ChromosomeAwareProcessor.py \
  --gstp_name example \
  --data_dir example \
  --traits Trait1 Trait2
```
### 2.1 Processed results
Processed results are saved under:
```php-template
data_preprocess/<gstp_name>/<trait>/ directory
```
For each trait:
| File                                        | Description                                                   |
| ------------------------------------------- | ------------------------------------------------------------- |
| `data_full_<trait>.npz`                     | Standardized genotype and phenotype arrays for model training |
| `samples_<trait>.txt`                       | List of sample names                                          |
| `all_<trait>_feature_importance.csv`        | Global SNP feature importance from LightGBM                   |
| `all_<trait>_lgb.txt`                       | Global LightGBM model file                                    |
| `chr<chrom>_<trait>_feature_importance.csv` | Chromosome-specific SNP feature importance                    |
| `chr<chrom>_<trait>_lgb.txt`                | Chromosome-specific LightGBM model file                       |
- data_full_<trait>.npz contains arrays:
- X – genotype data
- y – standardized phenotype values
- samples – sample IDs
### 2.2 Feature selection
After generating feature importance files, selected SNPs can be combined using:
```bash
python ChromosomeAwareSelected.py \
  --gstp_name example \
  --traits Trait1 Trait2 \
  --chr_name_file chromosomes.txt \
  --data_path data_preprocess \
  --save_path data_selected
```
Arguments:
| Argument          | Type | Description                                                       |
| ----------------- | ---- | ----------------------------------------------------------------- |
| `--gstp_name`     | str  | Dataset name, matching the prefix of genotype and phenotype files |
| `--traits`        | list | Phenotype traits to process, e.g., `Trait1 Trait2`                |
| `--chr_name_file` | str  | Path to a text file listing chromosome names, one per line        |
| `--data_path`     | str  | Input path containing `data_preprocess/<gstp_name>/<trait>/`      |
| `--save_path`     | str  | Output path where selected features will be saved                 |
### 2.3 Selected Feature Outputs
After running ChromosomeAwareSelected.py, results are saved under:
```php-template
data_selected/<gstp_name>/<trait>/
```
For each trait:
| File                                   | Description                            |
| ---------------------------------------| ---------------------------------------|
| `data_selected_<trait>.npz`            | Genotype arrays for selected features  |
| `selected_feature_indices_<trait>.txt` | Indices of selected SNPs/features      |
| `samples_<trait>.txt`                  | List of sample names                   |
## 3.0 Training
Once data_selected is ready, you can train the contrastive model using the selected features:
```bash
python train.py \
  --input_path data_selected \
  --output_path weight \
  --GSTP_NAME example \
  --traits Trait1 Trait2 \
  --epochs 100 \
  --batch_size 32
```
- Input path: data_selected directory containing <gstp_name>/<trait>/
- Output path: directory where trained models will be saved (default: weight/)
- Epochs & Batch Size: configurable via command line
After training, models are saved as:
```php-template
weight/<GSTP_NAME>_<trait>_local_aware.pth
```
## 4.0 Prediction
After training, evaluate the model using the selected features:
```bash
python pred.py \
  --input_path data_selected \
  --model_path weight \
  --GSTP_NAME example \
  --traits Trait1 Trait2 \
  --batch_size 16
```
- Input path: same data_selected directory used for training
- Model path: directory containing trained .pth weights
- Batch size: configurable
Evaluation results for each trait are saved as:
```php-template
<gstp_name>_<trait>_pred_results.xlsx
```
A combined results file across all traits is saved as:
```php-template
<gstp_name>_CLCNet_all_traits_pred.xlsx
```
## 5.0 Notes
1.Ensure that the chromosome names in chromosomes.txt match those used during feature selection.  
2.The workflow is fully compatible:
```nginx
ChromosomeAwareProcessor.py → ChromosomeAwareSelected.py → train.py → pred.py
```
3.Selected features and sample indices are stored for reproducibility and downstream analyses.  
4.Validation split is automatically handled (default: 80/20 split).  
## Citation
If you find this work useful, please cite:
```markdown
https://doi.org/10.1101/2024.12.29.630569
```
```bibtex
@article{
  title   = {CLCNet: A Contrastive Learning and Chromosome-aware Network for Genomic Prediction in Plants},
  author  = {Jiangwei Huang#, Zhihan Yang#, Mou Yin, Chao Li, Jinmin Li, Yu Wang, Lu Huang, Miaomiao Li, Chengzhi Liang, Fei He, Rongcheng Han, and Yuqiang Jiang*},
  journal = {bioRxiv},
  year    = {2024}
}
```







