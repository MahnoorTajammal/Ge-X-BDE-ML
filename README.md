# Ge-X bond dissociation energy (De) prediction: code and data

Code, data and results accompanying the article

**Machine Learning Prediction of Ge-X (X=C, H, N, O) Bond Dissociation Energies via DFT Trained Data: A Comparative Study**

**Authors:** Mahnoor Tajammal<sup>a</sup> (tajammalmahnoor@gmail.com) and Khurshid Ayub<sup>a,*</sup> (khurshid@cuiatd.edu.pk, ORCID 0000-0003-0990-1860)
<sup>a</sup> Department of Chemistry, COMSATS University, Abbottabad Campus, Abbottabad, 22060, Pakistan. \*Corresponding author.

**Archive DOI:** assigned by Zenodo at the first release (insert here and in the article's Code availability section).
**Licence:** MIT (see `LICENSE`).

## What this repository does
Machine-learning models (XGBoost, Random Forest, a descriptor-based feed-forward deep neural network and TPOT AutoML) are trained on 1,183 organogermanium compounds to reproduce the **electronic dissociation energy De** of Ge-X bonds (X = C, H, N, O) computed at the **CAM-B3LYP/LANL2DZ** level (Gaussian 16, Revision A.03). `De = E(fragment 1) + E(fragment 2) - E(parent)` from SCF energies of the optimized structures, with no zero-point or thermal correction. In the data files "BDE" means De. Models reproduce this DFT protocol, not experiment.

## Repository layout
```
Ge-X-BDE-ML/
|-- README.md, LICENSE, CITATION.cff, .zenodo.json, requirements.txt, .gitignore
|-- GE_BDE.csv                    raw dataset: compound_id, SMILES, broken_bond, BDE_EE_kcal_mol (= De)
|-- Ge-Test-Dataset.csv           the 183 external-test compounds (defines the split)
|-- GE_BDE_descriptors.csv        RDKit descriptors for all compounds (produced by Ge_X_BDE_revised.py)
|-- split_assignments.csv         compound_id, SMILES, broken_bond, set (train/test)
|-- Ge_X_BDE_revised.py           MAIN script (descriptors, split, CV, RF, XGBoost, DNN, baselines, ablations, conformal, SHAP, t-SNE, figures)
|-- Ge_X_BDE_TPOT.py              TPOT search on the training set only + external test
|-- Ge_X_BDE_TPOT_extras.py       no new search: cluster-grouped CV of the saved TPOT pipeline, TPOT figures, permutation importance
|-- scan_gaussian_logs.py         scans Gaussian logs (termination, <S2>, ZPE, De, D0, DH298)
|-- results/main/                 outputs of Ge_X_BDE_revised.py reported in the article
|-- results/tpot/                 outputs of Ge_X_BDE_TPOT.py (pipeline .pkl/.txt, predictions, manifest)
|-- results/tpot_extras/          outputs of Ge_X_BDE_TPOT_extras.py
|-- gaussian_summary/             jobs_final_1183.csv: one row per Gaussian job (3,549 jobs)
|-- benchmark/                    Ge benchmark against experimental bond enthalpies (inputs, logs, reference values, analysis script)
`-- supplementary/                Supplementary_Tables_S1-S4_v3.xlsx
```
The full Gaussian output files and optimized geometries of all species are archived in the Zenodo record (too large for GitHub).

## Data dictionary
- `GE_BDE.csv`: `compound_id` (compound_0001...), `SMILES`, `broken_bond` (Ge-C, Ge-H, Ge-N, Ge-O; counts {'Ge-H': 394, 'Ge-N': 272, 'Ge-O': 269, 'Ge-C': 248}), `BDE_EE_kcal_mol` (De, kcal/mol).
- `split_assignments.csv`: 1,000 training and 183 external-test compounds (stratified by bond type); test list in `Ge-Test-Dataset.csv`.
- `GE_BDE_descriptors.csv`: 33 descriptors + `Bond_Type` (fixed ordinal code Ge-C=0, Ge-H=1, Ge-N=2, Ge-O=3). Models use 19 curated descriptors + Bond_Type (20 inputs): MolWt, LogP, TPSA, HBD, HBA, EN_diff, X_TotalValence, X_degree, Ge_degree, Ge_nb_C_count, Ge_nb_N_count, Ge_nb_O_count, Ge_nb_mean_EN, Ge_nb_max_EN, Shell2_N_count, Shell2_O_count, bond_Ge_C, bond_Ge_N, bond_Ge_O.
- `gaussian_summary/jobs_final_1183.csv`: per job: route, charge, multiplicity, status, number of normal terminations, SCF method label, SCF energy (Hartree, 4 decimals), <S2> before/after annihilation, zero-point correction, species (parent, frag1, frag2). Parents: restricted CAM-B3LYP, singlet; fragments: unrestricted CAM-B3LYP, doublet; `Opt Freq`, `scf=converge=12`.

## How to reproduce
Python 3.13 (3.10+ should work). Install: `pip install -r requirements.txt`. Run from the repository folder.
```bash
python Ge_X_BDE_revised.py        # about 5-8 min; writes outputs/ and outputs_Ge_X_BDE.zip   (QUICK=1 for a short smoke test)
python Ge_X_BDE_TPOT.py           # 1-2 h (120 min search); reads outputs/GE_BDE_descriptors.csv and outputs/split_assignments.csv; writes tpot_outputs/
python Ge_X_BDE_TPOT_extras.py    # about 1 min; reads the saved TPOT pipeline; writes tpot_extras/
python scan_gaussian_logs.py LOGS_FOLDER_OR_ZIP --dataset GE_BDE.csv   # optional; scan your own Gaussian logs
```
Seed is 42 everywhere; package versions are in `results/main/run_manifest.json` and `results/tpot/tpot_manifest.json`. The TPOT search is time limited, so repeating it can select a slightly different pipeline; the exported pipeline in `results/tpot/` reproduces the TPOT numbers reported in the article. Run on Google Colab by uploading the scripts and the two input CSVs (`GE_BDE.csv`, `Ge-Test-Dataset.csv`) and running `!python Ge_X_BDE_revised.py`.

## Validation protocol (summary)
All target-dependent steps (scaling and, in the optional automatic mode, descriptor filtering) are fitted inside training folds with scikit-learn pipelines. The external set is used once. Random 5-fold CV and fingerprint-cluster-grouped 5-fold CV (Morgan radius 2, 2048 bits; Butina cutoff 0.6; 175 clusters; GroupKFold) are reported. Also included: bond-type-mean and ridge baselines, ablations without bond-identity descriptors, split-conformal prediction intervals, SHAP (XGBoost) and t-SNE.

## Main results (R2 / MAE in kcal/mol; external set also MAPE %)
| Model | Random 5-fold CV | Cluster 5-fold CV | External test (n = 183) |
|---|---|---|---|
| XGBoost | 0.942 / 1.44 | 0.923 / 1.59 | 0.951 / 1.39 / 1.71 |
| Random Forest | 0.919 / 1.73 | 0.888 / 1.95 | 0.938 / 1.59 / 1.97 |
| Descriptor-based DNN | 0.908 / 1.78 | 0.859 / 2.17 | 0.939 / 1.51 / 1.87 |
| TPOT 1.1.0 | 0.940 / 1.51 (optimistic) | 0.926 / 1.62 (optimistic) | 0.956 / 1.37 / 1.67 |

TPOT CV values are optimistic because its pipeline was selected on the same training compounds.

## Where each article table/figure comes from
| Article item | File |
|---|---|
| Tables 3 and 4 (CV and external metrics) | `results/main/Table_model_performance.csv`, `results/tpot/TPOT_results.csv`, `results/tpot_extras/TPOT_cluster_CV.csv` |
| Table 5 (baselines and ablations) | `results/main/Table_model_performance.csv` (rows Bond-type mean, Ridge, XGBoost ablations) |
| Per bond-class metrics | `results/main/Table_per_bond_class_test.csv` |
| Conformal intervals and coverage | `results/main/Table_conformal_intervals.csv` |
| SHAP of XGBoost | `results/main/Table_SHAP_XGBoost.csv`, `Fig_SHAP_XGBoost.png` |
| External parity plots, MAE comparison | `results/main/Fig_parity_external_test.png`, `Fig_MAE_comparison.png`; with TPOT: `results/tpot_extras/*.png` |
| t-SNE (Figure 18) | `results/main/Fig_tSNE.png`, `tsne_*.csv` |
| Supplementary Tables S1-S4 | `supplementary/Supplementary_Tables_S1-S4_v3.xlsx` |

## Ge benchmark against experiment (Supplementary Table S1)
`benchmark/` holds Gaussian inputs and logs for GeH4, Me3GeH, Me4Ge and Me6Ge2 at CAM-B3LYP/LANL2DZ and CAM-B3LYP/6-31G(d,p), `experimental_reference.csv` (Luo, *Comprehensive Handbook of Chemical Bond Energies*, CRC Press, 2007) and `analyze_benchmark.py`. Run `cd benchmark && python analyze_benchmark.py` to regenerate the errors (MAE of the bond dissociation enthalpy at 298 K: 3.98 kcal/mol for LANL2DZ and 11.00 kcal/mol for 6-31G(d,p)). Compare the experimental D298 with DH298, not De.

## Limitations
De is a gas-phase electronic energy at one DFT level; agreement with experiment was checked only for four Ge molecules (Ge-C error +9.3 kcal/mol with LANL2DZ). The external set is distribution-matched, and the cluster-grouped CV is the more conservative estimate of transferability.

## Citation
Please cite the article and this archive (see `CITATION.cff`).

## Contact
Khurshid Ayub (khurshid@cuiatd.edu.pk) or Mahnoor Tajammal (tajammalmahnoor@gmail.com).
