# AI-AMP: Machine Learning Suite for Antimicrobial Peptide Classification, Gating, and Generation

Author: Amirtesh Raghuram  
License: MIT Open Source License  

---

## 1. Overview and Core Modules

AI-AMP is an open-source suite of machine learning models and pipelines for antimicrobial peptide (AMP) research. The codebase is organized into three primary modules:

1. **Model 1 (Antimicrobial Spectrum Classifier):** A multi-label classifier predicting targeted antimicrobial spectrum (Gram-positive, Gram-negative, antifungal) directly from primary peptide sequence.
2. **Model 2 (Binary AMP Gatekeeper):** An upstream binary classifier distinguishing genuine AMPs from non-antimicrobial background proteins, utilizing exact length-matched background curation and ESM2 language model representations. Externally validated and benchmarked head-to-head against AMPir and AMPEPpy on the LMPred corpus (AUROC 0.9123).
3. **Model 3 (Generative Engine - SmallPeptideGPT):** An autoregressive peptide generator consisting of Stage 1 pretraining (3.37M natural peptides), Stage 2 supervised fine-tuning (18,878 verified AMPs), and Stage 3 Group Relative Policy Optimization (GRPO) reinforcement learning infrastructure.

For the complete scientific methodology, benchmark tables, ablation studies, SHAP interpretability analyses, and detailed failure-mode diagnoses, please refer to [TUTORIAL.md](TUTORIAL.md).

---

## 2. Repository Layout

```
Final-Codes/
|-- LICENSE                                    # MIT open source license
|-- README.md                                  # Setup guide and quickstart
|-- TUTORIAL.md                                # Comprehensive scientific tutorial
|
|-- Model1-AMP_Class_Classifier/               # Spectrum classifier (GP, GN, Fungal)
|   |-- env.yml                                # Conda environment file
|   |-- predict.py                             # Spectrum prediction CLI
|   |-- model_artifacts/                       # Native XGBoost models and config
|   |-- External-Validation-APD3/              # APD3 external validation suite
|   `-- SHAP/                                  # SHAP interpretability scripts
|
|-- Model2-AMP_Predictor/                      # Binary AMP/non-AMP gatekeeper
|   |-- env.yml                                # Conda environment file
|   |-- predict.py                             # Binary prediction CLI
|   |-- model2_esm2.joblib                     # Selected production model
|   |-- feature_columns.json                   # Feature column definitions
|   `-- Benchmark-External-Dataset/            # Head-to-head comparison vs AMPir, AMPEPpy (LMPred dataset)
|
`-- Generative-Model/                          # SmallPeptideGPT generator & RL
    |-- data/                                  # Shared training and reference data
    |   |-- amp_positive_filtered.csv          # Curated AMP positives for Stage 2 / GRPO
    |   |-- stage1_corpus_weighted.csv         # PeptideAtlas pretraining corpus
    |   |-- training_seq.csv                   # Merged sequence list for novelty checks
    |   `-- amp_clusters_assignments.csv       # AMP cluster assignments (k=5)
    |-- model2/                                # Model 2 client and ESM2 embedder
    |   |-- model2_esm2.joblib                 # Model 2 ESM2-only classifier
    |   |-- model2_client.py                   # Inference client for GRPO reward
    |   |-- esm2_embedder.py                   # Shared ESM2 embedding utility
    |   `-- feature_columns.json               # ESM2 feature column definitions
    |-- model3_analysis/                       # AMP family clustering tools
    |   |-- cluster_amp_families.py            # ESM2 k-means clustering script
    |   |-- amp_clusters_kmeans_k5.joblib      # Fitted k=5 cluster model
    |   `-- amp_clusters_k_selection.csv       # Silhouette scores across k values
    |-- model3_generative/                     # Stages 1 & 2: Pretraining and SFT
    |   |-- env.yml                            # Conda environment for Stages 1 & 2
    |   |-- train_stage1.py                    # Stage 1 pretraining script
    |   |-- train_stage2.py                    # Stage 2 SFT script
    |   |-- generate_base.py                   # Sampling from Stage 1 model
    |   |-- generate_finetuned.py              # Sampling from Stage 2 (validated deliverable)
    |   |-- plot_training_curves.py            # Loss and perplexity visualization
    |   |-- stage1_checkpoints/                # Stage 1 pretrained weights (best_model.pt)
    |   `-- stage2_checkpoints/                # Stage 2 production weights (best_model.pt)
    `-- model3_grpo/                           # Stage 3: GRPO reinforcement learning
        |-- env.yml                            # Conda environment for RLGEN
        |-- grpo_train.py                      # Multi-objective GRPO training engine
        |-- reward.py                          # Composite reward module
        |-- peptide_features.py                # Physicochemical scoring functions
        |-- diversity_penalty.py               # Dipeptide cosine diversity penalty
        |-- class_diversity.py                 # ESM2 cluster diversity bonus
        |-- hemopi2_client.py                  # In-process HemoPI2 Hybrid2 scoring client
        |-- toxinpred3_server.py               # Background ToxinPred3 HTTP service
        |-- toxinpred3_client.py               # ToxinPred3 HTTP client
        |-- test_reward_components.py          # Standalone reward unit tests
        |-- compare_checkpoints.py             # Checkpoint diversity and entropy analysis
        |-- plot_grpo_progress.py              # Reward trajectory visualization
        |-- track_compounds.py                 # Novel sequence extraction tool
        `-- env_requirements/                  # Requirements files and version notes
            |-- README.md                      # Dependency conflict notes
            |-- requirements_RLGEN.txt         # Pip freeze for RLGEN (Python 3.12)
            |-- requirements_toxin_hemo.txt    # Pip freeze for toxin_hemo (Python 3.10)
            |-- env_RLGEN.yml                  # Conda YAML for RLGEN
            `-- env_toxin_hemo.yml             # Conda YAML for toxin_hemo
```

---

## 3. Environment Setup and Installation

Due to an upstream dependency requirement in ToxinPred3 (its bundled model pickle requires `scikit-learn==1.2.2`, whereas modern PyTorch and language model libraries require `scikit-learn>=1.3.1`), isolated conda environments are used.

### Summary of Environments

| Environment Name | Module | Python | Primary Method |
|---|---|---|---|
| `amp_model1` | Model 1 Spectrum Classifier | 3.11 | `conda env create -f Model1-AMP_Class_Classifier/env.yml` |
| `amp_model2` | Model 2 Binary Gatekeeper | 3.11 | `conda env create -f Model2-AMP_Predictor/env.yml` |
| `amp_generative` | Model 3 Stages 1 & 2 | 3.11 | `conda env create -f Generative-Model/model3_generative/env.yml` |
| `RLGEN` | Model 3 Stage 3 GRPO Training | 3.12 | `pip install -r requirements_RLGEN.txt` |
| `toxin_hemo` | Model 3 ToxinPred3 HTTP Server | 3.10 | `pip install -r requirements_toxin_hemo.txt` |

---

### 3.1 Setting Up Model 1 (`amp_model1`)

From the repository root:

```bash
cd Model1-AMP_Class_Classifier
conda env create -f env.yml
conda activate amp_model1
```

Alternatively, set up manually:

```bash
conda create -n amp_model1 python=3.11 -y
conda activate amp_model1
conda install -c conda-forge -c bioconda numpy pandas scipy scikit-learn xgboost biopython matplotlib seaborn tqdm optuna shap openpyxl requests cd-hit -y
pip install torch fair-esm modlamp
```

---

### 3.2 Setting Up Model 2 (`amp_model2`)

From the repository root:

```bash
cd Model2-AMP_Predictor
conda env create -f env.yml
conda activate amp_model2
```

Alternatively, set up manually:

```bash
conda create -n amp_model2 python=3.11 -y
conda activate amp_model2
conda install -c conda-forge -c bioconda numpy pandas scipy scikit-learn xgboost joblib biopython matplotlib seaborn tqdm optuna shap openpyxl requests cd-hit -y
pip install torch fair-esm modlamp
```

---

### 3.3 Setting Up Generative Model Stages 1 & 2 (`amp_generative`)

For pretraining, fine-tuning, and sampling from the production Stage 2 checkpoint:

```bash
cd Generative-Model/model3_generative
conda env create -f env.yml
conda activate amp_generative
```

Alternatively, set up manually:

```bash
conda create -n amp_generative python=3.11 -y
conda activate amp_generative
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib tqdm -y
pip install torch
```

---

### 3.4 Setting Up Generative Model Stage 3 (GRPO & Toxicity Suite)

Stage 3 reinforcement learning uses two cooperating environments:
1. `toxin_hemo` — hosts the persistent ToxinPred3 scoring server (sklearn 1.2.2 isolated environment).
2. `RLGEN` — runs the GRPO training loop, Model 2 scoring, and HemoPI2 client (sklearn 1.3.1).

#### Step 1: Create the `toxin_hemo` Environment Using Requirements File

```bash
cd Generative-Model/model3_grpo/env_requirements
conda create -n toxin_hemo python=3.10 -y
conda activate toxin_hemo
pip install -r requirements_toxin_hemo.txt
```

*(Alternatively, run `conda env create -f env_toxin_hemo.yml`)*

#### Step 2: Create the `RLGEN` Environment Using Requirements File

```bash
cd Generative-Model/model3_grpo/env_requirements
conda create -n RLGEN python=3.12 -y
conda activate RLGEN
pip install -r requirements_RLGEN.txt
```

*(Alternatively, run `conda env create -f env_RLGEN.yml`)*

#### Step 3: Start the ToxinPred3 Scoring Server

In a background terminal:

```bash
conda activate toxin_hemo
cd Generative-Model/model3_grpo
nohup python3 toxinpred3_server.py --port 8765 > toxinpred3_server.log 2>&1 &
disown
```

Verify that the server is active:

```bash
curl http://127.0.0.1:8765/health
```

Expected response:
```json
{"status":"ok"}
```

---

## 4. Quickstart and CLI Usage

### 4.1 Model 1: Spectrum Prediction

Predict Gram-positive, Gram-negative, and antifungal activity from sequence inputs:

```bash
conda activate amp_model1
cd Model1-AMP_Class_Classifier

# Single sequence prediction across all three spectrum classes
python3 predict.py --sequence GLFDIVKKVVGALGSL -all

# Batch prediction from a FASTA file
python3 predict.py --fasta candidates.fasta -gp -fungal --output spectrum_results.csv

# Batch prediction from a CSV file (specify the sequence column)
python3 predict.py --csv candidates.csv --column sequence -all --device cpu
```

Flags:
- `-gp`: Predict Gram-positive activity.
- `-gn`: Predict Gram-negative activity.
- `-fungal`: Predict antifungal activity.
- `-all`: Predict all three targets.
- `--device`: ESM2 computation device (`cuda` or `cpu`).

---

### 4.2 Model 2: Binary AMP Gating

Filter sequences to verify antimicrobial activity before downstream tasks:

```bash
conda activate amp_model2
cd Model2-AMP_Predictor

# Predict on a single sequence
python3 predict.py --sequence GLFDIVKKVVGALGSL

# Predict on a CSV dataset
python3 predict.py --input candidates.csv --column sequence --output gatekeeper_results.csv

# Predict on a text file (one sequence per line)
python3 predict.py --input test_peptides.txt --output gatekeeper_results.csv
```

Notes on output:
- Invalid entries (non-canonical amino acids, or length outside 5-100aa) are preserved in the output table with an `invalid_reason` column.
- Valid sequences under 15 amino acids receive a `confidence_note` highlighting observed sensitivity limitations on very short sequences.

---

### 4.3 Model 3: De Novo Sampling (Stage 2 Production Model)

Generate novel AMP candidates using the validated Stage 2 model:

```bash
conda activate amp_generative
cd Generative-Model/model3_generative

python3 generate_finetuned.py \
    --nseq 100 \
    --min 15 \
    --max 50 \
    --temperature 1.0 \
    --top_p 0.9 \
    --training_seq ../data/training_seq.csv \
    --output novel_amp_candidates.csv
```

---

### 4.4 Model 3: Reinforcement Learning (Stage 3 GRPO)

Ensure the ToxinPred3 server is running first (`curl http://127.0.0.1:8765/health`), then:

```bash
conda activate RLGEN
cd Generative-Model/model3_grpo

# Run unit tests on reward calculations before training
python3 test_reward_components.py

# Launch GRPO training with default weights
python3 grpo_train.py \
    --init_checkpoint ../model3_generative/stage2_checkpoints/best_model.pt \
    --checkpoint_dir grpo_checkpoints \
    --group_size 256 \
    --total_steps 300 \
    --min_len 15 \
    --max_len 50 \
    --lr 1e-5 \
    --kl_coef 0.05

# Override individual reward weights via CLI flags
python3 grpo_train.py \
    --w_model2 1.0 \
    --w_charge 0.1 \
    --w_hmom 0.1 \
    --w_diversity_penalty 0.5 \
    --w_class_diversity 0.4 \
    --w_hemolysis_penalty 0.4 \
    --w_toxicity_penalty 0.4
```

See [TUTORIAL.md](TUTORIAL.md#4-model-3-generative-engine-smallpeptidegpt--grpo) for the reward function specification and GRPO methodology.

---

## 5. Citation

If you use AI-AMP models, feature pipelines, or diagnostic tooling in your research, please cite:

```bibtex

```

---

## 6. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
