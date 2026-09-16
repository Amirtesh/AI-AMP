# AI-AMP Comprehensive Scientific Tutorial and Methodological Reference

Author: Amirtesh Raghuram  
Project: AI-AMP Design Suite  
License: MIT Open Source License  

---

## 1. Overview and Architecture

AI-AMP is an integrated computational pipeline for the discovery, classification, and generative exploration of antimicrobial peptides (AMPs). The system consists of three modular components:

1. Model 2 (Binary Gatekeeper): Filters prospective sequences into active AMPs versus non-antimicrobial background proteins.
2. Model 1 (Spectrum Classifier): Multi-label classifier predicting targeted antimicrobial spectrum (Gram-positive, Gram-negative, antifungal) directly from primary sequence.
3. Model 3 (Generative Engine - SmallPeptideGPT): Autoregressive 3-stage peptide generator:
   - Stage 1: Unsupervised peptide grammar pretraining across 9 proteomes (3.37M sequences).
   - Stage 2: Supervised fine-tuning on 18,878 verified AMPs (15-50aa). This is the validated generative deliverable.
   - Stage 3: Experimental Group Relative Policy Optimization (GRPO) reinforcement learning with multi-objective reward shaping and in-process toxicity scoring.

### End-to-End Workflow

```
[Candidate Sequences / De Novo Generations]
                    |
                    v
      +----------------------------+
      | Model 2: Binary Gatekeeper |
      | (ESM2-35M / XGBoost)       |
      +----------------------------+
         |                      |
  [Predicted Non-AMP]     [Predicted AMP]
         |                      |
         v                      v
   Filtered Out   +-------------------------------------+
                  | Model 1: Spectrum Classifier        |
                  | - Gram-Positive (Biophysical Head)  |
                  | - Gram-Negative (ESM2-35M Head)     |
                  | - Antifungal (Biophysical Head)     |
                  +-------------------------------------+
                              |
                              v
          [Multi-Label Spectrum Predictions: GP, GN, F]
```

---

## 2. Model 1: Antimicrobial Spectrum Classification

Model 1 predicts the target antimicrobial spectrum of active peptides across three non-mutually exclusive categories: Gram-positive bacteria, Gram-negative bacteria, and fungi.

### 2.1 Dataset Curation and Preprocessing

- Sources: DBAASP (API pulls and category exports), dbAMP (FASTA exports), and DRAMP (annotated spreadsheets).
- Sequence Deduplication: Merged and deduplicated across all three repositories by exact amino acid sequence. A natural 72% cross-database duplication rate was observed and accounted for.
- Label Distribution: Each sequence is assigned non-mutually exclusive labels: `gram_positive`, `gram_negative`, and `fungal`. In the merged dataset, 71.1% of peptides possess at least two active labels. Specifically, 89.6% of Gram-positive peptides also carry a Gram-negative label, reflecting true broad-spectrum biological activity common to many natural bacteriocins.
- Length Window: Sequences are filtered strictly between 11 and 60 amino acids. The lower bound ensures word-size compatibility (n=3) with CD-HIT clustering at 50% sequence identity. The upper bound removes non-peptide protein outliers.
- Final Curated Size: 17,323 merged entries yielded 14,817 unique, length-conforming, canonical amino acid sequences.

### 2.2 Leakage-Free Splitting Discipline

To prevent data leakage across homologues, clustering was performed using CD-HIT:
- Parameters: 50% sequence identity (`-c 0.5 -n 3 -M 4000 -T 1 -d 0`).
- Determinism Control: Input FASTA files were pre-sorted by sequence length in descending order, sequence IDs were generated from 12-character MD5 hashes of upper-case sequences, and single-threaded execution was enforced to guarantee bitwise reproducibility.
- Partitions: The 2,077 resulting sequence clusters were partitioned at the cluster level into:
  - Training split: 80% (11,703 sequences)
  - Validation split: 10% (1,645 sequences)
  - Test split: 10% (1,469 sequences)
- Canonical Locked File: The dataset split is frozen in `AMP_features_SPLIT_FINAL.csv`.

### 2.3 Feature Engineering and Three-Way Ablation

Two distinct feature modalities were computed:
1. Biophysical Descriptors (34 features):
   - Amino acid composition fractions (20 canonical residues).
   - GRAVY grand average of hydropathy.
   - Secondary structure fractions (alpha-helix, beta-turn, beta-sheet) computed via Biopython ProtParam.
   - modlAMP GlobalDescriptor metrics: sequence length, molecular weight, net charge, charge density, isoelectric point, instability index, aromaticity, aliphatic index, Boman index, and hydrophobic ratio.
2. ESM2 Latent Representations (480 dimensions):
   - Extracted from `esm2_t12_35M_UR50D` (layer 12 representations, mean-pooled over all residue positions, excluding BOS and EOS tokens).

For each spectrum target, a 3-way ablation (Biophysical vs. ESM2 vs. Combined) was conducted with native XGBoost classifiers. Hyperparameters were tuned across 60 Optuna trials maximizing validation Matthews Correlation Coefficient (MCC).

Ablation Results across Feature Sets:

| Feature Set | Gram-Positive (Val MCC / Test AUROC) | Gram-Negative (Val MCC / Test AUROC) | Antifungal (Val MCC / Test AUROC) |
|---|---|---|---|
| Biophysical (34) | 0.5565 / 0.8102 | 0.5407 / 0.8609 | 0.5156 / 0.8363 |
| ESM2 (480) | 0.5362 / 0.8065 | 0.5569 / 0.8825 | 0.4398 / 0.7993 |
| Combined (514) | 0.5352 / 0.8072 | 0.5245 / 0.8771 | 0.4522 / 0.8195 |

Final Architecture Selection:
- Gram-Positive Head: Biophysical features selected (highest validation MCC).
- Gram-Negative Head: ESM2 representations selected (ESM2 test AUROC 0.883 [0.861, 0.904] significantly outperformed biophysical 0.861 [0.839, 0.882]).
- Antifungal Head: Biophysical features selected (Biophysical test AUROC 0.836 [0.815, 0.854] significantly outperformed ESM2 0.799 [0.776, 0.820]).

### 2.4 Final Benchmark Performance on Locked Test Set

Evaluated once on the held-out test partition (1,469 sequences):

| Target Spectrum | Selected Representation | Decision Threshold | Test AUROC | Test AUPRC | Test MCC | Test F1 |
|---|---|---|---|---|---|---|
| Gram-Positive | Biophysical (34) | 0.50 | 0.8102 | 0.9070 | 0.4264 | 0.8583 |
| Gram-Negative | ESM2 (480) | 0.52 | 0.8825 | 0.9493 | 0.6307 | 0.9100 |
| Antifungal | Biophysical (34) | 0.48 | 0.8363 | 0.8364 | 0.5145 | 0.7573 |

### 2.5 Feature Interpretability via SHAP

SHAP TreeExplainer values were computed for the final models:
- Gram-Positive Model: Primary importance was assigned to length/molecular weight (`modlamp_Length`, `modlamp_MW`), glycine fraction (`frac_G`, conferring conformational flexibility for bilayer penetration), and sheet fraction (`ss_sheet_frac`).
- Antifungal Model: Dominated by charge density (`modlamp_ChargeDensity`), tryptophan frequency (`frac_W`, facilitating interfacial anchoring into ergosterol-containing fungal membranes), and length.
- Gram-Negative Model: Driven by latent ESM2 embedding dimensions. Post-hoc correlation probes showed that 7 of the top 15 SHAP-ranked ESM2 dimensions exhibited strong correlation (|r| > 0.50) with known biophysical descriptors (net charge, hydrophobicity, and molecular weight), confirming biological grounding in the latent features.

### 2.6 External Validation on Novel APD3 Sequences

Model 1 was validated against the Antimicrobial Peptide Database (APD3). Sequences present in the training set were identified and purged:

| Target Spectrum | APD3 Total | Training Overlap | Novel Sequences | Length-Matched (11-60aa) | Measured Sensitivity |
|---|---|---|---|---|---|
| Gram-Positive | 933 | 76.3% | 221 | 149 | 0.8725 (130 / 149) |
| Gram-Negative | 1,171 | 48.1% | 608 | 513 | 0.9532 (489 / 513) |
| Antifungal | 1,828 | 78.3% | 397 | 277 | 0.5596 (155 / 277) |

Analysis of the Antifungal Sensitivity Gap:  
Investigation of false negatives on the novel APD3 fungal set revealed they are heavily concentrated in synthetic point-mutant analog series (e.g., temporin derivatives) whose primary literature activity is antibacterial, with antifungal activity reported at marginal or unverified potency. This discrepancy reflects annotation strictness divergence between APD3 and the training databases rather than algorithmic discrimination failure.

---

## 3. Model 2: Binary AMP vs. Non-AMP Gatekeeper

Model 2 acts as an upstream gatekeeper separating bona fide antimicrobial peptides from non-antimicrobial background proteins.

### 3.1 Positive Dataset Curation

- Sources: DBAASP monomer records (`all_active` set, confirmed active across all tested targets), dbAMP, and DRAMP `general_amps`.
- Filtering: Deduplicated by upper-case sequence; non-canonical amino acids removed (10.08% of raw sequences, representing cyclic/branched notations or ambiguous X residues).
- Length Range: 5 to 100 amino acids (derived from literature consensus across ampir, HMAMP, and OmegAMP).
- Positive Count: 37,638 raw entries yielding 34,484 clean positive sequences.

### 3.2 Negative Dataset Curation and Length Confounder Resolution

Naïve negative background extraction from Swiss-Prot causes an extreme length mismatch: natural non-antimicrobial proteins have a median length of 75aa (IQR 57-89), while AMPs have a median length of 20aa (IQR 13-35). Uncorrected models learn trivial length boundaries instead of antimicrobial biochemistry.

Resolution Protocol:
1. Tier 1 Assayed Negatives (Held Out): 585 DBAASP peptides confirmed inactive across all tested target species (`1111111.0` sentinel float value). Excluded entirely from training and reserved for hard-negative stress testing. An additional 2,361 mixed-spectrum peptides (active on some targets, inactive on others) were excluded as ambiguous.
2. Tier 2 Swiss-Prot Background: 564,751 reviewed entries filtered via expanded GO-terms and keywords (`Antimicrobial`, `Antibiotic`, `Fungicide`, `Antiviral protein`, `Amphibian defense peptide`, `Plant defense`, `Bacteriocin`, `Lantibiotic`).
3. Exact Length Matching & Synthetic Fragmentation: Where sufficient natural proteins existed at a given length, direct sampling was used (11,963 sequences). For short lengths (5-30aa), surplus long background proteins were sliced into synthetic non-antimicrobial fragments (22,345 sequences). The resulting negative length distribution (mean 27.3aa, IQR 13-35) matches the positive distribution (mean 27.2aa, IQR 13-35).
4. Outlier Removal Rejection: Statistical IQR outlier removal was tested and rejected. It disproportionately removed positives over negatives by 3.3 to 1 (974 positives vs. 298 negatives), driven by high `instability_index` values that directly reflect the high charge density required for antimicrobial activity.

Final Dataset: 68,792 sequences (34,484 positive, 34,308 negative). Partitioned via CD-HIT at 50% sequence identity (`-c 0.5 -n 3 -l 4 -T 1 -d 0`) across 9,204 clusters into an 80/10/10 split (54,884 train, 7,654 val, 6,254 test).

### 3.3 Three-Way Ablation and Model Selection

| Feature Set | Test AUROC | Test AUPRC | Test MCC | Test F1 |
|---|---|---|---|---|
| Biophysical (34) | 0.9317 | 0.9308 | 0.7136 | 0.8472 |
| ESM2 (480) | 0.9470 | 0.9507 | 0.7603 | 0.8695 |
| Combined (514) | 0.9481 | 0.9517 | 0.7618 | 0.8716 |

Selection Rationale:  
A 1,000-fold bootstrap test on the MCC difference between Combined and ESM2-only gave a mean difference of 0.0014 with a 95% confidence interval of [-0.0088, 0.0115]. Because the confidence interval straddles zero, the two models are statistically indistinguishable. ESM2-only was selected on a tie-break to eliminate runtime dependencies on biophysical feature calculation libraries during high-throughput inference.

### 3.4 External Validation on Independent Corpus (LMPred)

Evaluated on 1,251 non-overlapping, length-matched sequences from the independent LMPred benchmark:
- External Test AUROC: 0.9123 (95% bootstrap CI: [0.8722, 0.9483]).

### 3.5 Hard-Negative Stress Test and Failure Mode Diagnosis

Model 2 was stress-tested against 116 experimentally confirmed inactive DBAASP Tier 1 peptides paired with 234 clean, non-overlapping APD6 natural positive peptides:
- Measured AUROC: 0.655 to 0.675 across all feature representations (ESM2 = 0.6751).
- Error Pattern: Sensitivity remained solid (0.77 to 0.79), while specificity collapsed to chance (0.51 to 0.56).
- False Positive Mechanism: Inactive candidate peptides were classified as positive with high confidence (median predicted probability 0.95). These peptides were synthesized specifically as AMP candidates and share cationic/amphipathic features, exposing an intrinsic blind spot in models trained against generic background proteins.
- Identified Boundary Weaknesses:
  1. Sub-15aa Sequences: Sensitivity drops to 31% in the 5-10aa range due to token averaging in ESM2 mean-pooling.
  2. Large (60-100aa) Neutral/Anionic Peptides: Model enforces a learned "AMP = cationic" heuristic, failing on non-canonical anionic AMPs.

The inference script `Model2-AMP_Predictor/predict.py` emits an explicit runtime low-confidence warning for any input sequence under 15 amino acids.

---

## 4. Model 3: Generative Engine (SmallPeptideGPT & GRPO)

Model 3 is an autoregressive generative framework designed for the de novo exploration of functional AMP candidates.

### 4.1 Transformer Architecture: SmallPeptideGPT

SmallPeptideGPT was constructed specifically for peptide sequences:
- Parameters: 4.77 million parameters.
- Architecture: 6 causal transformer layers, hidden dimension d_model = 256, 8 attention heads, GELU non-linearities, layer normalization, dropout = 0.10.
- Vocabulary: 23 tokens (20 canonical amino acids plus `<PAD>`, `<BOS>`, `<EOS>`).
- Context Length: 85 tokens (max sequence length 83 + BOS + EOS).

### 4.2 Stage 1: Pretraining on PeptideAtlas

- Corpus: 3.37 million sequences compiled from PeptideAtlas across 9 diverse organisms (Human, Mouse, Yeast, C. elegans, Drosophila, Zebrafish, M. tuberculosis, Pig, Cow).
- Exponential Reweighting: Sequences were deduplicated within-organism, and natural sampling probabilities were smoothed via alpha=0.5 exponentiation (`p_smooth = p ** 0.5`), balancing human representation (~75% raw pool) against other organisms.
- Optimization: 15 epochs, batch size 1536, AdamW, OneCycleLR learning rate schedule peaking at 6e-4, bf16 mixed precision.
- Result: Validation cross-entropy loss converged from 2.77 to 2.44, successfully acquiring general peptide sequence grammar.

### 4.3 Stage 2: Supervised Fine-Tuning (Working Generative Deliverable)

- Corpus: 18,878 curated AMP sequences (15-50aa) from `amp_positive_filtered.csv`.
- Optimization: Initialized from Stage 1 weights. Positional embedding table size preserved at 83 residues. Trained for 20 epochs at learning rate 5e-5 (AdamW, batch size 512).
- Convergence: Validation loss improved from 2.90 to 2.10.
- Structural Validation:
  - Real-vs-scrambled entropy gap: +1.70 to +2.65 nat/token, proving true structural learning over surface memorization.
  - Low memorization: Only 1 in 20 generated samples matched training data.
  - Structural diversity: Preserves multiple distinct topologies, including cysteine-rich defensin-like and proline-rich motifs.

Stage 2 (`Generative-Model/model3_generative/stage2_checkpoints/best_model.pt`) is the verified, working generative model.

### 4.4 Stage 3: GRPO Reinforcement Learning

Policy updates were implemented using Group Relative Policy Optimization (GRPO):
- Group Sampling: At each step, a group of G = 256 candidates is generated from the current policy.
- Relative Advantage: Rewards are normalized across the group via z-scoring:
  ```
  Advantage_i = (Reward_i - Group_Mean_Reward) / (Group_Std_Reward + 1e-6)
  ```
- Policy Loss: Updates combine advantage-weighted log-probabilities with a per-token KL divergence penalty against the frozen Stage 2 reference policy:
  ```
  Token_KL = exp(log_ratio) - log_ratio - 1
  Loss = - Mean(Advantage * Seq_Logprob) + kl_coef * Mean(Token_KL)
  ```

### 4.5 Composite Multi-Objective Reward Function

The reward function scores candidate sequences across eight terms:

```
Total_Reward = (
    w_model2 * Score_Model2
    + w_charge * Score_Charge
    + w_length * Score_Length
    + w_hmom * Score_Hydrophobic_Moment
    + w_div * Score_Diversity_Penalty
    + w_class * Score_Class_Diversity
    + w_hemo * Score_Hemolysis_Penalty
    + w_tox * Score_Toxicity_Penalty
)
```

Reward Component Specifications:
1. `Score_Model2`: Model 2 gatekeeper probability in [0, 1] (Default weight = 1.00).
2. `Score_Charge`: Net charge soft-banded in [+2.0, +9.0] with 2.0 margin (Default weight = 0.30).
3. `Score_Length`: Gaussian length shaping centered at 26 residues with standard deviation 8:
   ```
   Score_Length = exp(-0.5 * ((Length - 26) / 8)^2)
   ```
   (Default weight = 0.20).
4. `Score_Hydrophobic_Moment`: Eisenberg hydrophobic moment normalized to the 90th percentile of real AMPs (0.65):
   ```
   Score_Hydrophobic_Moment = min(Moment / 0.65, 1.0)
   ```
   (Default weight = 0.20).
5. `Score_Diversity_Penalty`: Negative penalty based on the maximum dipeptide cosine similarity to any peer in the same generated group:
   ```
   Penalty = max(0.0, (Max_Similarity - 0.70) / 0.30)
   Score_Diversity_Penalty = -Penalty
   ```
   (Default weight = 0.30).
6. `Score_Class_Diversity`: Bonus combining underrepresentation of the sequence's ESM2 KMeans (k=5) cluster in natural AMPs with a running visitation count across the current training run (Default weight = 0.25).
7. `Score_Hemolysis_Penalty`: In-process HemoPI2 Hybrid2 scoring:
   ```
   Score_Hemolysis_Penalty = 1.0 - HemoPI2_Hybrid2_Score
   ```
   (Default weight = 0.40).
8. `Score_Toxicity_Penalty`: In-process ToxinPred3 Hybrid scoring via HTTP:
   ```
   Score_Toxicity_Penalty = 1.0 - ToxinPred3_Hybrid_Score
   ```
   (Default weight = 0.40).
9. `Score_AD_Penalty`: Applicability-domain distance penalty (Weight = 0.00; tested and disabled as a documented negative result).

### 4.6 Running GRPO Experiments

The GRPO reinforcement learning infrastructure is fully implemented and available for user experimentation. The reward function is configurable at the CLI level, allowing systematic exploration of multi-objective AMP optimization.

To run a GRPO experiment:

1. Start the ToxinPred3 server in the `toxin_hemo` environment (see README.md Section 3.4).
2. Activate the `RLGEN` environment and navigate to `Generative-Model/model3_grpo/`.
3. Run unit tests to verify reward components are functioning:
   ```bash
   python3 test_reward_components.py
   ```
4. Launch training:
   ```bash
   python3 grpo_train.py \
       --init_checkpoint ../model3_generative/stage2_checkpoints/best_model.pt \
       --checkpoint_dir grpo_checkpoints \
       --group_size 256 \
       --total_steps 300 \
       --min_len 15 \
       --max_len 50 \
       --lr 1e-5 \
       --kl_coef 0.05
   ```

All eight reward component weights (`--w_model2`, `--w_charge`, `--w_hmom`, `--w_diversity_penalty`, `--w_class_diversity`, `--w_hemolysis_penalty`, `--w_toxicity_penalty`) are individually overridable via CLI flags. Diagnostic and analysis tools are provided in `model3_grpo/`: `compare_checkpoints.py` for entropy and diversity tracking, `plot_grpo_progress.py` for reward trajectory visualization, and `track_compounds.py` for novel sequence extraction.

Mode collapse is a documented challenge in RL-based peptide generation and an active area of research. The infrastructure here provides a starting point for systematic multi-objective reward shaping experiments.

---

## 5. Reusable Infrastructure and Software Modules

The codebase provides several standalone, validated tools:

1. `Model1-AMP_Class_Classifier/predict.py`: Standalone CLI supporting single sequences, FASTA, CSV, and TXT files, predicting Gram-positive, Gram-negative, and antifungal activity with native XGBoost JSON models.
2. `Model2-AMP_Predictor/predict.py`: Standalone binary gatekeeper CLI. Keeps invalid rows in output tables with explicit error reasons, and flags sub-15aa sequences with confidence notes.
3. `Generative-Model/model3_generative/generate_finetuned.py`: Nucleus/temperature-controlled autoregressive generator sampling from the validated Stage 2 SmallPeptideGPT model.
4. `Generative-Model/model3_grpo/hemopi2_client.py`: In-process Python wrapper for HemoPI2 Hybrid2 (ESM + MERCI) scoring (<0.5s per 256-sequence batch).
5. `Generative-Model/model3_grpo/toxinpred3_server.py` and `toxinpred3_client.py`: Local microservice architecture isolating scikit-learn 1.2.2 requirements from modern Python environments.
6. `Generative-Model/model3_grpo/test_reward_components.py`: Unit test suite verifying physicochemical and diversity reward calculations without requiring GPU or server resources.

---

## 6. Open Items and Limitations

1. Out-of-Scope Characteristics: Model 1 predicts spectrum category labels. It does not model source organism compatibility, food-matrix stability (pH, salt, thermal resilience), or mammalian toxicity on its own.
2. Gram-Negative False Positives: Internal test matrices showed a ~50% false positive rate at the standard decision threshold. An audit against empirical MIC assays is planned for future work.
3. Group-Quota Sampling: Implementing structural family quotas during generation is identified as the required architectural mechanism to resolve GRPO mode collapse.
