import pandas as pd
import subprocess
import json
from sklearn.metrics import recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("external_results", exist_ok=True)
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_and_filter(csv_path, min_len=11, max_len=60):
    df = pd.read_csv(csv_path)
    df["sequence"] = df["sequence"].str.upper().str.strip()
    before = len(df)

    df = df[df["sequence"].apply(lambda s: len(s) > 0 and set(s) <= STANDARD_AA)]
    after_aa = len(df)

    df["seq_len"] = df["sequence"].str.len()
    df = df[(df["seq_len"] >= min_len) & (df["seq_len"] <= max_len)]
    after_len = len(df)

    df = df.drop_duplicates(subset="sequence")
    after_dedup = len(df)

    print(f"  {before} -> {after_aa} (standard AA only) -> {after_len} (length {min_len}-{max_len}) "
          f"-> {after_dedup} (deduped)")
    return df.reset_index(drop=True)

tasks_config = {
    "gram_positive": {"csv": "gp_apd3_novel.csv", "flag": "-gp"},
    "gram_negative": {"csv": "gn_apd3_novel.csv", "flag": "-gn"},
    "fungal": {"csv": "f_apd3_novel.csv", "flag": "-fungal"},
}

summary_rows = []

for task, cfg in tasks_config.items():
    print(f"\n=== {task} ===")
    df = clean_and_filter(cfg["csv"])

    fasta_path = f"external_results/{task}_filtered.fasta"
    with open(fasta_path, "w") as f:
        for i, row in df.iterrows():
            f.write(f">seq_{i}\n{row['sequence']}\n")

    output_csv = f"external_results/{task}_predictions_filtered.csv"
    cmd = ["python", "predict.py", "--fasta", fasta_path, cfg["flag"], "--output", output_csv]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        continue

    preds = pd.read_csv(output_csv)
    y_true = [1] * len(preds)
    y_pred = preds[f"{task}_pred"].tolist()
    sensitivity = recall_score(y_true, y_pred)

    print(f"Sensitivity: {sensitivity:.4f} ({sum(y_pred)}/{len(y_pred)})")
    summary_rows.append({
        "task": task, "n_filtered": len(preds),
        "n_correct": sum(y_pred), "sensitivity": sensitivity
    })

    thresh = json.load(open("model_artifacts/config.json"))[task]["threshold"]
    plt.figure(figsize=(7, 4))
    plt.hist(preds[f"{task}_prob"], bins=30, color="steelblue", edgecolor="black")
    plt.axvline(thresh, color="red", linestyle="--", label=f"threshold ({thresh})")
    plt.title(f"{task} - length/AA-filtered external set (N={len(preds)})")
    plt.xlabel("predicted probability"); plt.legend(); plt.tight_layout()
    plt.savefig(f"external_results/{task}_filtered_prob_dist.png", dpi=150)
    plt.close()

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("external_results/external_validation_summary_filtered.csv", index=False)
print("\n=== FILTERED EXTERNAL VALIDATION SUMMARY ===")
print(summary_df.to_string(index=False))

plt.figure(figsize=(6,4))
plt.bar(summary_df["task"], summary_df["sensitivity"], color=["#4C72B0","#DD8452","#55A868"])
plt.ylim(0,1); plt.ylabel("Sensitivity")
for i, v in enumerate(summary_df["sensitivity"]):
    plt.text(i, v+0.02, f"{v:.3f}", ha="center")
plt.title("External sensitivity, length/AA-matched to training domain")
plt.tight_layout()
plt.savefig("external_results/sensitivity_comparison_filtered.png", dpi=150)
plt.close()
