from Bio import SeqIO
import pandas as pd
import os

def fasta_to_sequences(fasta_path):
    seqs = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]
    print(f"{fasta_path}: {len(seqs)} sequences parsed")
    return seqs

def merge_into_existing(fasta_path, csv_path):
    new_seqs = fasta_to_sequences(fasta_path)
    new_df = pd.DataFrame({"SEQUENCE": new_seqs})

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        before_existing = len(existing_df)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        print(f"{csv_path} not found yet - creating fresh from dbAMP only")
        before_existing = 0
        combined = new_df

    before = len(combined)
    combined = combined.drop_duplicates(subset="SEQUENCE").reset_index(drop=True)
    after = len(combined)
    print(f"{csv_path}: {before_existing} existing + {len(new_df)} from dbAMP "
          f"-> {before} total -> {after} after dedup ({before - after} cross-database duplicates removed)")

    combined.to_csv(csv_path, index=False)
    return combined

fungal = merge_into_existing("dbAMP_Antifungal_2024.fasta", "Antifungal.csv")
gram_pos = merge_into_existing("dbAMP_AntiGram_p_2024.fasta", "Antigramp.csv")
gram_neg = merge_into_existing("dbAMP_AntiGram_n_2024.fasta", "Antigramn.csv")

print("\nFinal counts after DBAASP + dbAMP merge:")
print(f"Antifungal: {len(fungal)}")
print(f"Antigramp: {len(gram_pos)}")
print(f"Antigramn: {len(gram_neg)}")
