import pandas as pd
import glob

def merge_sequences(pattern, output_name):
    files = sorted(glob.glob(pattern))
    print(f"{output_name}: found {len(files)} files -> {files}")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df[["SEQUENCE"]])
    
    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset="SEQUENCE").reset_index(drop=True)
    after = len(combined)
    print(f"{output_name}: {before} rows -> {after} after dedup ({before - after} duplicates removed)")
    
    combined.to_csv(output_name, index=False)
    return combined

fungal = merge_sequences("f*.csv", "Antifungal.csv")
gram_pos = merge_sequences("gp*.csv", "Antigramp.csv")
gram_neg = merge_sequences("gn*.csv", "Antigramn.csv")

print("\nFinal counts:")
print(f"Antifungal: {len(fungal)}")
print(f"Antigramp: {len(gram_pos)}")
print(f"Antigramn: {len(gram_neg)}")
