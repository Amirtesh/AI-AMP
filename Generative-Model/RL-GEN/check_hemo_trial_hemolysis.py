import pandas as pd
import hemopi2_client as hc

df = pd.read_csv('grpo_hemo_trial_checkpoints/compounds_generated.csv')
recent = df[df['step'] >= 75]['sequence'].dropna().unique().tolist()
print(f'{len(recent)} unique sequences from step>=75')

# score in batches to avoid one giant MERCI call
batch_size = 200
all_scores = []
for i in range(0, len(recent), batch_size):
    batch = recent[i:i+batch_size]
    scores = hc.score_batch(batch)
    all_scores.extend(scores)
    print(f'scored {i+len(batch)}/{len(recent)}')

n_hemolytic = sum(1 for s in all_scores if s >= 0.55)
pct = 100 * n_hemolytic / len(all_scores)
print(f'\nHemolytic: {n_hemolytic}/{len(all_scores)} ({pct:.1f}%)')
print(f'Mean hemolysis score: {sum(all_scores)/len(all_scores):.3f}')

# compare directly to v4 (75.7%) and Case 3 (69.0%)
print(f'\nv4 baseline: 75.7% hemolytic')
print(f'Case 3 baseline: 69.0% hemolytic')
print(f'This run: {pct:.1f}% hemolytic')
