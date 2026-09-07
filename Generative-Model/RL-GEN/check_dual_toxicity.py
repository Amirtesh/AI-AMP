import pandas as pd
import hemopi2_client as hc
import toxinpred3_client as tc

df = pd.read_csv('grpo_dual_toxicity_trial_checkpoints/compounds_generated.csv')
recent = df[df['step'] >= 75]['sequence'].dropna().unique().tolist()
print(f'{len(recent)} unique sequences from step>=75')

batch_size = 200
hemo_scores, tox_scores = [], []
for i in range(0, len(recent), batch_size):
    batch = recent[i:i+batch_size]
    hemo_scores.extend(hc.score_batch(batch))
    tox_scores.extend(tc.score_batch(batch))
    print(f'scored {i+len(batch)}/{len(recent)}')

n_hemolytic = sum(1 for s in hemo_scores if s >= 0.55)
n_toxic = sum(1 for s in tox_scores if s >= 0.38)
n_either = sum(1 for h, t in zip(hemo_scores, tox_scores) if h >= 0.55 or t >= 0.38)

print(f'\nHemolytic: {n_hemolytic}/{len(recent)} ({100*n_hemolytic/len(recent):.1f}%)')
print(f'Toxic: {n_toxic}/{len(recent)} ({100*n_toxic/len(recent):.1f}%)')
print(f'Either flagged: {n_either}/{len(recent)} ({100*n_either/len(recent):.1f}%)')
print(f'Mean hemolysis score: {sum(hemo_scores)/len(hemo_scores):.3f}')
print(f'Mean toxicity score: {sum(tox_scores)/len(tox_scores):.3f}')

print(f'\n--- comparison ---')
print(f'v4 (no toxicity terms):        75.7% hemolytic')
print(f'Case 3 (no toxicity terms):    69.0% hemolytic')
print(f'HemoPI2-only trial:            13.0% hemolytic')
print(f'This run (both terms):         {100*n_hemolytic/len(recent):.1f}% hemolytic, {100*n_toxic/len(recent):.1f}% toxic')
