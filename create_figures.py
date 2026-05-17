import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Figure 7 — ROUGE Scores ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
scores  = [0.4121, 0.2051, 0.4121]
colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(metrics, scores, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
ax.set_ylim(0, 0.6)
ax.set_title('ROUGE Score Results — RAG System', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('F1 Score', fontsize=12)
ax.axhline(y=0.35, color='red', linestyle='--', alpha=0.5, label='Benchmark lower bound')
ax.axhline(y=0.45, color='green', linestyle='--', alpha=0.5, label='Benchmark upper bound')
ax.legend(fontsize=9)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, str(score), ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('figure7_rouge.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ Figure 7 saved!')

# ── Figure 8 — BERTScore ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
metrics2 = ['BERTScore\nPrecision', 'BERTScore\nRecall', 'BERTScore\nF1']
scores2  = [0.8918, 0.9361, 0.9131]
colors2  = ['#9467bd', '#8c564b', '#e377c2']
bars2 = ax.bar(metrics2, scores2, color=colors2, width=0.5, edgecolor='black', linewidth=0.5)
ax.set_ylim(0.85, 0.96)
ax.set_title('BERTScore Results — RAG System', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Score', fontsize=12)
for bar, score in zip(bars2, scores2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, str(score), ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('figure8_bertscore.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ Figure 8 saved!')

# ── Figure 9 — All 8 Metrics ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
all_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore P', 'BERTScore R', 'BERTScore F1', 'Faithfulness', 'Answer Rel.']
all_scores  = [0.4121, 0.2051, 0.4121, 0.8918, 0.9361, 0.9131, 0.5567, 0.2857]
colors3 = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22']
bars3 = ax.bar(all_metrics, all_scores, color=colors3, width=0.6, edgecolor='black', linewidth=0.5)
ax.set_ylim(0, 1.1)
ax.set_title('All Evaluation Metrics — RAG System', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Score', fontsize=12)
ax.set_xticklabels(all_metrics, rotation=30, ha='right', fontsize=10)
for bar, score in zip(bars3, all_scores):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, str(score), ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('figure9_all_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ Figure 9 saved!')

# ── Figure 11 — Latency ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
latencies = [3.1, 3.8, 4.2, 4.5, 4.3, 5.1, 5.8, 6.2, 6.8, 7.1, 4.9, 5.3, 4.1, 3.9, 5.5, 6.0, 4.7, 5.9, 3.5, 4.4]
ax.hist(latencies, bins=10, color='#1f77b4', edgecolor='black', linewidth=0.5)
ax.axvline(x=np.mean(latencies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(latencies):.2f}s')
ax.set_title('Query Response Latency Distribution', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Latency (seconds)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('figure11_latency.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ Figure 11 saved!')

print('\n🎉 All charts created in C:/thesis/ folder!')