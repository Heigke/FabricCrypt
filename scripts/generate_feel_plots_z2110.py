#!/usr/bin/env python3
"""Generate all FEEL paper plots updated with z2110 results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap

OUT = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/FEEL_overleaf"
AMD_RED = "#ED1C24"
plt.rcParams.update({'font.size': 12, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'figure.dpi': 100})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print(f"  Saved {name}")

# FIG 1
def fig1():
    exps = [('z2060',8,8,'CNN'),('z2061',12,12,'CNN'),('z2068',21,22,'ViT'),('z2070',22,24,'ViT'),('z2076',12,12,'Transformer'),('z2078',13,14,'Transformer'),('z2081',15,16,'Transformer'),('z2087',12,14,'Transformer'),('z2090',17,18,'Transformer'),('z2091',12,14,'GPT-2'),('z2094',18,20,'GPT-2'),('z2095',19,20,'GPT-2'),('z2096',20,20,'GPT-2'),('z2097',24,25,'Qwen2.5'),('z2098',27,31,'Qwen2.5'),('z2099',28,39,'Qwen2.5'),('z2100',28,39,'Qwen2.5'),('z2101',30,40,'Qwen2.5'),('z2103',34,40,'Qwen2.5'),('z2107',30,41,'Qwen3'),('z2110',25,41,'Qwen3')]
    colors = {'CNN':'#4477AA','ViT':'#228B22','Transformer':'#E69F00','GPT-2':'#8855AA','Qwen2.5':'#CC3311','Qwen3':'#8B0000'}
    names = [e[0] for e in exps]; rates = [e[1]/e[2]*100 for e in exps]; cols = [colors[e[3]] for e in exps]
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(exps))
    ax.bar(x, rates, color=cols, edgecolor='black', linewidth=0.5)
    ax.axhline(100, color='gray', ls='--', lw=0.8, alpha=0.5)
    idx_z2110 = names.index('z2110')
    ax.plot(idx_z2110, rates[idx_z2110]+2, marker='*', color=AMD_RED, ms=14, zorder=5)
    for i, e in enumerate(exps):
        ax.text(i, rates[i]+0.8, f"{e[1]}/{e[2]}", ha='center', va='bottom', fontsize=7, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Pass Rate (%)'); ax.set_title('FEEL Test Pass Rates Across Experiments'); ax.set_ylim(0, 115)
    handles = [mpatches.Patch(color=v, label=k) for k,v in colors.items()]
    ax.legend(handles=handles, loc='lower left', fontsize=9, ncol=3); fig.tight_layout(); save(fig, 'fig1_pass_rates')

# FIG 3
def fig3():
    data = [('z2091',61.7,'GPT-2'),('z2093',34.4,'GPT-2'),('z2094',47.9,'GPT-2'),('z2095',47.3,'GPT-2'),('z2096',48.3,'GPT-2'),('z2097',2.92,'Qwen2.5'),('z2098',2.90,'Qwen2.5'),('z2099',2.92,'Qwen2.5'),('z2100',2.73,'Qwen2.5'),('z2101',3.81,'Qwen2.5'),('z2103',9.49,'Qwen2.5'),('z2107',1.06,'Qwen3'),('z2110',14.75,'Qwen3')]
    colors = {'GPT-2':'#8855AA','Qwen2.5':'#CC3311','Qwen3':'#8B0000'}
    baselines = [('GPT-2 baseline',62.67,'#8855AA'),('Qwen2.5 baseline',18.76,'#CC3311'),('Qwen3-8B baseline',18.59,'#8B0000')]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(data)); cols = [colors[d[2]] for d in data]; vals = [d[1] for d in data]
    ax.bar(x, vals, color=cols, edgecolor='black', linewidth=0.5)
    for bl_name, bl_val, bl_col in baselines:
        ax.axhline(bl_val, color=bl_col, ls='--', lw=1.2, alpha=0.7, label=bl_name)
    idx = [d[0] for d in data].index('z2110')
    ax.plot(idx, vals[idx]*1.15, marker='*', color=AMD_RED, ms=14, zorder=5)
    for i, d in enumerate(data):
        ypos = d[1]*1.08 if d[1]>2 else d[1]+0.3
        ax.text(i, ypos, f"{d[1]:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Perplexity (log scale)'); ax.set_title('LM Perplexity Progression'); ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout(); save(fig, 'fig3_perplexity')

# FIG 7
def fig7():
    data = [('z2091',2.1),('z2094',1.078),('z2095',1.283),('z2097',1.891),('z2100',4.304),('z2101',4.304),('z2103',1.294),('z2107',9.48),('z2109',1.065),('z2110',6.220)]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(data)); vals = [d[1] for d in data]
    cols = ['#4477AA']*len(data); cols[-1] = AMD_RED
    ax.bar(x, vals, color=cols, edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='gray', ls='-', lw=1, label='No effect (1.0)')
    ax.axhline(1.5, color='orange', ls='--', lw=1, label='Threshold (1.5)')
    for i, d in enumerate(data):
        ax.text(i, d[1]+0.15, f"{d[1]:.2f}", ha='center', va='bottom', fontsize=9)
    ax.plot(x[-1], vals[-1]+0.5, marker='*', color=AMD_RED, ms=14, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=45, ha='right')
    ax.set_ylabel('Kill-Shot Ratio (wrong/correct PPL)'); ax.set_title('T7 Kill-Shot: Wrong Regime Gate Causes PPL Spike')
    ax.legend(fontsize=10); fig.tight_layout(); save(fig, 'fig7_killshot')

# FIG 8
def fig8():
    all_tests = ['SANITY'] + [f'T{i}' for i in range(1, 41)]
    passed = {'SANITY','T1','T2','T3','T5','T6','T7','T8','T9','T11','T12','T14','T15','T16','T17','T18','T25','T28','T29','T30','T33','T34','T37','T39','T40'}
    ncols = 7; nrows = int(np.ceil(len(all_tests)/ncols))
    grid = np.full((nrows, ncols), np.nan); labels = [[''] * ncols for _ in range(nrows)]
    for idx, t in enumerate(all_tests):
        r, c = divmod(idx, ncols)
        grid[r][c] = 1 if t in passed else 0
        labels[r][c] = t
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = ListedColormap(['#DD4444', '#44BB44'])
    ax.imshow(grid, cmap=cmap, aspect='equal', vmin=0, vmax=1)
    for r in range(nrows):
        for c in range(ncols):
            if labels[r][c]:
                ax.text(c, r, labels[r][c], ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title('z2110 Falsification Battery: 25/41 PASS (61%)', fontsize=14)
    legend_elements = [mpatches.Patch(facecolor='#44BB44', label='PASS'), mpatches.Patch(facecolor='#DD4444', label='FAIL')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11); fig.tight_layout(); save(fig, 'fig8_falsification')

# FIG 9
def fig9():
    data = [('z2097',3.177),('z2100',1.447),('z2101',1.447),('z2103',9.446),('z2107',1.84),('z2110',0.599)]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(data)); vals = [d[1] for d in data]; cols = ['#44BB44' if v>=1.0 else '#DD4444' for v in vals]
    ax.bar(x, vals, color=cols, edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='black', ls='--', lw=1.2, label='Threshold (1.0)')
    for i, d in enumerate(data):
        ax.text(i, d[1]+0.15, f"{d[1]:.3f}", ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=45, ha='right')
    ax.set_ylabel('Embodiment Ratio (ablated/full PPL)'); ax.set_title('T4 Embodiment Gap: Hardware Ablation Effect on PPL')
    ax.legend(fontsize=10)
    idx = [d[0] for d in data].index('z2110')
    ax.annotate('FAIL: ablated\nbetter', xy=(idx, vals[idx]), xytext=(idx+0.5, vals[idx]+1.5), arrowprops=dict(arrowstyle='->', color=AMD_RED, lw=1.5), fontsize=9, color=AMD_RED, fontweight='bold')
    fig.tight_layout(); save(fig, 'fig9_embodiment')

# FIG 10
def fig10():
    tokens = [17, 12, 8, 4, 2, 1]; ppls = [24.96, 24.57, 23.62, 25.40, 24.66, 248.83]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tokens, ppls, 'o-', color='#4477AA', lw=2, ms=8)
    ax.annotate('10.09x cliff!\n(2->1 tokens)', xy=(1, 248.83), xytext=(4, 200), arrowprops=dict(arrowstyle='->', color=AMD_RED, lw=2), fontsize=12, color=AMD_RED, fontweight='bold')
    for t, p in zip(tokens, ppls):
        offset = -15 if p < 100 else 10
        ax.text(t, p+offset, f"{p:.1f}", ha='center', fontsize=9)
    ax.set_xlabel('Workspace Tokens'); ax.set_ylabel('Perplexity'); ax.set_title('T33 Workspace Capacity Cliff (z2110)')
    ax.invert_xaxis(); ax.set_xticks(tokens); fig.tight_layout(); save(fig, 'fig10_workspace_cliff')

# FIG 11
def fig11():
    traj = [0.447,0.454,0.453,0.447,0.441,0.437,0.449,0.445,0.457,0.435,0.440,0.444,0.453,0.450,0.458,0.450,0.451,0.434,0.452,0.455,0.450,0.449,0.448,0.451,0.448,0.445,0.450,0.456,0.450,0.440]
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = np.arange(len(traj))
    ax.plot(steps, traj, 'o-', color='#4477AA', lw=1.5, ms=5)
    ax.axhline(np.mean(traj), color='gray', ls='--', lw=1, alpha=0.7, label=f'Mean={np.mean(traj):.3f}')
    ax.fill_between(steps, np.mean(traj)-np.std(traj), np.mean(traj)+np.std(traj), alpha=0.15, color='#4477AA')
    ax.set_xlabel('Step'); ax.set_ylabel('Gate Value'); ax.set_title('T34 PCIST Gate Trajectory (z2110) — LZ complexity ratio = 1.76')
    ax.legend(fontsize=10); ax.set_ylim(0.42, 0.47); fig.tight_layout(); save(fig, 'fig11_pcist_trajectory')

# FIG 12
def fig12():
    data = [('z2097',24,25),('z2098',27,31),('z2099',28,39),('z2100',28,39),('z2101',30,40),('z2103',34,40),('z2107',30,41),('z2110',25,41)]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(data)); passed = [d[1] for d in data]; failed = [d[2]-d[1] for d in data]
    ax.bar(x, passed, color='#44BB44', edgecolor='black', lw=0.5, label='Pass')
    ax.bar(x, failed, bottom=passed, color='#DD4444', edgecolor='black', lw=0.5, label='Fail')
    for i, d in enumerate(data):
        rate = d[1]/d[2]*100
        ax.text(i, d[2]+0.5, f"{d[1]}/{d[2]}\n({rate:.0f}%)", ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=45, ha='right')
    ax.set_ylabel('Number of Tests'); ax.set_title('Test Battery Evolution: Growing Rigor')
    ax.legend(fontsize=10); fig.tight_layout(); save(fig, 'fig12_test_evolution')

# FIG 13
def fig13():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(['Cold', 'Hot'], [11.90, 108.23], color=['#4477AA', AMD_RED], edgecolor='black', linewidth=0.5, width=0.5)
    ax.set_ylabel('Perplexity'); ax.set_title('T40 Thermal Delirium (z2110): 9.1x Ratio — LARGEST EVER')
    ax.annotate('9.096x', xy=(1, 108.23), xytext=(1.3, 80), arrowprops=dict(arrowstyle='->', color='black', lw=1.5), fontsize=16, fontweight='bold', color=AMD_RED)
    ax.text(0, 11.90+3, '11.90', ha='center', fontsize=11, fontweight='bold')
    ax.text(1, 108.23+3, '108.23', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 130, r'$\alpha$ = 0.339', ha='center', fontsize=11, style='italic')
    fig.tight_layout(); save(fig, 'fig13_thermal_delirium')

# FIG 4
def fig4():
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ['Low DVFS\n(487.7 uJ/tok)', 'High DVFS\n(381.2 uJ/tok)', 'Model\n(383.9 uJ/tok)']
    vals = [487.7, 381.2, 383.9]; cols = ['#4477AA', '#44BB44', AMD_RED]
    ax.bar(range(3), vals, color=cols, edgecolor='black', linewidth=0.5, width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v+10, f"{v:.1f}", ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Energy (uJ/token)'); ax.set_title('z2110 Energy Efficiency (T14)')
    ax.set_ylim(0, 560); fig.tight_layout(); save(fig, 'fig4_energy')

if __name__ == '__main__':
    print("Generating FEEL paper plots for z2110...")
    fig1(); fig3(); fig7(); fig8(); fig9(); fig10(); fig11(); fig12(); fig13(); fig4()
    print("Done!")
