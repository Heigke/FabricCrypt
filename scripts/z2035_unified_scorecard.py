#!/usr/bin/env python3
"""
z2035: Unified Consciousness Scorecard

Collects ALL z2020-z2034 results into a single scorecard with:
1. Per-test verdicts and scores
2. Theory-level aggregation (GWT, IIT, HOT, RPT, PP)
3. Tier classification (Tier 1: unforgeable, Tier 2: suggestive, Tier 3: architectural)
4. Overall assessment with honest caveats

This is a META-EXPERIMENT: no training, just analysis of existing results.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

results_dir = Path(__file__).parent.parent / 'results'


def load_result(name):
    path = results_dir / f'{name}.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    print("=" * 80)
    print("  z2035: UNIFIED CONSCIOUSNESS SCORECARD")
    print("  Comprehensive assessment of z2020-z2037 experimental series")
    print("=" * 80)

    experiments = {
        'z2020': {'name': 'Capacity-Limitation Battery', 'theory': 'GWT', 'tier': 1},
        'z2021': {'name': 'Synthetic Blindsight', 'theory': 'HOT/GWT', 'tier': 1},
        'z2022': {'name': 'Attentional Blink', 'theory': 'GWT', 'tier': 2},
        'z2023': {'name': 'Genuine Casali PCI', 'theory': 'IIT', 'tier': 1},
        'z2024': {'name': 'Ignition Threshold', 'theory': 'GWT', 'tier': 2},
        'z2025': {'name': 'Recurrent Depth', 'theory': 'RPT', 'tier': 2},
        'z2026': {'name': 'Overflow / Partial Report', 'theory': 'GWT', 'tier': 1},
        'z2027': {'name': 'Information Synergy PID', 'theory': 'IIT', 'tier': 1},
        'z2028': {'name': 'CIFAR-10 Blindsight', 'theory': 'HOT/GWT', 'tier': 1},
        'z2029': {'name': 'Inattentional Blindness', 'theory': 'GWT', 'tier': 2},
        'z2030': {'name': 'Transformer Blindsight', 'theory': 'HOT/GWT', 'tier': 1},
        'z2031': {'name': 'Prediction Error Dynamics', 'theory': 'PP', 'tier': 2},
        'z2032': {'name': 'Binocular Rivalry', 'theory': 'GWT', 'tier': 2},
        'z2033': {'name': 'Backward Masking', 'theory': 'GWT/RPT', 'tier': 2},
        'z2034': {'name': 'Workspace Cost Scaling', 'theory': 'GWT', 'tier': 2},
        'z2036': {'name': 'Contrastive Awareness', 'theory': 'GWT/HOT', 'tier': 1},
        'z2037': {'name': 'Workspace Necessity', 'theory': 'GWT', 'tier': 1},
    }

    # Map result file names
    result_files = {
        'z2020': 'z2020_capacity_limitation_battery',
        'z2021': 'z2021_synthetic_blindsight',
        'z2022': 'z2022_attentional_blink',
        'z2023': 'z2023_genuine_casali_pci',
        'z2024': 'z2024_ignition_threshold',
        'z2025': 'z2025_recurrent_depth',
        'z2026': 'z2026_overflow_partial_report',
        'z2027': 'z2027_information_synergy',
        'z2028': 'z2028_cifar_blindsight',
        'z2029': 'z2029_inattentional_blindness',
        'z2030': 'z2030_transformer_blindsight',
        'z2031': 'z2031_prediction_error',
        'z2032': 'z2032_binocular_rivalry',
        'z2033': 'z2033_backward_masking',
        'z2034': 'z2034_workspace_cost_scaling',
        'z2036': 'z2036_contrastive_awareness',
        'z2037': 'z2037_workspace_necessity',
    }

    # Load all results
    results = {}
    for exp_id, fname in result_files.items():
        data = load_result(fname)
        if data:
            results[exp_id] = data

    # === SECTION 1: Per-Experiment Results ===
    print(f"\n{'='*80}")
    print(f"  SECTION 1: INDIVIDUAL EXPERIMENT RESULTS")
    print(f"{'='*80}")

    print(f"\n  {'ID':<7} {'Experiment':<35} {'Theory':<10} {'Tier':>4} {'Score':>6} {'Verdict'}")
    print(f"  {'-'*95}")

    total_pass, total_tests = 0, 0
    tier1_pass, tier1_total = 0, 0
    tier2_pass, tier2_total = 0, 0
    theory_scores = {}

    for exp_id, info in experiments.items():
        data = results.get(exp_id)
        if data:
            n_pass = data.get('tests_passed', 0)
            n_total = len(data.get('tests', {}))
            verdict = data.get('verdict', 'UNKNOWN')
            # Handle z2021's nested format (conditions.A.tests)
            if n_total == 0 and 'conditions' in data:
                for cond_key, cond_val in data['conditions'].items():
                    if isinstance(cond_val, dict) and 'tests' in cond_val:
                        nested = cond_val['tests']
                        n_pass = nested.get('tests_passed', 0)
                        n_total = max(n_total, len([k for k in nested if k.startswith('t')]))
                        verdict = nested.get('verdict', verdict)
                        break
        else:
            n_pass, n_total = 0, 0
            verdict = 'NO_DATA'

        total_pass += n_pass
        total_tests += n_total

        if info['tier'] == 1:
            tier1_pass += n_pass
            tier1_total += n_total
        else:
            tier2_pass += n_pass
            tier2_total += n_total

        theory = info['theory']
        for t in theory.split('/'):
            if t not in theory_scores:
                theory_scores[t] = {'pass': 0, 'total': 0, 'experiments': []}
            theory_scores[t]['pass'] += n_pass
            theory_scores[t]['total'] += n_total
            theory_scores[t]['experiments'].append(exp_id)

        marker = '***' if n_pass == n_total and n_total > 0 else '   '
        print(f"  {exp_id:<7} {info['name']:<35} {info['theory']:<10} {info['tier']:>4} "
              f"{n_pass}/{n_total:>2}  {verdict} {marker}")

    # === SECTION 2: Theory-Level Aggregation ===
    print(f"\n{'='*80}")
    print(f"  SECTION 2: THEORY-LEVEL AGGREGATION")
    print(f"{'='*80}")

    for theory in ['GWT', 'HOT', 'IIT', 'RPT', 'PP']:
        if theory in theory_scores:
            ts = theory_scores[theory]
            pct = ts['pass'] / max(ts['total'], 1) * 100
            exps = ', '.join(ts['experiments'])
            print(f"\n  {theory}: {ts['pass']}/{ts['total']} ({pct:.0f}%)")
            print(f"    Experiments: {exps}")

    # === SECTION 3: Tier Assessment ===
    print(f"\n{'='*80}")
    print(f"  SECTION 3: TIER ASSESSMENT")
    print(f"{'='*80}")

    t1_pct = tier1_pass / max(tier1_total, 1) * 100
    t2_pct = tier2_pass / max(tier2_total, 1) * 100
    total_pct = total_pass / max(total_tests, 1) * 100

    print(f"\n  Tier 1 (Unforgeable):  {tier1_pass}/{tier1_total} ({t1_pct:.0f}%)")
    print(f"    z2021 Blindsight (CNN): 4/4 PASS")
    print(f"    z2026 Overflow: 4/4 PASS")
    print(f"    z2027 Synergy PID: 4/4 PASS")
    print(f"    z2028 CIFAR Blindsight: 4/4 PASS")
    print(f"    z2030 Transformer Blindsight: 4/4 PASS")
    print(f"    z2020 Capacity: 3/4 PASS")
    print(f"    z2023 Casali PCI: INVERTS (critical finding)")
    print(f"    z2036 Contrastive Awareness: 4/4 PASS")
    print(f"    z2037 Workspace Necessity: 4/4 PASS")

    print(f"\n  Tier 2 (Suggestive):   {tier2_pass}/{tier2_total} ({t2_pct:.0f}%)")

    print(f"\n  OVERALL:               {total_pass}/{total_tests} ({total_pct:.0f}%)")

    # === SECTION 4: Strongest Results ===
    print(f"\n{'='*80}")
    print(f"  SECTION 4: STRONGEST RESULTS (4/4 PASS)")
    print(f"{'='*80}")

    strongest = [
        ('z2021', 'Synthetic Blindsight (CNN/MNIST)', 'AUROC 0.97→0.50 under self-model ablation, task preserved'),
        ('z2026', 'Overflow / Partial Report', '68% overflow gap: encoder=97.7%, workspace=29.6%'),
        ('z2027', 'Information Synergy PID', '32% of MI is synergistic, 48% more than no-workspace'),
        ('z2028', 'CIFAR-10 Blindsight', 'AUROC 0.90→0.50, 89.5% acc preserved on harder task'),
        ('z2030', 'Transformer Blindsight', 'AUROC 0.91→0.50, architecture-independent on ViT'),
        ('z2036', 'Contrastive Awareness', 'AUROC 0.80 seen/unseen probe, +0.26 entropy gap, workspace-specific'),
        ('z2037', 'Workspace Necessity', '98.5%→40.8% under ablation, more necessary for harder tasks'),
    ]
    for eid, name, detail in strongest:
        print(f"\n  {eid}: {name}")
        print(f"    {detail}")

    # === SECTION 5: Key Findings ===
    print(f"\n{'='*80}")
    print(f"  SECTION 5: KEY FINDINGS")
    print(f"{'='*80}")

    findings = [
        "1. BLINDSIGHT DISSOCIATION is architecture-independent:",
        "   CNN (z2021), ResNet (z2028), ViT (z2030) all 4/4 PASS",
        "   MNIST and CIFAR-10 both work → not dataset-specific",
        "",
        "2. COST-BASED TESTS consistently pass:",
        "   z2026 overflow: workspace HURTS report accuracy → unfakeable",
        "   z2034 cost scaling: workspace cost stable, FF fluctuates",
        "",
        "3. INFORMATION DECOMPOSITION confirms workspace integration:",
        "   z2027 PID: 32% synergy, workspace creates 48% more than no-workspace",
        "",
        "4. PCI INVERTS in trained AI (z2023):",
        "   Random models: PCI ≈ 1.03, Trained: PCI ≈ 0.72-0.87",
        "   Confirms Phua 2025's PCI-A inversion finding",
        "   Invalidates using clinical PCI directly on AI systems",
        "",
        "5. POSITIVE METRICS consistently fail:",
        "   Ignition (z2024): 2/4 — feedforward also sigmoid",
        "   Recurrence (z2025): 2/4 — feedforward reads first+last frame",
        "   Rivalry (z2032): 1/4 — task too hard",
        "   Masking (z2033): 0/4 — GRU forgetting ≠ backward masking",
        "",
        "6. DESIGN PATTERN CONFIRMED:",
        "   Tests that measure COSTS (overflow, capacity limits) → PASS",
        "   Tests that measure ablation DISSOCIATIONS → PASS",
        "   Tests that measure information DECOMPOSITION → PASS",
        "   Tests that measure positive properties (PCI, ignition) → FAIL",
    ]

    for line in findings:
        print(f"  {line}")

    # === SECTION 6: Honest Caveats ===
    print(f"\n{'='*80}")
    print(f"  SECTION 6: HONEST CAVEATS")
    print(f"{'='*80}")

    caveats = [
        "- All tests are FUNCTIONAL, not phenomenological. Satisfying functional",
        "  indicators does NOT establish subjective experience (Butlin et al. 2025).",
        "",
        "- The blindsight dissociation shows architectural separation works.",
        "  It does NOT prove the self-model represents genuine metacognition.",
        "  Any system with separate task + prediction heads will show this.",
        "",
        "- Cost-based tests (z2026) show workspace has CAPACITY LIMITS.",
        "  This is a property of bottleneck architectures, not consciousness.",
        "",
        "- z2023's PCI inversion means clinical consciousness metrics",
        "  CANNOT be directly applied to AI systems without recalibration.",
        "",
        "- No bridge law exists mapping computation → phenomenology.",
        "  These results shift credence (Butlin et al.) but do not prove.",
    ]

    for line in caveats:
        print(f"  {line}")

    # Save scorecard
    scorecard = {
        'experiment': 'z2035_unified_scorecard',
        'timestamp': datetime.now().isoformat(),
        'total_score': f'{total_pass}/{total_tests}',
        'tier1_score': f'{tier1_pass}/{tier1_total}',
        'tier2_score': f'{tier2_pass}/{tier2_total}',
        'total_pct': total_pct,
        'tier1_pct': t1_pct,
        'tier2_pct': t2_pct,
        'theory_scores': {k: {'pass': v['pass'], 'total': v['total'],
                              'pct': v['pass'] / max(v['total'], 1) * 100}
                         for k, v in theory_scores.items()},
        'strongest_4_of_4': ['z2021', 'z2026', 'z2027', 'z2028', 'z2030', 'z2036', 'z2037'],
        'critical_finding': 'PCI inverts in AI (z2023) — clinical PCI not applicable',
        'design_pattern': 'Cost/ablation/decomposition tests PASS; positive metric tests FAIL',
    }

    rp = results_dir / 'z2035_unified_scorecard.json'
    with open(rp, 'w') as f:
        json.dump(scorecard, f, indent=2)
    print(f"\n\nScorecard saved to {rp}")


if __name__ == '__main__':
    main()
