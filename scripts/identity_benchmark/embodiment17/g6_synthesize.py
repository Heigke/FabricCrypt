"""G6: Final synthesis & verdict for Phase 17.

Combines G1 sample examples, G2 classifier, G3 clone-defeat, G4 consistency,
G5 divergence. Reports overall verdict.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import RESULTS, save_json


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    g2 = load('g2_classifier.json')
    g3 = load('g3_clone_defeat.json')
    g4 = load('g4_consistency.json')
    g5 = load('g5_divergence.json')

    verdict = {
        'measurable_personality': None,
        'chip_bound': None,
        'clone_resistant': None,
        'stable_over_time': None,
        'tests_passed': [],
        'tests_failed': [],
        'detail': {}
    }

    # Test 1 (G2): measurable + chip-bound from classifier
    if g2 is not None:
        t = g2.get('gate_all', False)
        verdict['measurable_personality'] = bool(
            g2['results']['embodied']['mean'] > g2['results']['vanilla']['mean'])
        verdict['detail']['G2'] = g2
        (verdict['tests_passed'] if t else verdict['tests_failed']).append('G2_classifier')

    # Test 2 (G3): clone-defeat
    if g3 is not None:
        t = g3.get('pass_clone_defeat', False)
        verdict['clone_resistant'] = bool(t)
        verdict['detail']['G3'] = g3
        (verdict['tests_passed'] if t else verdict['tests_failed']).append('G3_clone_defeat')

    # Test 3 (G4): stable over time
    if g4 is not None:
        t = g4.get('pass_intra_consistency_gt_0p7', False)
        verdict['stable_over_time'] = bool(t)
        verdict['detail']['G4'] = g4
        (verdict['tests_passed'] if t else verdict['tests_failed']).append('G4_consistency')

    # Test 4 (G5): cross-chip divergence
    if g5 is not None:
        t = g5.get('pass_embodied_ks_lt_1e-3', False)
        verdict['chip_bound'] = bool(t and verdict.get('chip_bound', True))
        verdict['detail']['G5'] = g5
        (verdict['tests_passed'] if t else verdict['tests_failed']).append('G5_divergence')

    n_pass = len(verdict['tests_passed'])
    verdict['n_tests_passed'] = n_pass
    verdict['overall_pass_4plus'] = n_pass >= 4
    verdict['overall_pass_3plus'] = n_pass >= 3

    print("\n=== PHASE 17 VERDICT ===")
    print(f"  measurable_personality : {verdict['measurable_personality']}")
    print(f"  chip_bound            : {verdict['chip_bound']}")
    print(f"  clone_resistant       : {verdict['clone_resistant']}")
    print(f"  stable_over_time      : {verdict['stable_over_time']}")
    print(f"  tests_passed          : {verdict['tests_passed']}")
    print(f"  tests_failed          : {verdict['tests_failed']}")
    print(f"  4/4 strong claim PASS : {verdict['overall_pass_4plus']}")

    save_json('g6_synthesis.json', verdict)


if __name__ == '__main__':
    main()
