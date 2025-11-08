#!/usr/bin/env python3
"""
Helper script to add path fixes to all scripts that need to import from src/
"""
import os

PATH_FIX = """import sys
import os
# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
"""

def add_path_fix_to_file(filepath):
    """Add path fix if not already present"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'sys.path.insert(0, os.path.join(os.path.dirname(__file__)' in content:
        print(f"✓ {filepath} already has path fix")
        return False

    # Find where to insert (after existing imports at top)
    lines = content.split('\n')
    insert_pos = 0

    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_pos = i
        elif insert_pos > 0 and not line.strip().startswith('#') and line.strip():
            break

    # Insert after last import
    if insert_pos > 0:
        lines.insert(insert_pos + 1, '')
        lines.insert(insert_pos + 2, '# Add parent directory to path to import from src')
        lines.insert(insert_pos + 3, 'sys.path.insert(0, os.path.join(os.path.dirname(__file__), \'..\'))')
        lines.insert(insert_pos + 4, '')

        new_content = '\n'.join(lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✓ Added path fix to {filepath}")
        return True

    print(f"✗ Could not find insertion point in {filepath}")
    return False

if __name__ == '__main__':
    # Files that need path fixes
    files = [
        'scripts/preprocess.py',
        'scripts/train.py',
        'experiments/run_focused_experiment.py',
        'experiments/run_gaussian_rate_experiments.py',
        'experiments/run_amp_phase_experiment.py',
        'analysis/analyze_subset_fixed.py',
    ]

    for f in files:
        if os.path.exists(f):
            add_path_fix_to_file(f)
