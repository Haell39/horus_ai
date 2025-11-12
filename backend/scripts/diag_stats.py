import json
import os
import statistics
from glob import glob

DIAG_DIR = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'diagnostics')
files = glob(os.path.join(DIAG_DIR, 'upload_*.diagnostic.json'))
blur_vals = []
edge_vals = []
counts = 0
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        diag = data.get('diagnostic') or data.get('diagnostic', [])
        for item in diag:
            h = item.get('heuristics') or {}
            if 'blur_var' in h and isinstance(h.get('blur_var'), (int, float)):
                blur_vals.append(float(h.get('blur_var')))
            if 'edge_density' in h and isinstance(h.get('edge_density'), (int, float)):
                edge_vals.append(float(h.get('edge_density')))
            counts += 1
    except Exception as e:
        print('ERR reading', f, e)

print(f'Found {len(files)} diagnostic files, {counts} frame samples total')

def describe(vals):
    if not vals:
        return None
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    mean = statistics.mean(vals_sorted)
    median = statistics.median(vals_sorted)
    p25 = vals_sorted[int(0.25 * (n-1))]
    p75 = vals_sorted[int(0.75 * (n-1))]
    p90 = vals_sorted[int(0.90 * (n-1))]
    p10 = vals_sorted[int(0.10 * (n-1))]
    return {'count': n, 'mean': mean, 'median': median, 'p10': p10, 'p25': p25, 'p75': p75, 'p90': p90}

bdesc = describe(blur_vals)
edesc = describe(edge_vals)

print('\nBlur variance stats:')
if bdesc:
    for k,v in bdesc.items():
        print(f'  {k}: {v}')
    # propose threshold: choose p25 for blur_var
    prop = bdesc['p25']
    print(f'  Proposed VIDEO_BLUR_VAR_THRESHOLD ~ {prop:.1f} (25th percentile)')
else:
    print('  No blur_var samples found (diagnostics may be old).')

print('\nEdge density stats:')
if edesc:
    for k,v in edesc.items():
        print(f'  {k}: {v}')
    # propose threshold: choose p10 for edge density (low detail)
    prop_e = edesc['p10']
    print(f'  Proposed VIDEO_EDGE_DENSITY_THRESHOLD ~ {prop_e:.5f} (10th percentile)')
else:
    print('  No edge_density samples found (diagnostics may be old).')

print('\nSummary recommendations:')
if bdesc:
    print(f"  VIDEO_BLUR_VAR_THRESHOLD = {bdesc['p25']:.1f}")
if edesc:
    print(f"  VIDEO_EDGE_DENSITY_THRESHOLD = {edesc['p10']:.5f}")
print('\nYou can export these to env or edit backend/app/core/config.py')
