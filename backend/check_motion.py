import json

with open(r'D:\GitHub Desktop\horus_ai\backend\app\static\diagnostics\upload_6c0ea0a2073b4e1dacdb0ca724e2d4e7.diagnostic.json', 'r') as f:
    data = json.load(f)

# Filtrar frames com motion < 3.0 (indicativo de freeze)
print('=== Frames com BAIXA MOTION (possível freeze) ===')
for d in data['diagnostic']:
    motion = d['heuristics'].get('motion', 0)
    if motion is not None and motion < 3.0 and motion != float('inf'):
        print(f"Time: {d['time_s']:.2f}s - Motion: {motion:.2f}")

print()
print('=== Frames entre 16s e 23s (área de interesse) ===')
for d in data['diagnostic']:
    t = d['time_s']
    if 16 <= t <= 23:
        motion = d['heuristics'].get('motion', 0)
        brightness = d['heuristics'].get('brightness', 0)
        print(f"Time: {t:.2f}s - Motion: {motion:.2f}, Brightness: {brightness:.1f}")
