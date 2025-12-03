"""
Validacao do modelo de Lipsync nos videos da pasta validate_normal.
Expectativa: Todos devem ser detectados como 'sincronizado' ou 'sem_fala'.
"""
import os
import sys
from pathlib import Path

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Importar inference
from app.ml.inference import analyze_lipsync, analyze_video_frames, analyze_audio_segments

FOLDER = r"D:\GitHub Desktop\horus_ai\validate_normal"

def main():
    videos = sorted(Path(FOLDER).glob("*.mp4"))
    print(f"[VIDEO] Encontrados {len(videos)} videos para validar\n")
    print("="*80)
    
    results = {
        'sincronizado': [],
        'dessincronizado': [],
        'sem_fala': [],
        'erro': []
    }
    
    for i, video in enumerate(videos, 1):
        print(f"\n[{i:02d}/{len(videos)}] {video.name}")
        
        try:
            # Primeiro verifica video/audio
            vid_class, vid_conf, _ = analyze_video_frames(str(video))
            aud_result = analyze_audio_segments(str(video))
            aud_class = aud_result[0]
            aud_conf = aud_result[1]
            
            print(f"   Video: {vid_class} ({vid_conf*100:.1f}%)")
            print(f"   Audio: {aud_class} ({aud_conf*100:.1f}%)")
            
            # Só roda lipsync se video+audio normais
            vid_ok = str(vid_class).lower() == 'normal' or vid_conf < 0.5
            aud_ok = str(aud_class).lower() == 'normal' or aud_conf < 0.5
            
            if not (vid_ok and aud_ok):
                print(f"   [!] SKIP lipsync (video/audio nao normal)")
                results['erro'].append((video.name, "skip_not_normal"))
                continue
            
            # Rodar lipsync
            lip_class, lip_conf, offset = analyze_lipsync(str(video))
            
            status = "[OK]" if lip_class == 'sincronizado' else ("[?]" if lip_class == 'sem_fala' else "[X]")
            print(f"   Lipsync: {status} {lip_class} ({lip_conf*100:.1f}%) offset={offset:.1f}ms")
            
            results[lip_class].append((video.name, lip_conf, offset))
            
        except Exception as e:
            print(f"   [X] ERRO: {e}")
            results['erro'].append((video.name, str(e)))
    
    # Resumo
    print("\n" + "="*80)
    print("[RESUMO] VALIDACAO LIPSYNC")
    print("="*80)
    
    total = len(videos)
    sync = len(results['sincronizado'])
    desync = len(results['dessincronizado'])
    sem_fala = len(results['sem_fala'])
    erros = len(results['erro'])
    
    print(f"\n[OK] Sincronizado:    {sync:3d} ({sync/total*100:.1f}%)")
    print(f"[X]  Dessincronizado: {desync:3d} ({desync/total*100:.1f}%)")
    print(f"[?]  Sem fala:        {sem_fala:3d} ({sem_fala/total*100:.1f}%)")
    print(f"[!]  Erros/Skip:      {erros:3d} ({erros/total*100:.1f}%)")
    
    # Acurácia (sincronizado + sem_fala são OK para vídeos normais)
    corretos = sync + sem_fala
    testados = total - erros
    if testados > 0:
        acc = corretos / testados * 100
        print(f"\n[ACURACIA] (sync+sem_fala / testados): {acc:.1f}%")
    
    # Lista de falsos positivos
    if desync > 0:
        print(f"\n[FALSOS POSITIVOS] (dessincronizado em videos normais):")
        for name, conf, offset in results['dessincronizado']:
            print(f"   - {name}: {conf*100:.1f}% conf, {offset:.1f}ms offset")
    
    return desync == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
