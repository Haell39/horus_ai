"""
Script para validar clipes normais contra os modelos de vídeo e áudio.
Roda inferência em todos os arquivos da pasta validate_normal/ e reporta os resultados.
"""
import os
import sys
import glob

# Adiciona o backend ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.ml import inference

def main():
    # Pasta com clipes normais
    validate_dir = os.path.join(os.path.dirname(__file__), 'validate_normal')
    
    # Encontra todos os vídeos
    patterns = ['*.mp4', '*.mkv', '*.avi', '*.mov', '*.webm']
    video_files = []
    for pattern in patterns:
        video_files.extend(glob.glob(os.path.join(validate_dir, pattern)))
    
    if not video_files:
        print(f"Nenhum vídeo encontrado em {validate_dir}")
        return
    
    print(f"=" * 80)
    print(f"VALIDAÇÃO DE CLIPES NORMAIS")
    print(f"Pasta: {validate_dir}")
    print(f"Total de arquivos: {len(video_files)}")
    print(f"=" * 80)
    print()
    
    # Contadores
    results = {
        'video': {'normal': 0, 'errors': {}, 'failed': 0},
        'audio': {'normal': 0, 'errors': {}, 'failed': 0}
    }
    
    for i, video_path in enumerate(sorted(video_files), 1):
        filename = os.path.basename(video_path)
        print(f"[{i}/{len(video_files)}] {filename}")
        print("-" * 60)
        
        # === TESTE DE VÍDEO ===
        try:
            video_class, video_conf, video_time = inference.analyze_video_frames(video_path, sample_rate_hz=2.0)
            
            if video_class.lower() == 'normal':
                results['video']['normal'] += 1
                video_status = f"✓ NORMAL ({video_conf:.2%})"
            else:
                results['video']['errors'][video_class] = results['video']['errors'].get(video_class, 0) + 1
                video_status = f"✗ {video_class} ({video_conf:.2%}) @ {video_time}s"
        except Exception as e:
            results['video']['failed'] += 1
            video_status = f"⚠ ERRO: {e}"
        
        print(f"  Vídeo: {video_status}")
        
        # === TESTE DE ÁUDIO ===
        try:
            audio_class, audio_conf, audio_start, audio_end = inference.analyze_audio_segments(video_path)
            
            if audio_class.lower() == 'normal':
                results['audio']['normal'] += 1
                audio_status = f"✓ NORMAL ({audio_conf:.2%})"
            else:
                results['audio']['errors'][audio_class] = results['audio']['errors'].get(audio_class, 0) + 1
                start_str = f"{audio_start}s" if audio_start else "N/A"
                end_str = f"{audio_end}s" if audio_end else "N/A"
                audio_status = f"✗ {audio_class} ({audio_conf:.2%}) @ {start_str}-{end_str}"
        except Exception as e:
            results['audio']['failed'] += 1
            audio_status = f"⚠ ERRO: {e}"
        
        print(f"  Áudio: {audio_status}")
        print()
    
    # === RESUMO ===
    print("=" * 80)
    print("RESUMO")
    print("=" * 80)
    
    total = len(video_files)
    
    print(f"\n📹 VÍDEO:")
    print(f"  Normal: {results['video']['normal']}/{total} ({results['video']['normal']/total:.1%})")
    if results['video']['errors']:
        print(f"  Falsos positivos:")
        for err_class, count in sorted(results['video']['errors'].items(), key=lambda x: -x[1]):
            print(f"    - {err_class}: {count}")
    if results['video']['failed']:
        print(f"  Erros de processamento: {results['video']['failed']}")
    
    print(f"\n🔊 ÁUDIO:")
    print(f"  Normal: {results['audio']['normal']}/{total} ({results['audio']['normal']/total:.1%})")
    if results['audio']['errors']:
        print(f"  Falsos positivos:")
        for err_class, count in sorted(results['audio']['errors'].items(), key=lambda x: -x[1]):
            print(f"    - {err_class}: {count}")
    if results['audio']['failed']:
        print(f"  Erros de processamento: {results['audio']['failed']}")
    
    # Taxa de acerto
    video_accuracy = results['video']['normal'] / total if total > 0 else 0
    audio_accuracy = results['audio']['normal'] / total if total > 0 else 0
    
    print(f"\n📊 TAXA DE ACERTO:")
    print(f"  Vídeo: {video_accuracy:.1%}")
    print(f"  Áudio: {audio_accuracy:.1%}")
    print(f"  Combinado (ambos normal): ", end="")
    
    # Conta quantos foram normais em AMBOS
    both_normal = min(results['video']['normal'], results['audio']['normal'])
    print(f"{both_normal}/{total} ({both_normal/total:.1%})" if total > 0 else "N/A")

if __name__ == '__main__':
    main()
