"""
Script para validar clipes de ERRO contra os modelos.
Testa se os modelos detectam corretamente os erros conhecidos.
"""
import os
import sys
import glob

# Adiciona o backend ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.ml import inference

def test_video_errors():
    """Testa clipes de erro de vídeo"""
    video_dir = os.path.join(os.path.dirname(__file__), 'validate_model_video')
    
    # Só arquivos na raiz (não entra em normal_clips/)
    video_files = [f for f in glob.glob(os.path.join(video_dir, '*.mp4'))]
    
    if not video_files:
        print("Nenhum vídeo de erro encontrado")
        return
    
    print("=" * 80)
    print("📹 VALIDAÇÃO DE ERROS DE VÍDEO")
    print(f"Total de arquivos: {len(video_files)}")
    print("=" * 80)
    
    results = {'correct': 0, 'wrong': 0, 'details': []}
    
    for i, video_path in enumerate(sorted(video_files), 1):
        filename = os.path.basename(video_path)
        
        # Determina o erro esperado pelo nome do arquivo
        expected = None
        if 'freeze' in filename.lower():
            expected = 'freeze'
        elif 'fade' in filename.lower():
            expected = 'fade'
        elif 'foco' in filename.lower() or 'fora' in filename.lower():
            expected = 'fora_de_foco'
        
        print(f"\n[{i}/{len(video_files)}] {filename}")
        print(f"  Esperado: {expected or 'ERRO (qualquer)'}")
        
        try:
            video_class, video_conf, video_time = inference.analyze_video_frames(video_path, sample_rate_hz=2.0)
            
            # Verifica se detectou corretamente
            detected_error = video_class.lower() != 'normal'
            correct_class = expected is None or video_class.lower() == expected.lower()
            
            if detected_error and correct_class:
                status = f"✅ CORRETO: {video_class} ({video_conf:.2%})"
                results['correct'] += 1
            elif detected_error:
                status = f"⚠️ ERRO DIFERENTE: {video_class} ({video_conf:.2%}) - esperava {expected}"
                results['wrong'] += 1
            else:
                status = f"❌ NÃO DETECTOU: {video_class} ({video_conf:.2%})"
                results['wrong'] += 1
                
            results['details'].append({
                'file': filename, 'expected': expected, 
                'detected': video_class, 'conf': video_conf,
                'correct': detected_error and correct_class
            })
            
        except Exception as e:
            status = f"⚠️ ERRO: {e}"
            results['wrong'] += 1
        
        print(f"  Resultado: {status}")
    
    # Resumo
    total = len(video_files)
    print(f"\n{'='*60}")
    print(f"RESUMO VÍDEO: {results['correct']}/{total} ({results['correct']/total:.1%})")
    print("="*60)
    
    return results


def test_audio_errors():
    """Testa clipes de erro de áudio"""
    audio_dir = os.path.join(os.path.dirname(__file__), 'validade_model_audio')
    
    # Só arquivos na raiz (não entra em normal_clips/)
    audio_files = [f for f in glob.glob(os.path.join(audio_dir, '*.mp4'))]
    
    if not audio_files:
        print("Nenhum vídeo de erro de áudio encontrado")
        return
    
    print("\n" + "=" * 80)
    print("🔊 VALIDAÇÃO DE ERROS DE ÁUDIO")
    print(f"Total de arquivos: {len(audio_files)}")
    print("=" * 80)
    
    results = {'correct': 0, 'wrong': 0, 'details': []}
    
    for i, audio_path in enumerate(sorted(audio_files), 1):
        filename = os.path.basename(audio_path)
        
        # Determina o erro esperado pelo nome do arquivo
        expected = None
        if 'eco' in filename.lower():
            expected = 'eco_reverb'
        elif 'mudo' in filename.lower() or 'mute' in filename.lower():
            expected = 'ausencia_audio'
        elif 'hiss' in filename.lower():
            expected = 'ruido_hiss'
        elif 'sinal' in filename.lower() or 'erro' in filename.lower():
            expected = 'sinal_teste'
        
        print(f"\n[{i}/{len(audio_files)}] {filename}")
        print(f"  Esperado: {expected or 'ERRO (qualquer)'}")
        
        try:
            audio_class, audio_conf, audio_start, audio_end = inference.analyze_audio_segments(audio_path)
            
            # Verifica se detectou corretamente
            detected_error = audio_class.lower() != 'normal'
            correct_class = expected is None or audio_class.lower() == expected.lower()
            
            start_str = f"{audio_start}s" if audio_start else "N/A"
            end_str = f"{audio_end}s" if audio_end else "N/A"
            
            if detected_error and correct_class:
                status = f"✅ CORRETO: {audio_class} ({audio_conf:.2%}) @ {start_str}-{end_str}"
                results['correct'] += 1
            elif detected_error:
                status = f"⚠️ ERRO DIFERENTE: {audio_class} ({audio_conf:.2%}) - esperava {expected}"
                results['wrong'] += 1
            else:
                status = f"❌ NÃO DETECTOU: {audio_class} ({audio_conf:.2%})"
                results['wrong'] += 1
                
            results['details'].append({
                'file': filename, 'expected': expected, 
                'detected': audio_class, 'conf': audio_conf,
                'correct': detected_error and correct_class
            })
            
        except Exception as e:
            status = f"⚠️ ERRO: {e}"
            results['wrong'] += 1
        
        print(f"  Resultado: {status}")
    
    # Resumo
    total = len(audio_files)
    print(f"\n{'='*60}")
    print(f"RESUMO ÁUDIO: {results['correct']}/{total} ({results['correct']/total:.1%})")
    print("="*60)
    
    return results


def main():
    video_results = test_video_errors()
    audio_results = test_audio_errors()
    
    print("\n" + "=" * 80)
    print("📊 RESUMO FINAL - DETECÇÃO DE ERROS")
    print("=" * 80)
    
    if video_results:
        v_total = video_results['correct'] + video_results['wrong']
        print(f"Vídeo: {video_results['correct']}/{v_total} ({video_results['correct']/v_total:.1%})")
    
    if audio_results:
        a_total = audio_results['correct'] + audio_results['wrong']
        print(f"Áudio: {audio_results['correct']}/{a_total} ({audio_results['correct']/a_total:.1%})")


if __name__ == '__main__':
    main()
