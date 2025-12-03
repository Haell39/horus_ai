#!/usr/bin/env python
"""
Script de validação dos modelos de vídeo e áudio.

Testa os modelos contra:
1. validate_model_video/ - clips com erros de vídeo (freeze, fade, fora_de_foco)
2. validade_model_audio/ - clips com erros de áudio (mudo, echo, hiss, sinal_teste)  
3. validate_normal/ - clips normais (devem detectar "normal")

Relatório final mostra acurácia por classe.
"""

import os
import sys

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Import inference module
from app.ml import inference


# ==============================================================================
# Configuração dos testes
# ==============================================================================

BASE_DIR = Path(__file__).parent.parent

# Pastas de validação
VIDEO_ERROR_DIR = BASE_DIR / "validate_model_video"
AUDIO_ERROR_DIR = BASE_DIR / "validade_model_audio"
NORMAL_DIR = BASE_DIR / "validate_normal"

# Mapeamento de nomes de arquivos para classes esperadas (vídeo)
VIDEO_EXPECTED_MAP = {
    "fade": "fade",
    "freeze": "freeze",
    "fora_de_foco": "fora_de_foco",
    "fora-foco": "fora_de_foco",
    "fora_foco": "fora_de_foco",
}

# Mapeamento de nomes de arquivos para classes esperadas (áudio)
AUDIO_EXPECTED_MAP = {
    "mudo": "ausencia_audio",
    "silencio": "ausencia_audio",
    "echo": "eco_reverb",
    "eco": "eco_reverb",
    "reverb": "eco_reverb",
    "hiss": "ruido_hiss",
    "ruido": "ruido_hiss",
    "sinalerro": "sinal_teste",
    "sinal_teste": "sinal_teste",
}


def get_expected_class_from_filename(filename: str, mapping: Dict[str, str]) -> str:
    """Detecta a classe esperada a partir do nome do arquivo."""
    filename_lower = filename.lower()
    for key, expected_class in mapping.items():
        if key in filename_lower:
            return expected_class
    return "unknown"


def run_video_validation() -> Tuple[List[dict], int, int]:
    """Valida o modelo de vídeo contra clips com erros de vídeo."""
    results = []
    correct = 0
    total = 0
    
    if not VIDEO_ERROR_DIR.exists():
        print(f"⚠️  Pasta não encontrada: {VIDEO_ERROR_DIR}")
        return results, correct, total
    
    print("\n" + "="*70)
    print("📹 VALIDAÇÃO DO MODELO DE VÍDEO (Odin v4.5)")
    print("="*70)
    
    # 1. Testar clips com erros de vídeo
    print("\n--- Clips com ERROS de Vídeo ---")
    for video_file in VIDEO_ERROR_DIR.glob("*.mp4"):
        expected = get_expected_class_from_filename(video_file.name, VIDEO_EXPECTED_MAP)
        if expected == "unknown":
            print(f"⚠️  {video_file.name}: Classe esperada não detectada, pulando...")
            continue
        
        total += 1
        # analyze_video_frames retorna (pred_class, confidence, event_time_s)
        final_class, confidence, _ = inference.analyze_video_frames(str(video_file))
        
        is_correct = (final_class == expected)
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} {video_file.name}")
        print(f"   Esperado: {expected} | Detectado: {final_class} ({confidence:.2%})")
        
        results.append({
            "file": video_file.name,
            "type": "video_error",
            "expected": expected,
            "detected": final_class,
            "confidence": confidence,
            "correct": is_correct,
        })
    
    # 2. Testar clips normais (devem ser "normal")
    normal_clips_dir = VIDEO_ERROR_DIR / "normal_clips"
    if normal_clips_dir.exists():
        print("\n--- Clips NORMAIS (dentro de validate_model_video) ---")
        for video_file in normal_clips_dir.glob("*.mp4"):
            total += 1
            expected = "normal"
            
            final_class, confidence, _ = inference.analyze_video_frames(str(video_file))
            
            is_correct = (final_class == expected)
            if is_correct:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{status} {video_file.name}")
            print(f"   Esperado: {expected} | Detectado: {final_class} ({confidence:.2%})")
            
            results.append({
                "file": f"normal_clips/{video_file.name}",
                "type": "video_normal",
                "expected": expected,
                "detected": final_class,
                "confidence": confidence,
                "correct": is_correct,
            })
    
    return results, correct, total


def run_audio_validation() -> Tuple[List[dict], int, int]:
    """Valida o modelo de áudio contra clips com erros de áudio."""
    results = []
    correct = 0
    total = 0
    
    if not AUDIO_ERROR_DIR.exists():
        print(f"⚠️  Pasta não encontrada: {AUDIO_ERROR_DIR}")
        return results, correct, total
    
    print("\n" + "="*70)
    print("🔊 VALIDAÇÃO DO MODELO DE ÁUDIO (Heimdall Ultra V1)")
    print("="*70)
    
    # 1. Testar clips com erros de áudio
    print("\n--- Clips com ERROS de Áudio ---")
    for audio_file in AUDIO_ERROR_DIR.glob("*.mp4"):
        expected = get_expected_class_from_filename(audio_file.name, AUDIO_EXPECTED_MAP)
        if expected == "unknown":
            print(f"⚠️  {audio_file.name}: Classe esperada não detectada, pulando...")
            continue
        
        total += 1
        # analyze_audio_segments retorna (classe, confiança, tempo_início, tempo_fim)
        final_class, confidence, _, _ = inference.analyze_audio_segments(str(audio_file))
        
        is_correct = (final_class == expected)
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} {audio_file.name}")
        print(f"   Esperado: {expected} | Detectado: {final_class} ({confidence:.2%})")
        
        results.append({
            "file": audio_file.name,
            "type": "audio_error",
            "expected": expected,
            "detected": final_class,
            "confidence": confidence,
            "correct": is_correct,
        })
    
    # 2. Testar clips normais (devem ser "normal")
    normal_clips_dir = AUDIO_ERROR_DIR / "normal_clips"
    if normal_clips_dir.exists():
        print("\n--- Clips NORMAIS (dentro de validade_model_audio) ---")
        for audio_file in normal_clips_dir.glob("*.mp4"):
            total += 1
            expected = "normal"
            
            final_class, confidence, _, _ = inference.analyze_audio_segments(str(audio_file))
            
            is_correct = (final_class == expected)
            if is_correct:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{status} {audio_file.name}")
            print(f"   Esperado: {expected} | Detectado: {final_class} ({confidence:.2%})")
            
            results.append({
                "file": f"normal_clips/{audio_file.name}",
                "type": "audio_normal",
                "expected": expected,
                "detected": final_class,
                "confidence": confidence,
                "correct": is_correct,
            })
    
    return results, correct, total


def run_normal_validation() -> Tuple[List[dict], int, int, int, int]:
    """Valida clips 100% normais - ambos modelos devem classificar como 'normal'."""
    results = []
    video_correct = 0
    audio_correct = 0
    video_total = 0
    audio_total = 0
    
    if not NORMAL_DIR.exists():
        print(f"⚠️  Pasta não encontrada: {NORMAL_DIR}")
        return results, video_correct, video_total, audio_correct, audio_total
    
    print("\n" + "="*70)
    print("📺 VALIDAÇÃO DE CLIPS NORMAIS (Transmissões T1, T2, T3)")
    print("="*70)
    
    for video_file in sorted(NORMAL_DIR.glob("*.mp4")):
        print(f"\n▶️  {video_file.name}")
        
        # Teste de Vídeo
        video_total += 1
        video_class, video_conf, _ = inference.analyze_video_frames(str(video_file))
        
        video_ok = (video_class == "normal")
        if video_ok:
            video_correct += 1
            video_status = "✅"
        else:
            video_status = "❌"
        
        # Teste de Áudio
        audio_total += 1
        audio_class, audio_conf, _, _ = inference.analyze_audio_segments(str(video_file))
        
        audio_ok = (audio_class == "normal")
        if audio_ok:
            audio_correct += 1
            audio_status = "✅"
        else:
            audio_status = "❌"
        
        print(f"   Vídeo: {video_status} {video_class} ({video_conf:.2%})")
        print(f"   Áudio: {audio_status} {audio_class} ({audio_conf:.2%})")
        
        results.append({
            "file": video_file.name,
            "type": "pure_normal",
            "video_expected": "normal",
            "video_detected": video_class,
            "video_confidence": video_conf,
            "video_correct": video_ok,
            "audio_expected": "normal",
            "audio_detected": audio_class,
            "audio_confidence": audio_conf,
            "audio_correct": audio_ok,
        })
    
    return results, video_correct, video_total, audio_correct, audio_total


# Pasta de teste do lipsync
LIPSYNC_TEST_DIR = BASE_DIR / "test_lipsync"


def run_lipsync_validation() -> List[dict]:
    """Testa o modelo de lipsync contra clips de teste."""
    results = []
    
    if not LIPSYNC_TEST_DIR.exists():
        print(f"\n⚠️  Pasta de lipsync não encontrada: {LIPSYNC_TEST_DIR}")
        return results
    
    lipsync_files = list(LIPSYNC_TEST_DIR.glob("*.mp4"))
    if not lipsync_files:
        print(f"\n⚠️  Nenhum vídeo encontrado em: {LIPSYNC_TEST_DIR}")
        return results
    
    print("\n" + "="*70)
    print("👄 VALIDAÇÃO DO MODELO DE LIPSYNC (SyncNet v2)")
    print("="*70)
    
    if not inference.is_lipsync_model_loaded():
        print("⚠️  Modelo de Lipsync não está carregado!")
        return results
    
    print(f"\nClasses: {inference.LIPSYNC_CLASSES}")
    print("\n--- Clips de Teste de Lipsync ---")
    
    for video_file in sorted(lipsync_files):
        print(f"\n▶️  {video_file.name}")
        
        try:
            # analyze_lipsync retorna (status, confidence, offset_ms)
            status, confidence, offset_ms = inference.analyze_lipsync(str(video_file))
            
            print(f"   Status: {status}")
            print(f"   Confiança: {confidence:.2%}")
            print(f"   Offset estimado: {offset_ms:.1f} ms")
            
            results.append({
                "file": video_file.name,
                "status": status,
                "confidence": confidence,
                "offset_ms": offset_ms,
            })
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            results.append({
                "file": video_file.name,
                "status": "error",
                "error": str(e),
            })
    
    return results


def print_summary(
    video_results: List[dict],
    video_correct: int,
    video_total: int,
    audio_results: List[dict],
    audio_correct: int,
    audio_total: int,
    normal_results: List[dict],
    normal_video_correct: int,
    normal_video_total: int,
    normal_audio_correct: int,
    normal_audio_total: int,
):
    """Imprime resumo final da validação."""
    print("\n" + "="*70)
    print("📊 RESUMO DA VALIDAÇÃO")
    print("="*70)
    
    # Vídeo (erros + normais internos)
    total_video_correct = video_correct + normal_video_correct
    total_video_total = video_total + normal_video_total
    video_acc = (total_video_correct / total_video_total * 100) if total_video_total > 0 else 0
    
    print(f"\n📹 MODELO DE VÍDEO (Odin v4.5)")
    print(f"   - Pasta validate_model_video/: {video_correct}/{video_total}")
    print(f"   - Pasta validate_normal/:      {normal_video_correct}/{normal_video_total}")
    print(f"   - TOTAL: {total_video_correct}/{total_video_total} ({video_acc:.1f}%)")
    
    # Áudio (erros + normais internos)
    total_audio_correct = audio_correct + normal_audio_correct
    total_audio_total = audio_total + normal_audio_total
    audio_acc = (total_audio_correct / total_audio_total * 100) if total_audio_total > 0 else 0
    
    print(f"\n🔊 MODELO DE ÁUDIO (Heimdall Ultra V1)")
    print(f"   - Pasta validade_model_audio/: {audio_correct}/{audio_total}")
    print(f"   - Pasta validate_normal/:      {normal_audio_correct}/{normal_audio_total}")
    print(f"   - TOTAL: {total_audio_correct}/{total_audio_total} ({audio_acc:.1f}%)")
    
    # Geral
    grand_total_correct = total_video_correct + total_audio_correct
    grand_total = total_video_total + total_audio_total
    overall_acc = (grand_total_correct / grand_total * 100) if grand_total > 0 else 0
    
    print(f"\n🎯 ACURÁCIA GERAL: {grand_total_correct}/{grand_total} ({overall_acc:.1f}%)")
    
    # Erros por classe (vídeo)
    print("\n" + "-"*40)
    print("❌ ERROS DETALHADOS (Vídeo):")
    video_errors = [r for r in video_results if not r.get("correct", True)]
    if video_errors:
        for err in video_errors:
            print(f"   {err['file']}: esperado={err['expected']}, detectado={err['detected']}")
    else:
        print("   Nenhum erro!")
    
    # Erros por classe (áudio)
    print("\n❌ ERROS DETALHADOS (Áudio):")
    audio_errors = [r for r in audio_results if not r.get("correct", True)]
    if audio_errors:
        for err in audio_errors:
            print(f"   {err['file']}: esperado={err['expected']}, detectado={err['detected']}")
    else:
        print("   Nenhum erro!")
    
    # Erros nos clips normais
    print("\n❌ ERROS EM CLIPS NORMAIS:")
    normal_errors = [r for r in normal_results if not r.get("video_correct", True) or not r.get("audio_correct", True)]
    if normal_errors:
        for err in normal_errors:
            if not err.get("video_correct", True):
                print(f"   {err['file']} (vídeo): esperado=normal, detectado={err['video_detected']}")
            if not err.get("audio_correct", True):
                print(f"   {err['file']} (áudio): esperado=normal, detectado={err['audio_detected']}")
    else:
        print("   Nenhum erro!")
    
    print("\n" + "="*70)


def main():
    """Executa a validação completa."""
    print("\n" + "#"*70)
    print("#  VALIDAÇÃO DOS MODELOS DE ML - Horus AI")
    print(f"#  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*70)
    
    # Carregar modelos
    print("\n⏳ Carregando modelos...")
    inference.load_all_models()
    print("✅ Modelos carregados!")
    
    # Rodar validações
    video_results, video_correct, video_total = run_video_validation()
    audio_results, audio_correct, audio_total = run_audio_validation()
    normal_results, normal_video_correct, normal_video_total, normal_audio_correct, normal_audio_total = run_normal_validation()
    lipsync_results = run_lipsync_validation()
    
    # Resumo
    print_summary(
        video_results, video_correct, video_total,
        audio_results, audio_correct, audio_total,
        normal_results, normal_video_correct, normal_video_total, normal_audio_correct, normal_audio_total,
    )
    
    # Salvar resultados em JSON
    output_file = BASE_DIR / "validation_results.json"
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "video_validation": video_results,
        "audio_validation": audio_results,
        "normal_validation": normal_results,
        "lipsync_validation": lipsync_results,
        "summary": {
            "video_accuracy": (video_correct + normal_video_correct) / (video_total + normal_video_total) if (video_total + normal_video_total) > 0 else 0,
            "audio_accuracy": (audio_correct + normal_audio_correct) / (audio_total + normal_audio_total) if (audio_total + normal_audio_total) > 0 else 0,
        }
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados salvos em: {output_file}")


if __name__ == "__main__":
    main()
