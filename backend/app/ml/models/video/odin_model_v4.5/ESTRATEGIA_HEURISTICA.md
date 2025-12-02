# 🧠 HORUS V4.5 - Estratégia Híbrida: Heurística + Modelo

## Visão Geral

O modelo V4.5 utiliza uma **estratégia de ensemble** que combina a precisão do Deep Learning com a confiabilidade de heurísticas clássicas de visão computacional. Isso garante que falhas "óbvias" nunca escapem, mesmo que o modelo tenha dúvidas.

---

## 🔧 Thresholds Calibrados (Valores Finais)

| Heurística              | Threshold           | Descrição                                                                             |
| :---------------------- | :------------------ | :------------------------------------------------------------------------------------ |
| **Freeze**              | `diff < 2.0`        | Diferença média de pixels entre frames consecutivos                                   |
| **Fade**                | `brightness < 15`   | Brilho médio dos frames (escala 0-255)                                                |
| **Blur (Fora de Foco)** | `sharpness < 130.0` | Variância do Laplaciano (nitidez)                                                     |
| **Override Threshold**  | `model_conf < 0.95` | Se o modelo tiver menos de 95% de certeza em "Normal", a heurística pode sobrescrever |

---

## 📐 Heurísticas Implementadas

### 1. Detecção de Congelamento (Freeze)

```python
def _check_freeze(self, frames: np.ndarray) -> Tuple[bool, float]:
    """
    Detecta congelamento comparando a diferença média entre frames consecutivos.
    Se os frames são quase idênticos (diff < 2.0), é um freeze.
    """
    if len(frames) < 2:
        return False, 0.0

    diffs = []
    for i in range(1, len(frames)):
        diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
        diffs.append(diff)

    avg_diff = np.mean(diffs)
    is_freeze = avg_diff < 2.0  # THRESHOLD CALIBRADO
    conf = 1.0 if is_freeze else 0.0

    return is_freeze, conf
```

**Lógica:** Se a diferença média entre frames consecutivos for menor que 2 (em escala 0-255), significa que a imagem está "parada".

---

### 2. Detecção de Fade (Tela Preta)

```python
def _check_fade(self, frames: np.ndarray) -> Tuple[bool, float]:
    """
    Detecta fade/tela preta calculando o brilho médio dos frames.
    Se o brilho for muito baixo (< 15), é um fade.
    """
    if len(frames) < 1:
        return False, 0.0

    brightnesses = [np.mean(f) for f in frames]
    avg_brightness = np.mean(brightnesses)

    is_fade = avg_brightness < 15  # THRESHOLD CALIBRADO
    conf = 1.0 if is_fade else 0.0

    return is_fade, conf
```

**Lógica:** Se o brilho médio dos pixels for menor que 15 (em escala 0-255), a tela está praticamente preta.

---

### 3. Detecção de Desfoque (Fora de Foco / Blur)

```python
def _check_blur(self, frames: np.ndarray) -> Tuple[bool, float]:
    """
    Detecta desfoque usando a variância do operador Laplaciano.
    Imagens nítidas têm alta variância (bordas definidas).
    Imagens borradas têm baixa variância (bordas suaves).
    """
    import cv2
    if len(frames) < 1:
        return False, 0.0

    sharpness_values = []
    for frame in frames:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Variância do Laplaciano = medida de nitidez
        sharpness_values.append(cv2.Laplacian(gray, cv2.CV_64F).var())

    avg_sharpness = np.mean(sharpness_values)

    is_blur = avg_sharpness < 130.0  # THRESHOLD CALIBRADO
    conf = min(0.99, 1.0 - (avg_sharpness / 130.0)) if is_blur else 0.0

    return is_blur, conf
```

**Lógica:** O operador Laplaciano destaca bordas. Se a variância for baixa (< 130), significa que não há bordas definidas = imagem borrada.

---

## 🎯 Lógica de Decisão (Ensemble)

```python
# 1. Rodar o modelo de Deep Learning
model_class = max(probs, key=probs.get)  # Classe com maior probabilidade
model_conf = probs[model_class]          # Confiança do modelo

# 2. Rodar as heurísticas
is_freeze, freeze_conf = self._check_freeze(frames)
is_fade, fade_conf = self._check_fade(frames)
is_blur, blur_conf = self._check_blur(frames)

# 3. Decisão final
final_class = model_class
final_conf = model_conf
method = "model"

# OVERRIDE: Se o modelo diz "Normal" mas está incerto (< 95%),
# e uma heurística detectou algo, confiar na heurística.
if model_class == "normal" and model_conf < 0.95:
    if is_freeze:
        final_class = "freeze"
        final_conf = freeze_conf
        method = "heuristic_override"
    elif is_fade:
        final_class = "fade"
        final_conf = fade_conf
        method = "heuristic_override"
    elif is_blur:
        final_class = "fora_de_foco"
        final_conf = blur_conf
        method = "heuristic_override"
```

---

## 📊 Resultados da Validação

| Métrica                      | Valor                   |
| :--------------------------- | :---------------------- |
| **Taxa de Detecção**         | 100% (7/7 vídeos)       |
| **Precisão Temporal (IoU)**  | 65.2%                   |
| **Erro Médio (Início)**      | ±0.9s                   |
| **Erro Médio (Fim)**         | ±1.1s                   |
| **Especificidade (Normais)** | 80% (4/5 clipes limpos) |

**Nota:** O único "falso positivo" foi um efeito artístico de bokeh (fundo desfocado intencional), detectado por apenas 1.7s. A regra de negócio de "reportar apenas erros > 2s" filtra isso automaticamente.

---

## 💡 Quando Usar Cada Componente

| Situação                                    | Quem Decide                                       |
| :------------------------------------------ | :------------------------------------------------ |
| Modelo tem alta confiança (≥ 95%)           | **Modelo**                                        |
| Modelo incerto + Heurística detecta algo    | **Heurística** (override)                         |
| Modelo diz "erro" + Heurística diz "normal" | **Modelo** (heurísticas podem perder casos sutis) |

---

## 🚀 Recomendação para Produção

Adicionar uma **regra de negócio** no sistema HORUS:

- Só reportar erros que **persistam por mais de 2 segundos**.
- Isso elimina falsos positivos causados por efeitos artísticos curtos (transições, bokeh, etc.).
