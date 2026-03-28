# Recherche: TransNet V2 auf Apple Silicon

## Ziel
Wir wollen TransNet V2 (Shot Boundary Detection) auf Apple Silicon (M1/M2) nutzen — ohne TensorFlow.

## Hintergrund
- TransNet V2 ist ein 3D-CNN Modell (dilated convolutions), kein Transformer
- Input: 100 Frames x 48x27 Pixel RGB (Sliding Window)
- Output: 100 Wahrscheinlichkeiten (Schnitt ja/nein pro Frame)
- Modell ist winzig (wenige MB)
- Original-Implementierung: TensorFlow (problematisch auf macOS, tensorflow-macos deprecated)
- GitHub Original: https://github.com/soCzech/TransNetV2

## Fragen zu klären

### 1. ONNX-Export
- Gibt es einen fertigen ONNX-Export des TransNet V2 Modells?
- Falls nein: Lässt sich das TF-Modell nach ONNX konvertieren (tf2onnx)?
- Welche ONNX opset version wird benötigt?

### 2. PyTorch-Port
- Gibt es einen PyTorch-Port von TransNet V2?
- Wie vollständig/getestet ist er?
- Gibt es vortrainierte Weights im PyTorch-Format?

### 3. Inference auf Apple Silicon
- `onnxruntime` auf macOS ARM64: Gibt es einen CoreML ExecutionProvider oder läuft es auf CPU?
- Falls PyTorch: Läuft das Modell auf MPS (Metal Performance Shaders)?
- Performance-Abschätzung: Wie schnell kann ein 90-min Film (24fps, ~130.000 Frames) verarbeitet werden?
  - Auf CPU?
  - Auf MPS/CoreML?

### 4. Dependencies
- Welche Python-Packages werden benötigt (mit exakten Versionen)?
- Gibt es Wheels für macOS ARM64 (aarch64/arm64)?
- Kann es standalone laufen ohne TensorFlow?

### 5. Alternativen falls TransNet V2 nicht geht
- Gibt es andere DL-basierte SBD-Modelle die ONNX/PyTorch-nativ sind?
- PySceneDetect AdaptiveDetector als Fallback — wie gut im Vergleich?

## Randbedingungen
- macOS Apple Silicon ONLY (M1 Max dev, M2 Ultra prod)
- Kein CUDA, kein ROCm, kein Windows, kein Linux
- Kein TensorFlow — zu fragil auf macOS
- Originale Modell-Weights (kein Quantizing)
- Darf nicht OOM crashen bei langen Filmen (2h+)
- Ergebnis soll eine klare Empfehlung sein: welcher Weg, welche Packages, welche Versionen
