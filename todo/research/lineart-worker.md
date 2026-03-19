# Research: Lineart Worker Dependencies

## Installation
`pip install controlnet-aux` — enthält TEEDDetector (AnyLine-Basis).

## Standalone Usage
```python
from controlnet_aux import TEEDdetector
from PIL import Image

teed = TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth")
result = teed(Image.open("input.png"))
```

## Modell
- HuggingFace: `fal-ai/teed`, Datei `5_model.pth`
- 58K Parameter — winzig

## Minimale Alternative (ohne controlnet-aux)
TEED-Repo `xavysp/TEED` extrahieren: `ted.py` + `utils/AF/`. Braucht nur torch + numpy + opencv.

## Canny Fallback
Nur `opencv-python-headless`, kein ML-Modell:
```python
import cv2
edges = cv2.Canny(image, 100, 200)
```

## Conda Env
- Python 3.12
- torch (ARM64 MPS)
- controlnet-aux
- opencv-python-headless
