# Research: Sketch/HED Worker Dependencies

## Installation
`pip install opencv-python-headless` — kein PyTorch nötig!

## Option A: OpenCV DNN (empfohlen, minimalistisch)
```python
import cv2

# Einmalig: CropLayer registrieren (~15 Zeilen)
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel")
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H), mean=(104.00698793, 116.66876762, 122.67891434))
net.setInput(blob)
edges = net.forward()
```

- Läuft auf CPU via OpenCV — schnell genug für Einzelbilder (sub-second auf M1/M2)
- Kein MPS/GPU nötig

## Modell
- `hed_pretrained_bsds.caffemodel` (~56MB) + `deploy.prototxt`
- Original: https://vcl.ucsd.edu/hed/

## Option B: controlnet_aux (falls PyTorch eh vorhanden)
```python
from controlnet_aux import HEDdetector
hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
result = hed(image)
```
- Modell: `ControlNetHED.pth` (~29.4MB)
- Zieht aber viele Deps rein (timm, scipy, scikit-image, einops)

## Conda Env (minimal)
- Python 3.12
- opencv-python-headless
- Kein torch, kein torchvision
