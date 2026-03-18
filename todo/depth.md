# Depth Map Engine

## Engine
`generate.py image --engine depth --images input.png -o depth.png`

## Modell
Apple Depth Pro (Oct 2024) — schnell, präzise, Apple Silicon optimiert.

## Aufgaben
- [ ] Depth Pro Repo klonen / installieren
- [ ] Eigenes Conda-Env oder ins flux2-Env integrieren
- [ ] Wrapper in `worker/depth/generate.py` oder `worker/image/` erweitern
- [ ] setup.sh, install.sh, backup, requirements.lock
- [ ] Tests schreiben
- [ ] README updaten
- [ ] Lizenz-Datei in ./licenses/

## Notizen
- Stand in der mflux-Tabelle, ist aber ein eigenständiges Apple-Modell
- Kein ControlNet nötig — eigenständige Depth-Estimation
- Kann als ControlNet-Input für FLUX.2 dienen (wie OpenPose)
