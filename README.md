# BiometricSyncAnalyzer v1.0

**Pipeline de análisis biométrico multimodal sincronizado — rostro + cuerpo**
Gabriel Cao Di Marco & Daniela Di Marco | Proyecto Voyager | 2026

---

## Descripción

BiometricSyncAnalyzer es un módulo de análisis biométrico multimodal que integra
biomecánica articular (BioAngles v1.2) con biometría facial (MediaPipe Face Mesh)
sobre el mismo instante temporal de un video de cuerpo entero en movimiento.

El problema que resuelve: en videos de movimiento corporal dinámico, el rostro
no puede analizarse en frames de alta velocidad (motion blur, tamaño insuficiente).
El módulo usa las fases de marcha detectadas por BioAngles como selector de frames
de baja aceleración corporal, aplicando análisis facial únicamente en esos instantes.

Analogía conceptual: como la cámara lenta en fútbol que revela lo que a velocidad
real es invisible — los frames de Mid Stance / Initial Contact son el equivalente
de la repetición lenta.

## Aporte original

- **Metodología de sincronización temporal** entre análisis articular y facial
- **Índice de Coherencia Biométrica Multimodal (ICBM)**: métrica original que
  cuantifica la convergencia estadística entre variabilidad facial y articular
  en el mismo sujeto, en los mismos frames, bajo condiciones de movimiento real
- **Crop facial adaptativo sobre video de cuerpo entero** guiado por fase de marcha
- Sin equivalente publicado en literatura biomecánica o biométrica

## Resultados iniciales (01/04/2026)

Dataset: 56 videos corporales de Daniela Di Marco (Proyecto Voyager)
- 30 videos con análisis multimodal válido (n≥3 frames sincronizados)
- ICBM global: 0.9293 (media) | 0.9452 (mediana)
- 23/30 videos (77%) con coherencia muy alta (ICBM ≥ 0.90)
- ratio_pomulos entre dominios: CV=4.14% — misma identidad facial en cuerpo entero
- ángulo nariz-labio: CV=1.47% — eje medio-sagital invariante bajo movimiento
- 615 frames sincronizados cara+cuerpo en el mismo instante temporal

## Dependencias

- Python 3.10+
- mediapipe >= 0.10.13
- opencv-python >= 4.8
- numpy >= 1.24
- pandas >= 2.0
- BioAngles v1.2 (CSV con columna fase_marcha)

## Uso

```bash
# Video individual
python3 biometric_sync_analyzer.py video.mp4 \
  --csv outputs/video_angulos.csv -o sync_output/

# Batch completo
python3 biometric_sync_analyzer.py --batch VIDS/ \
  --csv-dir RESULTS/03_Dataset/ -o RESULTS/05_Sync/
```

## Autores

- Gabriel Cao Di Marco, MD, PhD — CAECIHS, UAI-CONICET, Buenos Aires
- Daniela Di Marco — Proyecto Voyager / Operation Knowledge

## Licencia

Copyright © 2026 Gabriel Cao Di Marco & Daniela Di Marco
Todos los derechos reservados. Registro DNDA Argentina + US Copyright Office.
