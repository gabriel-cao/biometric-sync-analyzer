#!/usr/bin/env python3
"""
BiometricSyncAnalyzer v1.0
==========================
Pipeline integrado de análisis biométrico multimodal para Daniela Di Marco.
Proyecto Voyager — Gabriel Cao Di Marco & Daniela Di Marco — 2026

CONCEPTO:
  En videos de cuerpo entero en movimiento, el rostro no puede analizarse
  en frames de alta velocidad corporal (motion blur facial, tamaño insuficiente).
  Este módulo resuelve ese problema usando la fase de marcha detectada por
  BioAngles como selector de frames estables, y aplicando análisis facial
  únicamente en esos instantes de baja aceleración corporal.

  Analogía: como la cámara lenta en fútbol que revela lo que a velocidad
  real es invisible — aquí los frames de Mid Stance / Initial Contact son
  el equivalente de la repetición a cámara lenta.

PIPELINE:
  1. BioAngles procesa el video → CSV con ángulos + fase de marcha por frame
  2. BiometricSyncAnalyzer lee ese CSV → selecciona frames en fases estables
  3. Sobre esos frames → crop facial adaptativo (ROI cabeza estimada)
  4. MediaPipe Face Mesh sobre el crop → métricas biométricas óseas
  5. Output: vector biométrico multimodal (articular + facial) por video
  6. Análisis estadístico de coherencia entre dominios

FASES ESTABLES PARA ANÁLISIS FACIAL:
  - Mid Stance (D/I)       → apoyo completo, mínima velocidad
  - Initial Contact (D/I)  → talón en suelo, cuerpo casi quieto
  - Loading Response (D/I) → carga lenta de peso

MÉTRICAS FACIALES (solo óseas — inmutables al movimiento):
  - ratio_pomulos          → anchura zigomática (CV=1.94% en análisis previo)
  - angulo_mandibular_deg  → geometría arco mandibular
  - simetria_facial        → simetría bilateral (CV=2.90%)
  - angulo_nariz_labio_deg → perfil medio-sagital (CV=0.93%)

Autor: Daniela Di Marco — Operation Knowledge / Proyecto Voyager
Fecha: 01/04/2026
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

# Fases de marcha consideradas estables para análisis facial
FASES_ESTABLES = {
    "Mid Stance (D)", "Mid Stance (I)",
    "Initial Contact (D)", "Initial Contact (I)",
    "Loading Response (D)", "Loading Response (I)",
}

# Landmarks MediaPipe Face Mesh
OJO_IZQ_INTERNO  = 133
OJO_DER_INTERNO  = 362
OJO_IZQ_EXTERNO  = 33
OJO_DER_EXTERNO  = 263
PUNTA_NARIZ      = 1
RAIZ_NASAL       = 6
MENTON           = 152
POMULO_IZQ       = 234
POMULO_DER       = 454
MANDIBULA_IZQ    = 172
MANDIBULA_DER    = 397
LABIO_CENTRO_SUP = 0
FRENTE           = 10

# Umbral mínimo distancia interocular en el crop (px)
MIN_INTEROCULAR_CROP = 25

# Margen del crop facial respecto al bounding box de cabeza estimado
CROP_MARGIN = 0.25

# Fracción del frame que se estima corresponde a la cabeza
# (en un plano general de cuerpo entero vertical)
HEAD_FRACTION_TOP    = 0.0   # la cabeza empieza en el top del frame
HEAD_FRACTION_HEIGHT = 0.28  # ocupa aprox el 28% superior del frame


# ─────────────────────────────────────────────
# FUNCIONES GEOMÉTRICAS
# ─────────────────────────────────────────────

def distancia(p1, p2, w: int, h: int) -> float:
    return np.sqrt(((p1.x - p2.x) * w) ** 2 + ((p1.y - p2.y) * h) ** 2)


def angulo_tres_puntos(p1, p2, p3, w: int, h: int) -> float:
    a = np.array([p1.x * w, p1.y * h])
    b = np.array([p2.x * w, p2.y * h])
    c = np.array([p3.x * w, p3.y * h])
    v1, v2 = a - b, c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def simetria_bilateral(lm, w: int, h: int) -> float:
    pares = [
        (OJO_IZQ_INTERNO, OJO_DER_INTERNO),
        (OJO_IZQ_EXTERNO, OJO_DER_EXTERNO),
        (POMULO_IZQ,       POMULO_DER),
        (MANDIBULA_IZQ,    MANDIBULA_DER),
    ]
    eje_x = lm[PUNTA_NARIZ].x
    ratios = []
    for li, ri in pares:
        di = abs(lm[li].x - eje_x)
        dr = abs(lm[ri].x - eje_x)
        if max(di, dr) > 1e-6:
            ratios.append(min(di, dr) / max(di, dr))
    return float(np.mean(ratios)) if ratios else 0.0


# ─────────────────────────────────────────────
# CROP FACIAL ADAPTATIVO
# ─────────────────────────────────────────────

def extraer_crop_cabeza(frame: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    """
    Estima la región de la cabeza en un frame de cuerpo entero.
    Usa posición estimada: franja superior del frame con margen.
    Devuelve (crop, (x1, y1, x2, y2)).
    """
    h, w = frame.shape[:2]
    y1 = int(h * HEAD_FRACTION_TOP)
    y2 = int(h * (HEAD_FRACTION_TOP + HEAD_FRACTION_HEIGHT))
    # Centrar horizontalmente con margen lateral
    margin_x = int(w * 0.1)
    x1, x2 = margin_x, w - margin_x
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def extraer_metricas_faciales(crop: np.ndarray,
                               face_mesh) -> Optional[Dict]:
    """
    Aplica MediaPipe Face Mesh sobre el crop y extrae métricas óseas.
    Devuelve None si no detecta cara o distancia interocular insuficiente.
    """
    h, w = crop.shape[:2]
    if h < 30 or w < 30:
        return None

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark
    d_io = distancia(lm[OJO_IZQ_INTERNO], lm[OJO_DER_INTERNO], w, h)

    if d_io < MIN_INTEROCULAR_CROP:
        return None

    ratio_pomulos = distancia(lm[POMULO_IZQ], lm[POMULO_DER], w, h) / d_io
    angulo_mand   = angulo_tres_puntos(lm[MANDIBULA_IZQ], lm[MENTON],
                                        lm[MANDIBULA_DER], w, h)
    simetria      = simetria_bilateral(lm, w, h)
    ang_nz_lb     = angulo_tres_puntos(lm[PUNTA_NARIZ], lm[LABIO_CENTRO_SUP],
                                        lm[MENTON], w, h)

    return {
        "d_interocular_crop_px": round(d_io, 2),
        "ratio_pomulos":         round(ratio_pomulos, 4),
        "angulo_mandibular_deg": round(angulo_mand, 2),
        "simetria_facial":       round(simetria, 4),
        "angulo_nariz_labio":    round(ang_nz_lb, 2),
    }


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def analizar_video(video_path: str,
                   csv_bioangles: str,
                   output_dir: str,
                   imagen_referencia: Optional[str] = None,
                   verbose: bool = True) -> Optional[Dict]:
    """
    Pipeline completo para un video:
      1. Lee CSV de BioAngles → identifica frames estables
      2. Extrae esos frames del video
      3. Aplica crop + análisis facial
      4. Calcula estadísticas y vector biométrico multimodal
    """
    video_path = Path(video_path)
    csv_path   = Path(csv_bioangles)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"  ERROR: Video no encontrado: {video_path}")
        return None
    if not csv_path.exists():
        print(f"  ERROR: CSV BioAngles no encontrado: {csv_path}")
        return None

    # Leer CSV BioAngles
    df = pd.read_csv(csv_path)
    if "fase_marcha" not in df.columns:
        print(f"  ERROR: El CSV no tiene columna 'fase_marcha'. "
              f"Requiere BioAngles v1.2+")
        return None

    # Filtrar frames estables
    df_estables = df[df["fase_marcha"].isin(FASES_ESTABLES)].copy()
    if len(df_estables) == 0:
        print(f"  AVISO: No hay frames en fases estables para {video_path.name}")
        return None

    frames_estables = set(df_estables["frame"].tolist())
    if verbose:
        print(f"  Frames totales: {len(df)} | Frames estables: "
              f"{len(frames_estables)} ({100*len(frames_estables)/len(df):.1f}%)")
        fases_cnt = df_estables["fase_marcha"].value_counts()
        for fase, cnt in fases_cnt.items():
            print(f"    {fase}: {cnt}")

    # Inicializar Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    resultados_faciales = []

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as face_mesh:

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            if frame_num not in frames_estables:
                continue

            # Obtener fase y ángulos articulares del frame
            row = df_estables[df_estables["frame"] == frame_num].iloc[0]

            # Crop cabeza
            crop, bbox = extraer_crop_cabeza(frame)

            # Análisis facial
            metricas = extraer_metricas_faciales(crop, face_mesh)

            if metricas is not None:
                metricas["frame"]         = frame_num
                metricas["tiempo_s"]      = round(frame_num / fps, 3) if fps > 0 else 0
                metricas["fase_marcha"]   = row["fase_marcha"]
                metricas["confidence_corporal"] = row.get("confidence", None)
                # Incluir métricas articulares del mismo frame
                for col in ["rodilla_L_flexion", "rodilla_R_flexion",
                            "cadera_L_flexion",  "cadera_R_flexion",
                            "tronco_inclinacion"]:
                    if col in row:
                        metricas[f"bio_{col}"] = row[col]
                resultados_faciales.append(metricas)

    cap.release()

    if not resultados_faciales:
        print(f"  AVISO: No se detectó rostro en frames estables de {video_path.name}")
        return None

    return resultados_faciales, df, df_estables, video_path.stem, output_dir, fps


# ─────────────────────────────────────────────
# ESTADÍSTICAS Y REPORTE
# ─────────────────────────────────────────────

def calcular_estadisticas(resultados_faciales: List[Dict],
                           df_bio: pd.DataFrame,
                           nombre_base: str,
                           output_dir: Path) -> Dict:
    """
    Calcula estadísticas del vector biométrico multimodal y
    genera el Índice de Coherencia Biométrica Multimodal (ICBM).
    """
    df_facial = pd.DataFrame(resultados_faciales)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Guardar CSV
    csv_path = output_dir / f"{nombre_base}_sync_facial.csv"
    df_facial.to_csv(csv_path, index=False)

    # Métricas óseas faciales
    metricas_oseas = ["ratio_pomulos", "angulo_mandibular_deg",
                       "simetria_facial", "angulo_nariz_labio"]

    stats_facial = {}
    for col in metricas_oseas:
        if col in df_facial.columns:
            vals = df_facial[col].dropna()
            if len(vals) > 0:
                cv = (vals.std() / vals.mean() * 100) if vals.mean() != 0 else 0
                stats_facial[col] = {
                    "n":      len(vals),
                    "media":  round(float(vals.mean()), 4),
                    "std":    round(float(vals.std()),  4),
                    "cv_pct": round(float(cv),          2),
                    "min":    round(float(vals.min()),  4),
                    "max":    round(float(vals.max()),  4),
                }

    # Métricas articulares en frames estables
    metricas_bio = ["bio_rodilla_L_flexion", "bio_rodilla_R_flexion",
                     "bio_cadera_L_flexion",  "bio_cadera_R_flexion"]
    stats_bio = {}
    for col in metricas_bio:
        if col in df_facial.columns:
            vals = df_facial[col].dropna()
            if len(vals) > 0:
                cv = (vals.std() / vals.mean() * 100) if vals.mean() != 0 else 0
                stats_bio[col] = {
                    "media":  round(float(vals.mean()), 2),
                    "std":    round(float(vals.std()),  2),
                    "cv_pct": round(float(cv),          2),
                }

    # ── ÍNDICE DE COHERENCIA BIOMÉTRICA MULTIMODAL (ICBM) ──────────────
    # Principio: si la identidad es coherente, las métricas óseas de
    # ambos dominios deben mostrar CV similarmente bajos.
    # ICBM = 1 - |CV_facial_medio - CV_bio_medio| / 100
    # Rango: 0 (incoherente) → 1 (perfectamente coherente)

    cv_facial_medio = np.mean([s["cv_pct"] for s in stats_facial.values()]) \
                      if stats_facial else None
    cv_bio_medio    = np.mean([s["cv_pct"] for s in stats_bio.values()]) \
                      if stats_bio else None

    icbm = None
    if cv_facial_medio is not None and cv_bio_medio is not None:
        icbm = round(1.0 - abs(cv_facial_medio - cv_bio_medio) / 100.0, 4)
        icbm = max(0.0, min(1.0, icbm))

    interpretacion_icbm = (
        "Coherencia muy alta (identidad biométrica unificada)"  if icbm and icbm >= 0.90 else
        "Coherencia alta"                                        if icbm and icbm >= 0.80 else
        "Coherencia moderada — revisar condiciones de captura"  if icbm and icbm >= 0.65 else
        "Coherencia baja — posible varianza instrumental"        if icbm else
        "No calculable"
    )

    reporte = {
        "version":        "BiometricSyncAnalyzer_v1.0",
        "video":          nombre_base,
        "timestamp":      timestamp,
        "frames_analizados_sync": len(df_facial),
        "estadisticas_faciales":  stats_facial,
        "estadisticas_articulares_frames_estables": stats_bio,
        "ICBM": {
            "valor":            icbm,
            "cv_facial_medio":  round(float(cv_facial_medio), 2) if cv_facial_medio else None,
            "cv_bio_medio":     round(float(cv_bio_medio),    2) if cv_bio_medio    else None,
            "interpretacion":   interpretacion_icbm,
        },
        "firma_biometrica_multimodal": {
            col: {"media": s["media"], "cv_pct": s["cv_pct"]}
            for col, s in stats_facial.items()
        }
    }

    json_path = output_dir / f"{nombre_base}_sync_reporte.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    return reporte


# ─────────────────────────────────────────────
# IMPRESIÓN DE RESUMEN
# ─────────────────────────────────────────────

def imprimir_resumen(reporte: Dict):
    v = reporte["video"]
    n = reporte["frames_analizados_sync"]
    print(f"\n{'='*62}")
    print(f"  RESUMEN BIOMÉTRICO MULTIMODAL — {v}")
    print(f"{'='*62}")
    print(f"  Frames sincronizados (fase estable + rostro): {n}")

    print(f"\n  MÉTRICAS FACIALES ÓSEAS (frames estables):")
    print(f"  {'Métrica':<26} {'Media':>8} {'CV%':>7}  Consistencia")
    print(f"  {'-'*58}")
    for col, s in reporte["estadisticas_faciales"].items():
        nivel = (
            "✓✓ Muy alta" if s["cv_pct"] < 5  else
            "✓  Alta"     if s["cv_pct"] < 10 else
            "~  Moderada" if s["cv_pct"] < 20 else
            "✗  Revisar"
        )
        print(f"  {col:<26} {s['media']:>8.3f} {s['cv_pct']:>6.1f}%  {nivel}")

    print(f"\n  MÉTRICAS ARTICULARES (mismo frames estables):")
    print(f"  {'Métrica':<30} {'Media':>8} {'CV%':>7}")
    print(f"  {'-'*48}")
    for col, s in reporte["estadisticas_articulares_frames_estables"].items():
        nombre = col.replace("bio_", "")
        print(f"  {nombre:<30} {s['media']:>7.1f}° {s['cv_pct']:>6.1f}%")

    icbm = reporte["ICBM"]
    print(f"\n  ÍNDICE DE COHERENCIA BIOMÉTRICA MULTIMODAL (ICBM):")
    print(f"  Valor: {icbm['valor']}  |  CV facial: {icbm['cv_facial_medio']}%  "
          f"|  CV articular: {icbm['cv_bio_medio']}%")
    print(f"  → {icbm['interpretacion']}")
    print(f"{'='*62}\n")


# ─────────────────────────────────────────────
# BATCH
# ─────────────────────────────────────────────

def procesar_batch(videos_dir: str, csv_dir: str, output_dir: str):
    """
    Procesa todos los videos en videos_dir usando los CSV de csv_dir.
    Asume nombres coincidentes: bm1.mp4 → bm1_angulos.csv
    """
    videos_dir = Path(videos_dir)
    csv_dir    = Path(csv_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted([f for f in videos_dir.iterdir()
                     if f.suffix.lower() in {".mp4", ".avi", ".mov"}])

    print(f"\nBiometricSyncAnalyzer v1.0 — Batch")
    print(f"Videos: {len(videos)} | CSV dir: {csv_dir}")
    print(f"Output: {output_dir}\n")

    reportes_globales = []
    errores = []

    for i, video in enumerate(videos, 1):
        csv_candidato = csv_dir / f"{video.stem}_angulos.csv"
        print(f"[{i}/{len(videos)}] {video.name}")

        if not csv_candidato.exists():
            print(f"  SKIP: CSV no encontrado ({csv_candidato.name})")
            errores.append(video.name)
            continue

        resultado = analizar_video(str(video), str(csv_candidato), str(output_dir))

        if resultado is None:
            errores.append(video.name)
            continue

        resultados_faciales, df_bio, df_estables, nombre, outdir, fps = resultado
        reporte = calcular_estadisticas(resultados_faciales, df_bio, nombre, outdir)
        imprimir_resumen(reporte)
        reportes_globales.append(reporte)

    # Reporte consolidado
    if reportes_globales:
        consolidado_path = output_dir / "consolidado_sync.json"
        with open(consolidado_path, "w", encoding="utf-8") as f:
            json.dump(reportes_globales, f, indent=2, ensure_ascii=False)
        print(f"\nConsolidado guardado: {consolidado_path}")

    print(f"\nBatch completo: {len(reportes_globales)} OK | {len(errores)} sin procesar")
    if errores:
        print(f"Sin procesar: {', '.join(errores)}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BiometricSyncAnalyzer v1.0 — Pipeline biométrico multimodal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Un video (necesita CSV de BioAngles v1.2)
  python3 biometric_sync_analyzer.py video.mp4 --csv outputs/video_angulos.csv -o sync_output/

  # Batch completo (todos los videos + sus CSV)
  python3 biometric_sync_analyzer.py --batch VIDS_ANALISYS/Biometry_vids/ \\
      --csv-dir RESULTS_BIOMETRICOS/03_Dataset_Daniela_Biometry/ \\
      -o RESULTS_BIOMETRICOS/05_Sync_Multimodal/

NOTA: Requiere BioAngles v1.2 para que el CSV tenga columna 'fase_marcha'.
        """
    )
    parser.add_argument("video",   nargs="?", help="Ruta al video")
    parser.add_argument("--csv",             help="CSV de BioAngles del video")
    parser.add_argument("-o", "--output",    help="Directorio de salida",
                        default="sync_output/")
    parser.add_argument("--batch",           help="Directorio con videos para batch")
    parser.add_argument("--csv-dir",         help="Directorio con CSVs de BioAngles (batch)")
    parser.add_argument("-v", "--verbose",   action="store_true", default=True)

    args = parser.parse_args()

    if args.batch:
        if not args.csv_dir:
            parser.error("--batch requiere --csv-dir")
        procesar_batch(args.batch, args.csv_dir, args.output)

    elif args.video:
        if not args.csv:
            parser.error("Video individual requiere --csv")
        resultado = analizar_video(args.video, args.csv, args.output,
                                   verbose=args.verbose)
        if resultado:
            resultados_faciales, df_bio, df_estables, nombre, outdir, fps = resultado
            reporte = calcular_estadisticas(resultados_faciales, df_bio,
                                            nombre, Path(args.output))
            imprimir_resumen(reporte)
    else:
        parser.print_help()
