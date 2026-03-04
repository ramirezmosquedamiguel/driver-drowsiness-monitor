"""Sistema de monitoreo de párpados para detección de somnolencia en conducción.

Este script usa:
- OpenCV para captura de video.
- MediaPipe Face Mesh para puntos faciales.
- EAR (Eye Aspect Ratio) para estimar cierre ocular.

Acciones implementadas al detectar somnolencia:
1) Alarma sonora.
2) Activación de una acción de seguridad vehicular (simulada como "luces de emergencia").

Requisitos:
    pip install opencv-python mediapipe numpy
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import platform
import subprocess
import time
from typing import Deque, Iterable, List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


# Índices de landmarks (MediaPipe FaceMesh) para cada ojo.
# Referencia de mapeo común de FaceMesh para EAR.
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


@dataclasses.dataclass
class DrowsinessConfig:
    ear_threshold: float = 0.21
    min_closed_seconds: float = 1.5
    fps_window: int = 30
    cooldown_seconds: float = 8.0


class SafetyAction:
    """Interfaz para acciones de seguridad."""

    def trigger(self) -> None:
        raise NotImplementedError


class SoundAlarmAction(SafetyAction):
    """Acción 1: Alarma sonora multiplataforma."""

    def trigger(self) -> None:
        system = platform.system().lower()
        try:
            if "windows" in system:
                import winsound

                winsound.Beep(2500, 700)
            elif "darwin" in system:
                subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], check=False)
            else:
                # Linux/Unix: intenta beep, si no está disponible usa bell ASCII.
                subprocess.run(["beep", "-f", "2500", "-l", "700"], check=False)
                print("\a", end="", flush=True)
        except Exception:
            # Fallback silencioso para no romper el bucle de monitoreo.
            print("[ALERTA] No se pudo reproducir audio en este entorno.")


class HazardLightsAction(SafetyAction):
    """Acción 2: Simulación de activación de luces intermitentes de emergencia.

    En un vehículo real, este método debería conectarse al bus CAN o ECU
    (mediante una API autorizada) para encender intermitentes y/o asistir
    reducción de velocidad de forma segura.
    """

    def trigger(self) -> None:
        print("[ACCION VEHICULO] Activando luces de emergencia y recomendando parada segura.")


def euclidean_dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def eye_aspect_ratio(eye_points: Sequence[np.ndarray]) -> float:
    """Calcula EAR con 6 puntos del ojo.

    Fórmula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """

    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)

    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def extract_eye_points(
    landmarks: Sequence, frame_w: int, frame_h: int, indexes: Iterable[int]
) -> List[np.ndarray]:
    pts: List[np.ndarray] = []
    for idx in indexes:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * frame_w, lm.y * frame_h], dtype=np.float32))
    return pts


def draw_eye_contours(frame: np.ndarray, points: Sequence[np.ndarray], color: Tuple[int, int, int]) -> None:
    poly = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=1)


def run_monitor(config: DrowsinessConfig, camera_index: int) -> None:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Verifica permisos/dispositivo.")

    actions: List[SafetyAction] = [SoundAlarmAction(), HazardLightsAction()]

    ear_values: Deque[float] = collections.deque(maxlen=config.fps_window)
    closed_start: float | None = None
    last_trigger_time = 0.0

    print("Monitoreo iniciado. Presiona 'q' para salir.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] No se pudo leer un frame de cámara.")
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb)
            status_text = "No face"
            status_color = (0, 255, 255)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_eye_pts = extract_eye_points(landmarks, frame_w, frame_h, LEFT_EYE_IDX)
                right_eye_pts = extract_eye_points(landmarks, frame_w, frame_h, RIGHT_EYE_IDX)

                left_ear = eye_aspect_ratio(left_eye_pts)
                right_ear = eye_aspect_ratio(right_eye_pts)
                ear = (left_ear + right_ear) / 2.0
                ear_values.append(ear)
                smooth_ear = float(np.mean(ear_values))

                draw_eye_contours(frame, left_eye_pts, (255, 200, 0))
                draw_eye_contours(frame, right_eye_pts, (255, 200, 0))

                now = time.time()
                if smooth_ear < config.ear_threshold:
                    if closed_start is None:
                        closed_start = now
                    elapsed = now - closed_start
                    status_text = f"Ojos cerrados ({elapsed:.1f}s)"
                    status_color = (0, 0, 255)

                    should_trigger = (
                        elapsed >= config.min_closed_seconds
                        and (now - last_trigger_time) >= config.cooldown_seconds
                    )
                    if should_trigger:
                        print("[ALERTA] Posible somnolencia detectada.")
                        for action in actions:
                            action.trigger()
                        last_trigger_time = now
                else:
                    closed_start = None
                    status_text = "Conductor alerta"
                    status_color = (0, 255, 0)

                cv2.putText(
                    frame,
                    f"EAR: {smooth_ear:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                status_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Driver Drowsiness Monitor", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitorea párpados del conductor y dispara acciones de seguridad."
    )
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara (default: 0)")
    parser.add_argument("--ear-threshold", type=float, default=0.21, help="Umbral EAR de ojo cerrado")
    parser.add_argument(
        "--min-closed-seconds",
        type=float,
        default=1.5,
        help="Segundos mínimos de ojo cerrado para detectar somnolencia",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=8.0,
        help="Tiempo mínimo entre alertas consecutivas",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DrowsinessConfig(
        ear_threshold=args.ear_threshold,
        min_closed_seconds=args.min_closed_seconds,
        cooldown_seconds=args.cooldown_seconds,
    )
    run_monitor(cfg, camera_index=args.camera)


if __name__ == "__main__":
    main()
