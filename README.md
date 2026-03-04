# driver-drowsiness-monitor

Sistema en **Python** para monitorear el movimiento de los párpados del conductor y reducir riesgo de accidentes por somnolencia.

## ¿Qué hace?

El sistema usa cámara + visión por computador para estimar el cierre de ojos con el indicador **EAR (Eye Aspect Ratio)**.

Cuando detecta ojos cerrados por encima de un umbral de tiempo, ejecuta **dos acciones** de seguridad:

1. **Alarma sonora** escalonada (no bloqueante).
2. **Acción vehicular de protección** (simulada): activación de luces de emergencia y recomendación de parada segura.

> En integración real con vehículo, la segunda acción debe conectarse al sistema autorizado del automóvil (CAN/ECU/API del fabricante).

## Requisitos

```bash
pip install -r requirements.txt
```

(o manualmente: `opencv-python mediapipe numpy sounddevice`)

## Nuevo sistema de alarma escalonada (2 etapas)

La lógica de alarma fue rediseñada para no activarse de inmediato y escalar según la duración continua del cierre ocular:

1. **Retraso de activación (0s – 5s de ojos cerrados)**
   - No suena alarma todavía.
2. **Etapa 1 (5s – 15s)**
   - Sonido intermitente y moderado.
   - Onda seno generada con `numpy`.
   - Reproducción con `sounddevice`.
   - Patrón lento: **1s tono / 1s silencio**.
   - Frecuencia objetivo: alrededor de **800 Hz**.
3. **Etapa 2 (>15s)**
   - Escalado de intensidad (más amplitud).
   - Cadencia más rápida: **0.4s tono / 0.2s silencio**.
   - Frecuencias alternadas estilo emergencia: **700 Hz / 1200 Hz**.

Comportamiento adicional:
- La alarma continúa mientras los ojos sigan cerrados.
- Se detiene inmediatamente cuando los ojos se abren.
- El audio corre en segundo plano para no bloquear el bucle de cámara.

## Ejecución

```bash
python main.py --camera 0
```

### Parámetros útiles

- `--ear-threshold`: umbral EAR para considerar ojo cerrado (default `0.21`)
- `--min-closed-seconds`: segundos continuos de cierre ocular para activar la alarma (default `5.0`)

Ejemplo:

```bash
python main.py --camera 0 --ear-threshold 0.22 --min-closed-seconds 5
```

## Salida en pantalla

- Estado del conductor (`Conductor alerta`, `Ojos cerrados`, etc.).
- Valor EAR suavizado en tiempo real.
- Contorno de ambos ojos.

Presiona `q` para salir.
