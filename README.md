# driver-drowsiness-monitor

Sistema en **Python** para monitorear el movimiento de los párpados del conductor y reducir riesgo de accidentes por somnolencia.

## ¿Qué hace?

El sistema usa cámara + visión por computador para estimar el cierre de ojos con el indicador **EAR (Eye Aspect Ratio)**.

Cuando detecta ojos cerrados por encima de un umbral de tiempo, ejecuta **dos acciones** de seguridad:

1. **Alarma sonora** (acción inmediata para despertar al conductor).
2. **Acción vehicular de protección** (simulada): activación de luces de emergencia y recomendación de parada segura.

> En integración real con vehículo, la segunda acción debe conectarse al sistema autorizado del automóvil (CAN/ECU/API del fabricante).

## Requisitos

```bash
pip install opencv-python mediapipe numpy
```

## Ejecución

```bash
python main.py --camera 0
```

### Parámetros útiles

- `--ear-threshold`: umbral EAR para considerar ojo cerrado (default `0.21`)
- `--min-closed-seconds`: segundos mínimos de cierre ocular para detectar somnolencia (default `1.5`)
- `--cooldown-seconds`: tiempo entre alertas consecutivas (default `8.0`)

Ejemplo:

```bash
python main.py --camera 0 --ear-threshold 0.22 --min-closed-seconds 1.2
```

## Salida en pantalla

- Estado del conductor (`Conductor alerta`, `Ojos cerrados`, etc.).
- Valor EAR suavizado en tiempo real.
- Contorno de ambos ojos.

Presiona `q` para salir.
