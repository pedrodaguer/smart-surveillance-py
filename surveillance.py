import datetime
import time
import cv2
import winsound
import os
import json
import structlog
from rich.console import Console
from rich.text import Text

console = Console()

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()

def log_console(status, message, extra=None):
    time_str = datetime.datetime.now().strftime("[%H:%M:%S]")
    emoji = {
        "start": "✓",
        "camera": "📷",
        "motion": "⚠️",
        "info": "ℹ️",
        "error": "❌"
    }.get(status, "")

    text = Text(f"{time_str} {emoji} {message}")

    if status == "start":
        text.stylize("green")
    elif status == "camera":
        text.stylize("cyan")
    elif status == "motion":
        text.stylize("yellow bold")
    elif status == "info":
        text.stylize("blue")
    elif status == "error":
        text.stylize("red bold")

    console.print(text)

    if extra:
        for key, value in extra.items():
            console.print(f"    ├─ {key}: {value}")


cam = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2()

last_beep = 0
delay_frames = 10
recording = False
out = None
output_dir = ""

logs = {"detecoes": []}

motion_timeout = 3  # segundos para agrupar movimentos curtos
last_motion_time = 0

if not cam.isOpened():
    log_console("error", "Erro ao abrir a camera")
else:
    log_console("start", "Sistema de vigilância iniciado")
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log_console("camera", f"Câmara conectada (resolução: {frame_width}x{frame_height})")

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        log_console("error", "Erro ao ler a camera")
        break

    fg_mask = backSub.apply(frame)
    delay_frames -= 1

    contours, hierarchy = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 2500
    large_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    frame_out = frame.copy()

    if large_contours:
        leftX, topY, rightX, bottomY = [], [], [], []

        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            leftX.append(x)
            topY.append(y)
            rightX.append(x + w)
            bottomY.append(y + h)

        x_min = min(leftX)
        y_min = min(topY)
        x_max = max(rightX)
        y_max = max(bottomY)

        cv2.rectangle(frame_out, (x_min, y_min),
                      (x_max, y_max), (0, 0, 200), 3)

    now = time.time()

    if large_contours:
        ultimo_tempo_movimento = agora

        if not recording:
            timestamp_start = datetime.datetime.now().isoformat(timespec="seconds")
            movimento_start = agora
            output_dir = os.path.join("Alertas", datetime.date.today().strftime("%Y-%m-%d"))
            os.makedirs(output_dir, exist_ok=True)

            formatted_now = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_file_name = f"movimento_{formatted_now}.mp4"
            output_file = os.path.join(output_dir, output_file_name)

            out = cv2.VideoWriter(
                output_file,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (frame.shape[1], frame.shape[0]),
            )
            recording = True

            log_console(
                "motion",
                "Movimento capturado!",
                {
                        "Iniciado em": timestamp_inicio,
                        "bounding_boxes": [
                        {
                            "x": int(x_min),
                            "y": int(y_min),
                            "largura": int(x_max - x_min),
                            "altura": int(y_max - y_min),
                        }
                    ],
                },
            )

        if agora - last_beep > 0.75:
            winsound.Beep(1000, 75)
            last_beep = agora

        out.write(frame)
        delay_frames = 10

    else:
        if recording:
            # espera até não haver movimento por tempo suficiente
            if agora - ultimo_tempo_movimento > teempo_sem_movimento_limit:
                out.release()
                out = None
                recording = False

                timestamp_fim = datetime.datetime.now().isoformat(timespec="seconds")
                movimento_duration = ultimo_tempo_movimento - movimento_start

                logs.setdefault("detecoes", []).append({
                    "timestamp_inicio": timestamp_inicio,
                    "timestamp_fim": timestamp_fim,
                    "duracao_segundos": round(movimento_duration, 2),
                    "ficheiro": output_file_name,
                    "bounding_boxes": [{
                        "x": int(x_min),
                        "y": int(y_min),
                        "largura": int(x_max - x_min),
                        "altura": int(y_max - y_min)
                    }],
                })

                log_console(
                    "motion",
                    "Fim do Movimento capturado!",
                    {
                        "Duração": f"{round(movimento_duration, 2)} s",
                        "Finalizado em": timestamp_fim,
                        "Ficheiro": output_file_name,
                        "bounding_boxes": [
                            {
                                "x": int(x_min),
                                "y": int(y_min),
                                "largura": int(x_max - x_min),
                                "altura": int(y_max - y_min),
                            }
                        ],
                    },
                )

    cv2.imshow('Live Feed', frame_out)

    if cv2.waitKey(1) == 27:
        break

with open("log_movimentos.json", "w", encoding="utf-8") as f:
    json.dump(logs, f, indent=2, ensure_ascii=False)

cam.release()
cv2.destroyAllWindows()
