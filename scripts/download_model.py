from pathlib import Path
from shutil import move
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


MODEL_PATH = Path("models/yolov8n.onnx")


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ensure_onnx_is_available()

    print("Baixando yolov8n.pt e exportando para ONNX...")
    model = YOLO("yolov8n.pt")
    exported_path = Path(model.export(format="onnx", imgsz=640, opset=12, simplify=False))

    if exported_path.resolve() != MODEL_PATH.resolve():
        move(str(exported_path), MODEL_PATH)

    print(f"Modelo salvo em {MODEL_PATH}")


def ensure_onnx_is_available() -> None:
    try:
        import onnx  # noqa: F401
    except ModuleNotFoundError as error:
        raise SystemExit(
            "O pacote ONNX nao esta instalado corretamente.\n"
            "No Windows, rode estes comandos e tente de novo:\n\n"
            "python -m pip uninstall -y onnx\n"
            "python -m pip install --no-cache-dir --force-reinstall onnx==1.16.2\n"
            "python -m pip install -r requirements-model.txt\n"
        ) from error


if __name__ == "__main__":
    main()
