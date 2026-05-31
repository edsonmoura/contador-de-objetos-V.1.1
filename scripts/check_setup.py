from importlib.util import find_spec
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


PACKAGES = ["kivy", "cv2", "numpy", "onnx", "ultralytics"]


def main() -> None:
    print("Verificando ambiente...\n")

    for package in PACKAGES:
        status = "OK" if find_spec(package) else "FALTANDO"
        print(f"{package}: {status}")

    print()
    check_path("models/yolov8n.onnx", "modelo ONNX usado pelo app")
    check_path("yolov8n.pt", "modelo PyTorch baixado pelo Ultralytics")


def check_path(path_text: str, description: str) -> None:
    path = Path(path_text)
    status = "OK" if path.exists() else "FALTANDO"
    print(f"{path_text}: {status} ({description})")


if __name__ == "__main__":
    main()
