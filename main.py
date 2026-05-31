import os
from pathlib import Path
import shutil
import traceback

import cv2
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.utils import platform
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner

from object_counter.blob_counter import (
    detect_contrast_objects,
    detect_dark_objects,
    detect_light_objects,
)
from object_counter.counter import count_by_label, format_counts
from object_counter.detector import YoloOnnxDetector, draw_detections


LOG_PATH = Path("app_error.log")


MODES = {
    "beans": ("Feijao", "feijao", detect_dark_objects, 250),
    "rice": ("Arroz", "arroz", detect_light_objects, 40),
    "parts": ("Pecas", "peca", detect_contrast_objects, None),
    "dark": ("Escuros", "objeto escuro", detect_dark_objects, 800),
    "yolo": ("YOLO", None, None, None),
}

PARTS_PROFILE_LABELS = {
    "Ajuste: Automatico": "auto",
    "Ajuste: Mais preciso": "precise",
    "Ajuste: Mais sensivel": "sensitive",
    "Ajuste: Pecas pequenas": "small",
}

PARTS_TYPE_LABELS = {
    "Tipo: Todos": "all",
    "Tipo: Parafusos": "parafuso",
    "Tipo: Porcas": "porca",
    "Tipo: Arruelas": "arruela",
    "Tipo: Fixacoes": "fixacao",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def write_error_log(error: BaseException) -> None:
    LOG_PATH.write_text("".join(traceback.format_exception(error)), encoding="utf-8")


def request_android_camera_permission() -> None:
    try:
        from android.permissions import Permission, request_permissions

        request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE])
    except Exception:
        pass


def enable_android_camera_file_uri_compat() -> None:
    try:
        from jnius import autoclass

        StrictMode = autoclass("android.os.StrictMode")
        StrictMode.disableDeathOnFileUriExposure()
    except Exception:
        pass


def get_capture_output_path() -> Path:
    if platform == "android":
        try:
            from jnius import autoclass

            Environment = autoclass("android.os.Environment")
            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            activity = PythonActivity.mActivity
            pictures_dir = activity.getExternalFilesDir(Environment.DIRECTORY_PICTURES)
            if pictures_dir is not None:
                output_dir = Path(str(pictures_dir.getAbsolutePath()))
                output_dir.mkdir(parents=True, exist_ok=True)
                return output_dir / "capture.jpg"
        except Exception:
            pass

    output_dir = Path(App.get_running_app().user_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "capture.jpg"


def copy_android_content_uri_to_cache(uri: str) -> Path:
    from jnius import autoclass

    Uri = autoclass("android.net.Uri")
    PythonActivity = autoclass("org.kivy.android.PythonActivity")

    activity = PythonActivity.mActivity
    resolver = activity.getContentResolver()
    parsed_uri = Uri.parse(uri)
    mime_type = str(resolver.getType(parsed_uri) or "")
    extension = extension_from_mime_type(mime_type)

    cache_dir = Path(str(activity.getCacheDir().getAbsolutePath()))
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"selected_media{extension}"

    parcel = resolver.openFileDescriptor(parsed_uri, "r")
    if parcel is None:
        raise FileNotFoundError(f"Nao foi possivel abrir: {uri}")

    fd = parcel.detachFd()
    with os.fdopen(fd, "rb") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)

    return destination


def extension_from_mime_type(mime_type: str) -> str:
    if "png" in mime_type:
        return ".png"
    if "webp" in mime_type:
        return ".webp"
    if "video" in mime_type:
        return ".mp4"
    return ".jpg"


class ObjectCounterLayout(BoxLayout):
    def __init__(self, **kwargs) -> None:
        super().__init__(orientation="vertical", spacing=8, padding=8, **kwargs)

        request_android_camera_permission()

        self.mode = "beans"
        self.parts_profile = "auto"
        self.parts_type = "all"
        self.detector = self._create_detector()
        self.last_photo_path = None

        self.preview = Image(allow_stretch=True, keep_ratio=True)
        self.status = Label(
            text="Escolha o modo e toque em Tirar foto.",
            size_hint_y=None,
            height=150,
            halign="left",
            valign="top",
        )
        self.status.bind(size=self._sync_label_text_size)

        self.mode_buttons = GridLayout(cols=5, size_hint_y=None, height=52, spacing=4)
        for mode_key, mode_info in MODES.items():
            button = Button(text=mode_info[0])
            button.bind(on_press=lambda _button, key=mode_key: self.set_mode(key))
            self.mode_buttons.add_widget(button)

        self.capture_button = Button(
            text="Tirar foto e contar",
            size_hint_y=None,
            height=56,
            on_press=self.take_photo,
        )
        self.parts_profile_spinner = Spinner(
            text="Ajuste: Automatico",
            values=list(PARTS_PROFILE_LABELS.keys()),
            size_hint_y=None,
            height=50,
        )
        self.parts_profile_spinner.bind(text=self.set_parts_profile)
        self.parts_type_spinner = Spinner(
            text="Tipo: Todos",
            values=list(PARTS_TYPE_LABELS.keys()),
            size_hint_y=None,
            height=50,
        )
        self.parts_type_spinner.bind(text=self.set_parts_type)
        self.file_button = Button(
            text="Escolher arquivo e contar",
            size_hint_y=None,
            height=56,
            on_press=self.choose_media_file,
        )

        self.add_widget(self.preview)
        self.add_widget(self.mode_buttons)
        self.add_widget(self.parts_profile_spinner)
        self.add_widget(self.parts_type_spinner)
        self.add_widget(self.capture_button)
        self.add_widget(self.file_button)
        self.add_widget(self.status)

    def _create_detector(self) -> YoloOnnxDetector | None:
        try:
            return YoloOnnxDetector()
        except (FileNotFoundError, cv2.error):
            return None

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.status.text = f"Modo selecionado: {MODES[mode][0]}"

    def set_parts_profile(self, _spinner: Spinner, text: str) -> None:
        self.parts_profile = PARTS_PROFILE_LABELS.get(text, "auto")
        self.status.text = text

    def set_parts_type(self, _spinner: Spinner, text: str) -> None:
        self.parts_type = PARTS_TYPE_LABELS.get(text, "all")
        self.status.text = text

    def take_photo(self, _button: Button) -> None:
        output_path = get_capture_output_path()
        self.last_photo_path = output_path

        try:
            enable_android_camera_file_uri_compat()

            from plyer import camera

            camera.take_picture(str(output_path), self.on_photo_taken)
            self.status.text = "Camera aberta. Tire a foto e confirme."
        except Exception as error:
            write_error_log(error)
            self.status.text = (
                "Nao foi possivel abrir a camera nativa.\n"
                f"{type(error).__name__}: {error}\n"
                f"Detalhes em: {LOG_PATH}"
            )

    def choose_media_file(self, _button: Button) -> None:
        if platform == "android":
            try:
                from plyer import filechooser

                filechooser.open_file(
                    on_selection=self.on_media_file_selected,
                    multiple=False,
                    filters=["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"],
                )
                self.status.text = "Escolha uma foto ou video no seletor do celular."
                return
            except Exception as error:
                write_error_log(error)
                self.status.text = f"Nao foi possivel abrir o seletor nativo: {type(error).__name__}: {error}"
                return

        self.open_desktop_file_chooser()

    def open_desktop_file_chooser(self) -> None:
        chooser = FileChooserListView(
            path=str(Path.cwd()),
            filters=["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"],
        )
        actions = BoxLayout(size_hint_y=None, height=56, spacing=8)
        cancel_button = Button(text="Cancelar")
        count_button = Button(text="Contar")
        actions.add_widget(cancel_button)
        actions.add_widget(count_button)

        content = BoxLayout(orientation="vertical", spacing=8, padding=8)
        content.add_widget(chooser)
        content.add_widget(actions)

        popup = Popup(title="Escolha uma foto ou video", content=content, size_hint=(0.95, 0.95))
        cancel_button.bind(on_press=popup.dismiss)
        count_button.bind(on_press=lambda _instance: self.count_selected_media_file(chooser.selection, popup))
        popup.open()

    def on_media_file_selected(self, selection: list[str]) -> None:
        if not selection:
            self.status.text = "Nenhuma foto ou video selecionado."
            return

        if not self.analyze_media_reference(selection[0]):
            self.status.text = "Arquivo selecionado nao encontrado. Tente escolher outra foto."

    def count_selected_media_file(self, selection: list[str], popup: Popup) -> None:
        if not selection:
            self.status.text = "Selecione uma foto ou video para contar."
            return

        popup.dismiss()
        if not self.analyze_media_reference(selection[0]):
            self.status.text = "Arquivo selecionado nao encontrado. Tente escolher outra foto."

    def on_photo_taken(self, photo_path: str | None) -> None:
        candidates = [photo_path, str(self.last_photo_path) if self.last_photo_path else None]
        for candidate in candidates:
            if candidate and self.analyze_media_reference(candidate):
                return

        self.status.text = (
            "Foto nao encontrada. Tente novamente.\n"
            f"Destino esperado: {self.last_photo_path}"
        )

    def analyze_media_reference(self, reference: str) -> bool:
        try:
            if platform == "android" and reference.startswith("content://"):
                media_path = copy_android_content_uri_to_cache(reference)
                self.analyze_media_file(media_path)
                return True

            path = Path(reference)
            if not path.exists():
                return False

            self.analyze_media_file(path)
            return True
        except Exception as error:
            write_error_log(error)
            self.status.text = f"Erro ao abrir arquivo: {type(error).__name__}: {error}"
            return True

    def analyze_media_file(self, media_path: Path) -> None:
        extension = media_path.suffix.lower()
        if extension in IMAGE_EXTENSIONS:
            self.analyze_image(media_path)
            return

        if extension in VIDEO_EXTENSIONS:
            frame = self._read_video_frame(media_path)
            if frame is None:
                self.status.text = f"Nao foi possivel abrir o video: {media_path}"
                return

            self.analyze_frame(frame)
            return

        self.status.text = f"Formato nao suportado: {media_path.suffix}"

    def analyze_image(self, image_path: Path) -> None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            self.status.text = f"Nao foi possivel abrir a imagem: {image_path}"
            return

        self.analyze_frame(frame)

    def analyze_frame(self, frame) -> None:
        try:
            detections = self.detect(frame)
            counts, total = count_by_label(detections)
            annotated = draw_detections(frame, detections)
            self.preview.texture = self._frame_to_texture(annotated)
            self.status.text = format_counts(counts, total)
        except Exception as error:
            write_error_log(error)
            self.status.text = f"Erro ao contar: {type(error).__name__}: {error}"

    def _read_video_frame(self, video_path: Path):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return None

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

        success, frame = capture.read()
        capture.release()
        if not success:
            return None

        return frame

    def detect(self, frame):
        if self.mode == "yolo":
            if self.detector is None:
                raise RuntimeError("Modelo YOLO ONNX nao encontrado no APK.")
            return self.detector.detect(frame)

        _title, label, detector_function, min_area = MODES[self.mode]
        if self.mode == "parts":
            detections = detector_function(
                frame,
                min_area=min_area,
                label=label,
                profile=self.parts_profile,
            )
            return self.filter_parts_detections(detections)

        return detector_function(frame, min_area=min_area, label=label)

    def filter_parts_detections(self, detections):
        if self.parts_type == "all":
            return detections

        return [detection for detection in detections if detection.label == self.parts_type]

    def _frame_to_texture(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        flipped = cv2.flip(frame_rgb, 0)
        texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt="rgb")
        texture.blit_buffer(flipped.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        return texture

    def _sync_label_text_size(self, instance: Label, size: tuple[int, int]) -> None:
        instance.text_size = size


class ObjectCounterApp(App):
    title = "Contador de Objetos"

    def build(self):
        try:
            return ObjectCounterLayout()
        except Exception as error:
            write_error_log(error)
            return ErrorLayout(error)


class ErrorLayout(BoxLayout):
    def __init__(self, error: BaseException, **kwargs) -> None:
        super().__init__(orientation="vertical", padding=12, spacing=8, **kwargs)
        label = Label(
            text=(
                "O app encontrou um erro ao iniciar.\n\n"
                f"{type(error).__name__}: {error}\n\n"
                f"Detalhes salvos em: {LOG_PATH}"
            ),
            halign="left",
            valign="top",
        )
        label.bind(size=self._sync_label_text_size)
        self.add_widget(label)

    def _sync_label_text_size(self, instance: Label, size: tuple[int, int]) -> None:
        instance.text_size = size


if __name__ == "__main__":
    try:
        ObjectCounterApp().run()
    except Exception as error:
        write_error_log(error)
        raise
