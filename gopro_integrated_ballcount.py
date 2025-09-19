import sys
import os
import platform
# Fuerza software OpenGL **antes de importar PyQt5**
if platform.system() == "Windows":
    os.environ["QT_OPENGL"] = "software"

import numpy as np
import asyncio
from datetime import datetime
import socket
from threading import Thread
import threading
import requests
import csv
import queue
import time

import cv2
from open_gopro import WirelessGoPro
from open_gopro.models import constants, streaming

# Ahora sí, importa PyQt5
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEvent, QRectF, QPointF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QSlider, QGridLayout, QMessageBox,
    QDialog, QLineEdit, QButtonGroup, QSpacerItem, QSizePolicy, QFrame,
    QMenu, QInputDialog
)
from PyQt5.QtGui import QImage, QPixmap, QIntValidator, QPainter, QPen, QBrush

# --- Plano 2D con imagen + puntos ---
class PointsCanvas(QWidget):
    hovered_index_changed = pyqtSignal(int)          # para posibles hooks futuros
    request_assign_index = pyqtSignal(int, int)      # (idx del punto, nuevo índice asignado)
    request_erase_point = pyqtSignal(int)            # idx del punto en lista
    request_goto_point = pyqtSignal(float, float)

    def __init__(self, parent=None, mm_size=(1000.0, 1000.0), image_name="Test plate.png"):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.mm_w, self.mm_h = mm_size
        # lista de puntos: [{'x': float, 'y': float, 'idx': Optional[int]}]
        self.points = []
        # carga imagen
        self.img = QPixmap(os.path.join(os.path.dirname(__file__), image_name))
        if self.img.isNull():
            alt = "/mnt/data/Test plate.png"
            self.img = QPixmap(alt)
        self.img_rect = QRectF()
        self.hover_i = -1

    # API pública
    def set_points(self, points):
        """points: list of dicts {'x': float, 'y': float, 'idx': Optional[int]}"""
        self.points = points[:]
        self.update()

    def add_point(self, x, y, idx=None):
        self.points.append({'x': float(x), 'y': float(y), 'idx': idx})
        self.update()

    def erase_point_at(self, i):
        if 0 <= i < len(self.points):
            del self.points[i]
            self.update()

    def assign_index_at(self, i, new_index):
        if 0 <= i < len(self.points):
            self.points[i]['idx'] = int(new_index)
            self.update()

    # Geometría: encajar imagen centrada manteniendo aspect ratio
    def resizeEvent(self, e):
        self._recalc_img_rect()
        return super().resizeEvent(e)

    def _recalc_img_rect(self):
        if self.img.isNull():
            self.img_rect = QRectF(0, 0, self.width(), self.height())
            return
        W, H = self.width(), self.height()
        iw, ih = self.img.width(), self.img.height()
        scale = min(W/iw, H/ih) if iw and ih else 1.0
        rw, rh = iw*scale, ih*scale
        x = (W - rw)/2.0
        y = (H - rh)/2.0
        self.img_rect = QRectF(x, y, rw, rh)

    # Conversión mm -> píxel de pantalla (coordenadas de dibujo)
    def _mm_to_screen(self, xm, ym):
        if self.img.isNull():
            return QPointF(xm, ym)
        # píxel dentro de la imagen original
        px = (xm / self.mm_w) * self.img.width()
        py = (ym / self.mm_h) * self.img.height()
        # escalar a rectángulo dibujado
        sx = self.img_rect.left() + px * (self.img_rect.width() / self.img.width())
        sy = self.img_rect.top()  + py * (self.img_rect.height() / self.img.height())
        return QPointF(sx, sy)

    # Dibujo
    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.white)
        if not self.img.isNull():
            painter.drawPixmap(self.img_rect, self.img, QRectF(self.img.rect()))
        # Dibuja puntos
        radius = 5
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        for i, p in enumerate(self.points):
            sp = self._mm_to_screen(p['x'], p['y'])
            hovered = (i == self.hover_i)
            painter.setPen(QPen(Qt.black, 1.5))
            painter.setBrush(QBrush(Qt.red if hovered else Qt.black))
            painter.drawEllipse(sp, radius, radius)
            # Texto: " . (x,y)" o "n . (x,y)"
            tag = "."
            if p.get('idx') is not None:
                tag = f"{p['idx']} ."
            label = f"{tag} ({p['x']:.2f},{p['y']:.2f})"
            painter.setPen(Qt.black)
            painter.drawText(sp + QPointF(8, -8), label)
        painter.end()

    # Interacción
    def mouseMoveEvent(self, e):
        self.hover_i = self._hit_test(e.pos())
        self.hovered_index_changed.emit(self.hover_i)
        self.update()
        return super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            i = self._hit_test(e.pos())
            if i != -1:
                self._open_context_menu(i, e.globalPos())
        return super().mousePressEvent(e)

    def _hit_test(self, pos):
        """Devuelve el índice del punto bajo el mouse (radio 10 px) usando distancia euclidiana."""
        px, py = float(pos.x()), float(pos.y())
        for i, p in enumerate(self.points):
            sp = self._mm_to_screen(p['x'], p['y'])
            dx = float(sp.x()) - px
            dy = float(sp.y()) - py
            if (dx * dx + dy * dy) <= (10.0 * 10.0):
                return i
        return -1

    def _open_context_menu(self, i, global_pos):
        menu = QMenu(self)
        a_assign = menu.addAction("Assign index")
        a_erase  = menu.addAction("Erase coordinate")
        a_goto   = menu.addAction("Go to point")
        menu.addSeparator()
        a_exit   = menu.addAction("Exit")
        action = menu.exec_(global_pos)
        if action == a_assign:
            # solo enteros >= 0
            val, ok = QInputDialog.getInt(self, "Assign index", "Index (>=0):", 0, 0, 10_000, 1)
            if ok:
                # primero aplica al canvas...
                self.assign_index_at(i, val)
                # ...y luego notifica a MainWindow con (indice_punto, nuevo_valor)
                self.request_assign_index.emit(i, val)
        elif action == a_erase:
            self.request_erase_point.emit(i)
        elif action == a_goto:
            p = self.points[i]
            self.request_goto_point.emit(float(p['x']), float(p['y']))

        else:
            pass


# Configuration
STREAM_PORT = 8554
# Usa carpeta amigable según SO
if platform.system() == "Windows":
    DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Go_Pro_Videos")
else:
    DOWNLOAD_DIR = "/home/dtc-dresden/Go_Pro_Videos"
try:
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
except Exception:
    pass
POSITION_SERVER_PORT = 65432
RECONNECT_DELAY = 3

toggle_recording_int=0

if platform.system() == "Windows":
    # En Windows usa SelectorEventLoop (evita problemas con Proactor + librerías nativas)
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class PositionServerThread:
    def __init__(self, host='0.0.0.0', port=POSITION_SERVER_PORT):
        self.host = host
        self.port = port
        self.positions = {f"pos{i}": ["0.0", "0.0"] for i in range(1, 11)}
        self.queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        # Desbloquear accept() si está esperando
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                pass
        except Exception:
            pass
        self._thread.join(timeout=1.0)

    def _run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(5)
        s.settimeout(0.5)
        print(f"Position server listening on {self.host}:{self.port}")
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = s.accept()
                    t = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
                    t.start()
                except socket.timeout:
                    continue
        finally:
            s.close()

    def _handle_client(self, conn):
        with conn:
            buf = ""
            while not self._stop.is_set():
                data = conn.recv(1024)
                if not data:
                    break
                try:
                    buf += data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if "=" in line:
                        key, value = line.split("=", 1)
                        if key.startswith("pos") and key[3:].isdigit():
                            n = int(key[3:])
                            if 1 <= n <= 10 and "," in value:
                                x, y = value.split(",", 1)
                                self.positions[key] = [x, y]
                                # Encola una copia para el hilo Qt
                                self.queue.put(self.positions.copy())

class CNCPanel(QDialog):
    def __init__(self, duet_ip="192.168.185.2", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual CNC Control Panel")
        self.setFixedSize(400, 600)

        # Configuración
        self.duet_ip = duet_ip
        self.feeder_velocity = 600
        self.single_jump = 1.0  # Paso por defecto

        # Estados de movimiento
        self.y_positive = False
        self.y_negative = False
        self.x_positive = False
        self.x_negative = False

        # Timer para jog continuo
        self.jog_timer = QTimer()
        self.jog_timer.timeout.connect(self.send_continuous_move)
        self.jog_interval = 100  # ms

        # Estilos botones
        self.pressed_style = "background-color: #4CAF50; color: white;"
        self.released_style = ""

        self.init_ui()

        # Timer to poll Duet position
        self.position_timer = QTimer(self)
        self.position_timer.timeout.connect(self.poll_duet_position)
        self.position_timer.start(800)  # ms

    def init_ui(self):
        layout = QVBoxLayout()

        # --- Velocity control ---
        velocity_group = QGroupBox("Velocity Control (F)")
        velocity_layout = QVBoxLayout()

        self.velocity_input = QLineEdit(str(self.feeder_velocity))
        self.velocity_input.setValidator(QIntValidator(1, 6000))
        velocity_layout.addWidget(self.velocity_input)

        confirm_btn = QPushButton("Confirm Velocity")
        confirm_btn.clicked.connect(self.confirm_velocity)
        velocity_layout.addWidget(confirm_btn)

        self.velocity_display = QLabel(f"F: {self.feeder_velocity}")
        velocity_layout.addWidget(self.velocity_display)

        # Jump value selection
        self.jump_group = QButtonGroup(self)
        self.jump_group.setExclusive(True)
        jump_layout = QHBoxLayout()
        for value in [0.01, 0.1, 1, 10, 100]:
            btn = QPushButton(str(value))
            btn.setCheckable(True)
            if value == self.single_jump:
                btn.setChecked(True)
            btn.clicked.connect(lambda _, v=value: self.set_jump_value(v))
            self.jump_group.addButton(btn)
            jump_layout.addWidget(btn)
        velocity_layout.addLayout(jump_layout)
        velocity_group.setLayout(velocity_layout)
        layout.addWidget(velocity_group)

        # --- Duet live position ---
        pos_group = QGroupBox("Duet Position (M114)")
        pos_layout = QVBoxLayout()
        self.duet_pos_edit = QLineEdit("X: ---, Y: ---")
        self.duet_pos_edit.setReadOnly(True)
        pos_layout.addWidget(self.duet_pos_edit)
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)

        # --- D-pad section ---
        dpad_group = QGroupBox("Movement Control")
        dpad_layout = QGridLayout()

        self.y_plus_btn = QPushButton("Y+")
        self.y_minus_btn = QPushButton("Y-")
        self.x_minus_btn = QPushButton("X-")
        self.x_plus_btn = QPushButton("X+")

        for btn, direction in [
            (self.y_plus_btn, 'y_positive'),
            (self.y_minus_btn, 'y_negative'),
            (self.x_minus_btn, 'x_negative'),
            (self.x_plus_btn, 'x_positive')
        ]:
            btn.setCheckable(True)
            btn.setStyleSheet(self.released_style)
            btn.pressed.connect(lambda dir=direction, b=btn: self.start_jog(dir, b))
            btn.released.connect(lambda dir=direction, b=btn: self.stop_jog(dir, b))

        dpad_layout.addWidget(self.y_plus_btn, 0, 1)
        dpad_layout.addWidget(self.y_minus_btn, 2, 1)
        dpad_layout.addWidget(self.x_minus_btn, 1, 0)
        dpad_layout.addWidget(self.x_plus_btn, 1, 2)

        dpad_group.setLayout(dpad_layout)
        layout.addWidget(dpad_group)

        # --- Controls list ---
        controls_group = QGroupBox("Keyboard Controls")
        controls_layout = QVBoxLayout()
        for control in [
            "W: Y+",
            "A: X-",
            "S: Y-",
            "D: X+",
            "+ : Add point",
            "- : Remove last point (this session)"
        ]:
            controls_layout.addWidget(QLabel(f"- {control}"))
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        self.setLayout(layout)
        self.installEventFilter(self)

        # at the end of CNCPanel.init_ui()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    # ----------------------------
    # Configuración de valores
    # ----------------------------
    def confirm_velocity(self):
        text = self.velocity_input.text()
        if text:
            try:
                velocity = int(text)
                if 1 <= velocity <= 6000:
                    self.feeder_velocity = velocity
                    self.velocity_display.setText(f"F: {self.feeder_velocity}")
                else:
                    QMessageBox.warning(self, "Invalid Value", "Velocity must be between 1 and 6000")
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid number")

    def set_jump_value(self, value):
        self.single_jump = value
        print(f"Jump value set to: {self.single_jump}")

    # ----------------------------
    # Comunicación con Duet
    # ----------------------------
    def send_gcode_command(self, command):
        try:
            r = requests.get(
                f"http://{self.duet_ip}/rr_gcode",
                params={"gcode": command},
                timeout=2
            )
            if r.status_code == 200:
                data = r.json()
                if ("err" not in data) or (data.get("err") == 0):
                    print(f"GCode ejecutado correctamente: {command}")
                    return True
                else:
                    print(f"Error en GCode: {data}")
                    return False
            else:
                print(f"HTTP Error {r.status_code}")
                return False
        except Exception as e:
            print(f"Error de red: {e}")
            return False

    def send_gcode_line(self, command):
        if self.send_gcode_command(command):
            print(f"✅ Ejecutado: {command}")
        else:
            print(f"❌ Error ejecutando: {command}")

    def get_current_position(self):
        try:
            # Ask Duet to produce the position reply
            requests.get(
                f"http://{self.duet_ip}/rr_gcode",
                params = {"gcode": "M114"},
                timeout = 2
            )
            # Read the reply
            r = requests.get(f"http://{self.duet_ip}/rr_reply", timeout=2)
            raw = r.text
            print(f"Respuesta cruda: {raw}")
            import re
            match = re.search(r"X:([\d\.-]+)\s+Y:([\d\.-]+)", raw)
            if match:
                x = match.group(1)[:6]
                y = match.group(2)[:6]
                return x, y
            else:
                return None, None
        except Exception as e:
            print(f"Error obteniendo posición: {e}")
            return None, None

    def poll_duet_position(self):
        x, y = self.get_current_position()
        if x is not None and y is not None:
            self.duet_pos_edit.setText(f"X: {x}, Y: {y}")

    # ----------------------------
    # Jog continuo
    # ----------------------------
    def start_jog(self, direction, button):
        button.setStyleSheet(self.pressed_style)
        self.active_direction = direction
        # (optional) pause polling while jogging to reduce contention
        if self.position_timer.isActive():
            self.position_timer.stop()
        self.jog_timer.start(self.jog_interval)  # Inicia envío periódico

    def stop_jog(self, direction, button):
        button.setStyleSheet(self.released_style)
        self.jog_timer.stop()
        self.active_direction = None
        # (optional) resume polling after jogging
        if not self.position_timer.isActive():
            self.position_timer.start(800)

    def send_sequence(self, commands):
        ok = True
        for c in commands:
            if not self.send_gcode_command(c):
                ok = False
        return ok

    def send_continuous_move(self):
        if self.active_direction == 'y_positive':
            move = f"G1 Y{self.single_jump} F{self.feeder_velocity}"
        elif self.active_direction == 'y_negative':
            move = f"G1 Y-{self.single_jump} F{self.feeder_velocity}"
        elif self.active_direction == 'x_positive':
            move = f"G1 X{self.single_jump} F{self.feeder_velocity}"
        elif self.active_direction == 'x_negative':
            move = f"G1 X-{self.single_jump} F{self.feeder_velocity}"
        else:
            return
        # Relative move; you can comment out G90 if you want to stay in relative mode while jogging
        self.send_sequence(["G91", move, "G90"])

    # ----------------------------
    # Control por teclado
    # ----------------------------
    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress:
            key_map = {
                Qt.Key_W: ('y_positive', self.y_plus_btn),
                Qt.Key_S: ('y_negative', self.y_minus_btn),
                Qt.Key_A: ('x_negative', self.x_minus_btn),
                Qt.Key_D: ('x_positive', self.x_plus_btn)
            }
            # '+' agrega punto; '-' borra el último punto agregado en esta sesión
            if event.key() in (Qt.Key_Plus,):
                print("Tecla + presionada")
                # Tecla '+' (algunas distros usan '=' con Shift)
                x, y = self.get_current_position()
                if x is not None and y is not None and isinstance(self.parent(), QMainWindow):
                    self.parent().add_point_from_duet(float(x), float(y))
                return True
            if event.key() in (Qt.Key_Minus,):
                print("Tecla - presionada")
                if isinstance(self.parent(), QMainWindow):
                    self.parent().remove_last_session_point()
                return True

            if event.key() in key_map and not self.jog_timer.isActive():
                direction, btn = key_map[event.key()]
                btn.setChecked(True)
                self.start_jog(direction, btn)
                return True

        elif event.type() == QEvent.KeyRelease:
            key_map = {
                Qt.Key_W: ('y_positive', self.y_plus_btn),
                Qt.Key_S: ('y_negative', self.y_minus_btn),
                Qt.Key_A: ('x_negative', self.x_minus_btn),
                Qt.Key_D: ('x_positive', self.x_plus_btn)
            }
            if event.key() in key_map:
                direction, btn = key_map[event.key()]
                btn.setChecked(False)
                self.stop_jog(direction, btn)
                return True

        return super().eventFilter(source, event)

class MediaHttpGetter:
    pass

###############################
# GoPro manager
###############################
class GoProManager(QThread):
    frame_ready = pyqtSignal(QImage)
    status_update = pyqtSignal(str)
    position_update = pyqtSignal(tuple)
    recording_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.constants = None
        self.gopro = None
        self.cap = None
        self.recording = False
        self.reference_pos = None
        self.error_range = None
        self.hsv_values = [0, 10, 170, 180, 120, 255, 70, 255]
        self._run_flag = True
        self.show_filters = False
        self.loop = None
        self.stream_url = None

    async def initialize(self):
        try:
            self.gopro = WirelessGoPro(ble_timeout=15, ble_retries=3)
            await self.gopro.open()
            self.status_update.emit("✅ Connected to GoPro")
            return True
        except Exception as e:
            self.status_update.emit(f"⚠️ Connection error: {str(e)}")
            return False

    async def start_stream(self):
        try:
            await self.gopro.streaming.start_stream(
                streaming.StreamType.PREVIEW,
                streaming.PreviewStreamOptions(port=STREAM_PORT))
            self.stream_url = self.gopro.streaming.url
            self.cap = cv2.VideoCapture(self.stream_url)
            self.status_update.emit("📡 Stream started")
        except Exception as e:
            self.status_update.emit(f"⚠️ Stream error: {str(e)}")

    async def toggle_recording(self):
        print("Toggle recording async iniciado")
        try:
            self.status_update.emit("🎬 Attempting to toggle recording...")
            await asyncio.sleep(1)
            state_resp = await self.gopro.http_command.get_camera_state()
            encoding = state_resp.data.get(constants.StatusId.ENCODING, 0)
            recording_now = bool(encoding)
            print("Current real state (ENCODING):", encoding)
            toggle = constants.Toggle.DISABLE if recording_now else constants.Toggle.ENABLE
            print(f"Toggle value: {toggle}")
            try:
                resp = await asyncio.wait_for(
                    self.gopro.http_command.set_shutter(shutter=toggle),
                    timeout=5
                )
                print("HTTP set_shutter response:", resp)
            except asyncio.TimeoutError:
                print("⏰ Timeout at sending shutter (probably recording)")
                resp = None
            await asyncio.sleep(1)
            confirm_resp = await self.gopro.http_command.get_camera_state()
            confirmed_encoding = confirm_resp.data.get(constants.StatusId.ENCODING, 0)
            recording_confirmed = bool(confirmed_encoding)
            print("Confirmed state (ENCODING):", confirmed_encoding)
            if recording_confirmed:
                self.status_update.emit("🔴 Recording started")
            else:
                self.status_update.emit("⏹️ Recording stopped")
            print(">>> Recording state toggled successfully")
        except Exception as e:
            error_msg = f"Recording: error → {e}"
            print("❌", error_msg)
            self.status_update.emit(error_msg)

    async def download_and_log(self):
        print("Download routine started")
        try:
            media_resp = await self.gopro.http_command.get_media_list()
            files = media_resp.data.files
            print("DEBUG media count:", len(files))
            if not files:
                raise RuntimeError("No media files found")
            last = max(files, key=lambda x: x.filename)
            print(f"DEBUG most recent file: {last.filename}")
            fecha_str = datetime.now().strftime("%Y_%m_%d")
            ext = os.path.splitext(last.filename)[1]
            new_filename = f"{fecha_str}{ext}"
            os.makedirs(DOWNLOAD_DIR, exist_ok=True)
            dest_path = os.path.join(DOWNLOAD_DIR, new_filename)
            download_resp = await self.gopro.http_command.download_file(camera_file=last.filename)
            temp_path = download_resp.data
            os.rename(temp_path, dest_path)
            self.status_update.emit(f"✅ Video downloaded: {new_filename}")
            print(f">>> File saved in: {dest_path}")
        except Exception as e:
            error_msg = f"⚠️ Download error: {e}"
            print("❌", error_msg)
            self.status_update.emit(error_msg)

    def set_reference_position(self, pos):
        self.reference_pos = pos
        if pos:
            self.error_range = (int(pos[0] * 0.01), int(pos[1] * 0.01))
            self.status_update.emit(f"📍 Reference position set: {pos}")

    def update_hsv_values(self, values):
        self.hsv_values = values

    def toggle_filters(self):
        self.show_filters = not self.show_filters
        if not self.show_filters:
            cv2.destroyWindow("Red Mask")
            cv2.destroyWindow("Cleaned Mask")

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run())

    async def _run(self):
        if not await self.initialize():
            return
        while self._run_flag:
            try:
                await self.start_stream()
                while self._run_flag and self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        processed_frame, masks = self.process_frame(frame)
                        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                        self.frame_ready.emit(qt_image)
                        if self.show_filters:
                            red_mask, cleaned_mask = masks
                            cv2.imshow("Red Mask", red_mask)
                            cv2.imshow("Cleaned Mask", cleaned_mask)
                            cv2.waitKey(1)
                    else:
                        break
                if self.show_filters:
                    cv2.destroyWindow("Red Mask")
                    cv2.destroyWindow("Cleaned Mask")
                if self.cap:
                    self.cap.release()
                if self.gopro:
                    await self.gopro.streaming.stop_active_stream()
                if not self._run_flag:
                    break
                self.status_update.emit(f"🔄 Reconnecting in {RECONNECT_DELAY} seconds...")
                await asyncio.sleep(RECONNECT_DELAY)
            except Exception as e:
                self.status_update.emit(f"⚠️ Stream error: {str(e)}")
                await asyncio.sleep(RECONNECT_DELAY)
        if self.gopro:
            await self.gopro.close()

    def process_frame(self, frame):
        x_start, x_end = 480, 1440
        y_start, y_end = 270, 810
        roi = frame[y_start:y_end, x_start:x_end]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([self.hsv_values[0], self.hsv_values[4], self.hsv_values[6]])
        upper_red1 = np.array([self.hsv_values[1], self.hsv_values[5], self.hsv_values[7]])
        lower_red2 = np.array([self.hsv_values[2], self.hsv_values[4], self.hsv_values[6]])
        upper_red2 = np.array([self.hsv_values[3], self.hsv_values[5], self.hsv_values[7]])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        processed_frame = frame.copy()
        current_pos = None
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                x += x_start;
                y += y_start
                current_pos = (x + w // 2, y + h // 2)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(processed_frame, current_pos, 5, (0, 255, 0), -1)
                cv2.putText(processed_frame, 'X', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"({current_pos[0]}, {current_pos[1]})",
                            (current_pos[0] + 10, current_pos[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                self.position_update.emit(current_pos)
        if self.reference_pos is not None:
            ref_x, ref_y = self.reference_pos
            err_x, err_y = self.error_range
            cv2.circle(processed_frame, self.reference_pos, 5, (255, 0, 0), -1)
            cv2.rectangle(processed_frame,
                          (ref_x - err_x, ref_y - err_y),
                          (ref_x + err_x, ref_y + err_y),
                          (255, 255, 0), 2)
        return processed_frame, (red_mask, cleaned_mask)

    def stop(self):
        self._run_flag = False
        if self.loop and self.loop.is_running():
            self.loop.stop()
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Monitoring System")
        self.setGeometry(100, 100, 1400, 800)
        print("[INIT] MainWindow starting...")
        self.duet_ip = "192.168.185.2"  # ajusta si tu IP es otra

        # --- Setup GoPro (siempre habilitado) ---
        self.enable_gopro = True

        self.gopro_manager = GoProManager()  # 1) crear
        self.gopro_manager.frame_ready.connect(self.update_gopro_image)  # 2) conectar señales
        self.gopro_manager.status_update.connect(self.update_status)
        self.gopro_manager.position_update.connect(self.handle_position_update)
        self.gopro_manager.recording_changed.connect(self.update_recording_label)

        self.gopro_manager.start()  # 3) arrancar hilo (crea su propio event loop)

        # Current position tracking
        self.current_position = None
        # Modelo de puntos + pila por sesión (para '-')
        self.points = []  # [{'x':float,'y':float,'idx':Optional[int]}]
        self.session_stack = []  # guarda índices añadidos durante esta sesión del panel manual

        # Initialize UI
        print("[INIT] Building UI...")
        self.init_ui()
        print("[INIT] UI built.")

        # CSV de puntos (persistente)
        self.points_csv_path = os.path.join(os.path.dirname(__file__), "points.csv")
        self.load_points_from_csv()
        print("[INIT] Points CSV loaded.")

        # Ahora sí, arranca el servidor
        self.position_server = PositionServerThread()
        self.position_server.start()

        self._pos_timer = QTimer(self)
        self._pos_timer.timeout.connect(self._drain_pos_queue)
        self._pos_timer.start(100)

        self.duet_gpIn_index = 1  # J1 -> sensors.gpIn[1]
        self.ball_timer = QTimer(self)
        self.ball_timer.timeout.connect(self.poll_duet_ballcount)
        self.ball_timer.start(1000)
        print("[INIT] MainWindow ready.")

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel (GoPro and controls)
        left_panel = QVBoxLayout()

        # GoPro video group
        gopro_group = QGroupBox("GoPro View - Red X Detection")
        gopro_layout = QVBoxLayout()

        self.gopro_label = QLabel()
        self.gopro_label.setAlignment(Qt.AlignCenter)
        self.gopro_label.setMinimumSize(640, 480)

        gopro_layout.addWidget(self.gopro_label)
        gopro_group.setLayout(gopro_layout)
        left_panel.addWidget(gopro_group)

        # GoPro controls group
        control_group = QGroupBox("GoPro Controls")
        control_layout = QGridLayout()

        # --- NUEVOS BOTONES ---
        self.gcode_console_btn = QPushButton("Write command")
        self.gcode_console_btn.clicked.connect(self.show_gcode_console)

        self.draw_circle_btn = QPushButton("Draw circle")
        self.draw_circle_btn.clicked.connect(self.show_draw_circle_dialog)

        # Colocar en la grilla (fila 2)
        control_layout.addWidget(self.gcode_console_btn, 2, 0)
        control_layout.addWidget(self.draw_circle_btn, 2, 1)

        self.start_test_btn = QPushButton("Start test")
        self.start_test_btn.clicked.connect(self.start_test)

        # Colócalo en la grilla; por ejemplo, fila 3
        control_layout.addWidget(self.start_test_btn, 3, 0)

        self.record_btn = QPushButton("Toggle Recording")
        self.record_btn.clicked.connect(self.handle_record_click)

        self.set_ref_btn = QPushButton("Set Reference Position")
        self.set_ref_btn.clicked.connect(self.set_reference_position)

        self.manual_control_btn = QPushButton("Activate manual control")
        self.manual_control_btn.clicked.connect(self.show_cnc_panel)

        self.exit_btn = QPushButton("Exit Program")
        self.exit_btn.setStyleSheet("background-color: #FF4444; color: white;")
        self.exit_btn.clicked.connect(self.close_program)

        control_layout.addWidget(self.record_btn, 0, 0)
        control_layout.addWidget(self.set_ref_btn, 0, 1)
        control_layout.addWidget(self.manual_control_btn, 1, 0)
        control_layout.addWidget(self.exit_btn, 1, 1)
        control_group.setLayout(control_layout)
        left_panel.addWidget(control_group)

        # HSV controls for GoPro
        hsv_group = QGroupBox("HSV Detection Settings")
        hsv_layout = QGridLayout()

        self.sliders = []
        hsv_labels = ['H1 Low', 'H1 High', 'H2 Low', 'H2 High',
                      'S Low', 'S High', 'V Low', 'V High']
        default_values = [0, 10, 170, 180, 120, 255, 70, 255]

        for i, (label, value) in enumerate(zip(hsv_labels, default_values)):
            hsv_layout.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 255 if 'H' not in label else 180)
            slider.setValue(value)
            slider.valueChanged.connect(self.update_hsv_thresholds)
            hsv_layout.addWidget(slider, i // 2, (i % 2) * 2 + 1)
            self.sliders.append(slider)

        hsv_group.setLayout(hsv_layout)
        left_panel.addWidget(hsv_group)

        # Right panel (status, +2D plane)
        right_panel = QVBoxLayout()

        # System status group
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("Connecting to GoPro...")
        self.status_label.setStyleSheet("font-size: 14px;")

        self.ref_pos_label = QLabel("Reference position: Not set")
        self.ref_pos_label.setStyleSheet("font-size: 14px;")

        self.recording_label = QLabel("Recording: Not recording")
        self.recording_label.setStyleSheet("font-size: 14px; color: red;")

        self.server_status_label = QLabel(f"Position server: Listening on port {POSITION_SERVER_PORT}")
        self.server_status_label.setStyleSheet("font-size: 14px; color: green;")

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.ref_pos_label)
        status_layout.addWidget(self.recording_label)
        status_layout.addWidget(self.server_status_label)
        status_group.setLayout(status_layout)
        right_panel.addWidget(status_group)
        # --- Ball counter (Duet) ---
        ball_group = QGroupBox("Ball Counter (Duet)")
        ball_layout = QHBoxLayout()
        ball_layout.addWidget(QLabel("Count:"))
        self.ballcount_edit = QLineEdit("—")
        self.ballcount_edit.setReadOnly(True)
        self.ballcount_edit.setAlignment(Qt.AlignCenter)
        self.ballcount_edit.setFixedWidth(100)
        ball_layout.addWidget(self.ballcount_edit, 0)
        self.ballcount_reset_btn = QPushButton("Reset")
        self.ballcount_reset_btn.clicked.connect(self.reset_ballcount)
        ball_layout.addWidget(self.ballcount_reset_btn, 0)
        # Presence indicator
        ball_layout.addSpacing(12)
        ball_layout.addWidget(QLabel("Detected:"))
        self.presence_value = QLabel("—")
        self.presence_value.setAlignment(Qt.AlignCenter)
        self.presence_value.setFixedWidth(90)
        ball_layout.addWidget(self.presence_value, 0)
        ball_group.setLayout(ball_layout)
        right_panel.addWidget(ball_group)


        # Current position group
        current_pos_group = QGroupBox("Current Position")
        current_pos_layout = QVBoxLayout()

        self.current_pos_label = QLabel("Current: N/A    |    Reference: N/A")
        current_pos_layout.addWidget(self.current_pos_label)
        current_pos_group.setLayout(current_pos_layout)
        right_panel.addWidget(current_pos_group)

        # Plano 2D con imagen y puntos
        plane_group = QGroupBox("Plan View (1000 mm × 1000 mm)")
        plane_layout = QVBoxLayout()
        self.points_canvas = PointsCanvas(self, mm_size=(1000.0, 1000.0), image_name="Test plate.png")
        self.points_canvas.setMinimumSize(500, 500)
        self.points_canvas.set_points(self.points)
        # conectar menús
        self.points_canvas.request_assign_index.connect(self._on_canvas_assign_index)
        self.points_canvas.request_erase_point.connect(self._on_canvas_erase_point)
        self.points_canvas.request_goto_point.connect(self._on_canvas_goto_point)
        plane_layout.addWidget(self.points_canvas)
        plane_group.setLayout(plane_layout)
        right_panel.addWidget(plane_group)

        # Add panels to main layout
        main_layout.addLayout(left_panel, 60)
        main_layout.addLayout(right_panel, 40)
        main_widget.setLayout(main_layout)

        self.setCentralWidget(main_widget)

    def update_recording_label(self, is_recording):
        if is_recording:
            self.recording_label.setText("Recording: 🔴 Recording")
            self.recording_label.setStyleSheet("font-size: 14px; color: red;")
        else:
            self.recording_label.setText("Recording: ⏹️ Not recording")
            self.recording_label.setStyleSheet("font-size: 14px; color: green;")

    def update_gopro_image(self, qt_image):
        self.gopro_label.setPixmap(QPixmap.fromImage(qt_image))

    def update_status(self, message):
        self.status_label.setText(message)

        if "Recording" in message:
            if "stopped" in message.lower():
                self.recording_label.setText("Recording: Not recording")
                self.recording_label.setStyleSheet("font-size: 14px; color: red;")
            elif "started" in message.lower():
                self.recording_label.setText("Recording: Active")
                self.recording_label.setStyleSheet("font-size: 14px; color: green;")

    def handle_position_update(self, position):
        self.current_position = position
        # solo para info visual
        cur = f"({position[0]}, {position[1]})"
        txt = self.current_pos_label.text()
        parts = txt.split("|")
        ref = parts[1] if len(parts) > 1 else "    Reference: N/A"
        self.current_pos_label.setText(f"Current: {cur}    |{ref}")

    #CSV points
    def load_points_from_csv(self):
        if not os.path.exists(self.points_csv_path):
            self.points = []
            return
        try:
            self.points = []
            with open(self.points_csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    try:
                        x = float(row[0].strip())
                        y = float(row[1].strip())
                        idx = None
                        if len(row) >= 3 and row[2].strip() != "":
                            raw = row[2].strip()
                            # si es entero (incluye negativos), guárdalo como int; si no, deja string (p.ej. "C0")
                            if raw.lstrip("-").isdigit():
                                idx = int(raw)
                            else:
                                idx = raw
                        self.points.append({'x': x, 'y': y, 'idx': idx})
                    except Exception as e:
                        print(f"Fila inválida en CSV: {row} ({e})")
            if hasattr(self, 'points_canvas'):
                self.points_canvas.set_points(self.points)
        except Exception as e:
            print(f"Failed to load points CSV: {e}")

    def write_points_to_csv(self):
        try:
            with open(self.points_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                for p in self.points:
                    w.writerow([f"{p['x']:.6f}", f"{p['y']:.6f}", "" if p.get('idx') is None else p['idx']])
        except Exception as e:
            print(f"Failed to write points CSV: {e}")

    # --- Operaciones de puntos desde el panel manual ---
    def add_point_from_duet(self, x: float, y: float):
        """Agrega punto y lo registra en CSV y canvas. Guarda índice de sesión para poder deshacer con '-'."""
        self.points.append({'x': float(x), 'y': float(y), 'idx': None})
        self.session_stack.append(len(self.points) - 1)
        self.points_canvas.set_points(self.points)
        self.write_points_to_csv()

    def remove_last_session_point(self):
        """Borra el último punto agregado en esta sesión (si existe) y compone la lista para no dejar huecos."""
        if not self.session_stack:
            self.update_status("No session point to remove")
            return
        last_idx = self.session_stack.pop()
        if 0 <= last_idx < len(self.points):
            # eliminar y 'compactar' es natural con pop; también hay que actualizar los índices de session_stack
            self.points.pop(last_idx)
            # re-map stack: si apuntaba a algo después, retrocede una posición
            self.session_stack = [i - 1 if i > last_idx else i for i in self.session_stack]
            self.points_canvas.set_points(self.points)
            self.write_points_to_csv()

    # --- Callbacks del canvas ---
    def _on_canvas_assign_index(self, i, new_idx):
        """Sincroniza el índice asignado en el canvas con el modelo y persiste."""
        if 0 <= i < len(self.points):
            self.points[i]['idx'] = int(new_idx)
            # Refresca canvas desde el modelo para mantener una sola fuente de verdad
            self.points_canvas.set_points(self.points)
            self.write_points_to_csv()

    def _on_canvas_erase_point(self, i):
        if 0 <= i < len(self.points):
            self.points.pop(i)
            # también ajustar la pila de sesión
            self.session_stack = [idx - 1 if idx > i else idx for idx in self.session_stack if idx != i]
            self.points_canvas.set_points(self.points)
            self.write_points_to_csv()

    def _duet_send(self, gcode: str, read_reply: bool = True, timeout: float = 3.0):
        """
        Envía un G-code a la Duet. Considera éxito si:
          - HTTP 200, y
          - 'err' no existe (caso común con RRF) o existe y es 0.
        Devuelve (ok, reply_text, data_json)
        """
        try:
            r = requests.get(
                f"http://{self.duet_ip}/rr_gcode",
                params={"gcode": gcode},
                timeout=timeout
            )
            ok = (r.status_code == 200)
            data = {}
            try:
                data = r.json()
            except Exception:
                pass
            if 'err' in data and data['err'] != 0:
                ok = False

            reply_text = ""
            if read_reply:
                try:
                    rr = requests.get(f"http://{self.duet_ip}/rr_reply", timeout=timeout)
                    if rr.status_code == 200:
                        reply_text = rr.text
                except Exception as e:
                    reply_text = f"(rr_reply error: {e})"

            return ok, reply_text, data
        except Exception as e:
            return False, f"(network error: {e})", {}

    def update_positions_table(self, positions):
        # Ya no hay tabla; si en el futuro quieres usar las posiciones recibidas como puntos,
        # puedes mapearlas aquí con self.add_point_from_duet(float(x), float(y))
        pass

    def set_reference_position(self):
        if self.current_position:
            self.gopro_manager.set_reference_position(self.current_position)
            self.ref_pos_label.setText(f"Reference position: ({self.current_position[0]}, {self.current_position[1]})")
            cur_txt = self.current_pos_label.text().split("|")[0]
            self.points.append({'x': self.current_position[0], 'y': self.current_position[1], 'idx': "c"})
            self.points_canvas.set_points(self.points)
            self.write_points_to_csv()

        else:
            QMessageBox.warning(self, "Warning", "No X position detected")

    def _on_canvas_goto_point(self, x: float, y: float):
        """
        Envía G-code absoluto para ir a (x,y) en la Duet.
        Si el CNCPanel está abierto, reutiliza su lógica; si no, manda el HTTP directo.
        """
        # Comandos: absoluto (G90) y movimiento con feedrate
        cmds = ["G90", f"G1 X{float(x):.3f} Y{float(y):.3f} F6000"]

        # Si el panel manual está abierto, usa su método (ya maneja la Duet y prints)
        if hasattr(self, "cnc_panel") and self.cnc_panel is not None and self.cnc_panel.isVisible():
            ok = self.cnc_panel.send_sequence(cmds)
            if ok:
                self.update_status(f"➡️ Moving to ({x:.3f}, {y:.3f}) via CNC panel")
            else:
                self.update_status("⚠️ Error sending G-code via CNC panel")
            return

        # Si no hay panel abierto, manda directo a la Duet
        try:
            ok1, rep1, data1 = self._duet_send("G90")  # absoluto
            time.sleep(0.05)  # pequeño respiro al firmware
            cmd_move = f"G1 X{float(x):.3f} Y{float(y):.3f} F6000"
            ok2, rep2, data2 = self._duet_send(cmd_move)

            if not ok2:
                QMessageBox.warning(self, "Duet",
                                    f"Error sending G-code:\n{cmd_move}\nReply: {rep2}\nData: {data2}")
            else:
                self.update_status(f"➡️ Moving to ({x:.3f}, {y:.3f})")
        except Exception as e:
            print("G-code error:", e)
            QMessageBox.warning(self, "Duet", f"Error sending G-code:\n{e}")

    def show_gcode_console(self):
        """Abre una ventanita para escribir y enviar G-codes a la Duet."""
        dlg = QDialog(self)
        dlg.setWindowTitle("G-code Console")
        dlg.setMinimumSize(500, 400)

        layout = QVBoxLayout(dlg)

        # Entrada multilinea
        input_row = QHBoxLayout()
        self.gcode_input = QLineEdit()  # si prefieres multilinea: usa QTextEdit()
        self.gcode_input.setPlaceholderText("Escribe un G-code (ej: G1 X10 Y20 F6000)")
        send_btn = QPushButton("Send command")
        send_btn.clicked.connect(self._send_gcode_from_console)
        input_row.addWidget(QLabel("Write command:"))
        input_row.addWidget(self.gcode_input, 1)
        input_row.addWidget(send_btn)
        layout.addLayout(input_row)

        # Respuesta
        layout.addWidget(QLabel("Response:"))
        from PyQt5.QtWidgets import QTextEdit
        self.gcode_response = QTextEdit()
        self.gcode_response.setReadOnly(True)
        layout.addWidget(self.gcode_response, 1)

        dlg.setLayout(layout)
        dlg.show()
        self._gcode_console_dlg = dlg  # mantener referencia para que no lo recoja el GC

    def _send_gcode_from_console(self):
        """Lee el texto de la consola y envía a la Duet; muestra respuesta."""
        cmd = self.gcode_input.text().strip()
        if not cmd:
            QMessageBox.information(self, "G-code", "Escribe un comando primero.")
            return
        ok, reply_text, data = self._duet_send(cmd)
        stamp = datetime.now().strftime("%H:%M:%S")
        self.gcode_response.append(f"[{stamp}] SENT: {cmd}\nREPLY: {reply_text}\nDATA: {data}\n{'-' * 40}\n")
        if not ok:
            QMessageBox.warning(self, "G-code", f"Duet devolvió error para:\n{cmd}\nReply: {reply_text}\nData: {data}")

    def show_draw_circle_dialog(self):
        """Ventana para pedir el radio y crear 10 puntos de círculo: C0..C9 a 36°."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Draw circle")
        dlg.setFixedSize(320, 120)
        v = QVBoxLayout(dlg)

        row = QHBoxLayout()
        row.addWidget(QLabel("Radio:"))
        self.circle_radius_input = QLineEdit()
        self.circle_radius_input.setValidator(QIntValidator(0, 1_000_000, self))
        self.circle_radius_input.setPlaceholderText("mm")
        row.addWidget(self.circle_radius_input, 1)
        create_btn = QPushButton("Create circle")
        create_btn.clicked.connect(self._create_circle_points)
        row.addWidget(create_btn)
        v.addLayout(row)

        dlg.setLayout(v)
        dlg.show()
        self._circle_dlg = dlg

    def _create_circle_points(self):
        """Genera C0..C9 alrededor del punto con idx==0 y escribe al CSV."""
        txt = self.circle_radius_input.text().strip()
        if not txt:
            QMessageBox.information(self, "Draw circle", "Ingresa un radio en mm.")
            return
        try:
            R = float(txt)
        except ValueError:
            QMessageBox.warning(self, "Draw circle", "Radio inválido.")
            return

        # Buscar centro: punto con idx==0 (entero 0)
        center = None
        for p in self.points:
            if p.get('idx') == 0:
                center = (float(p['x']), float(p['y']))
                break
        if center is None:
            QMessageBox.warning(self, "Draw circle", "No existe un punto con índice 0.")
            return

        cx, cy = center

        # Eliminar previos C0..C9
        new_points = []
        for p in self.points:
            idx = p.get('idx')
            if isinstance(idx, str) and idx.startswith('C') and idx[1:].isdigit():
                # saltar (se eliminará)
                continue
            new_points.append(p)
        self.points = new_points

        # Crear 10 puntos cada 36°
        import math
        for k in range(10):
            ang = math.radians(36 * k)  # 0..324
            x = cx + R * math.cos(ang)
            y = cy + R * math.sin(ang)
            self.points.append({'x': x, 'y': y, 'idx': f"C{k}"})

        # Refrescar UI y persistir
        self.points_canvas.set_points(self.points)
        self.write_points_to_csv()
        self.update_status(f"Circle created (R={R:.3f}) around idx 0: C0..C9")
        if hasattr(self, "_circle_dlg") and self._circle_dlg:
            self._circle_dlg.close()

    def handle_record_click(self):
        if not self.enable_gopro:
            QMessageBox.information(self, "GoPro", "GoPro está deshabilitado en esta plataforma.")
            return
        if not self.gopro_manager.loop:
            QMessageBox.information(self, "GoPro", "GoPro aún iniciando, intenta de nuevo en unos segundos.")
            return
        future = asyncio.run_coroutine_threadsafe(
            self.gopro_manager.toggle_recording(),
            self.gopro_manager.loop
        )

        def callback(fut):
            try:
                fut.result()
            except Exception as e:
                print("Recording error:", e)

        future.add_done_callback(callback)

        def callback(fut):
            try:
                result = fut.result()
                print("Recording ended:", result)
            except Exception as e:
                print("Recording error:", e)

        future.add_done_callback(callback)

    def show_cnc_panel(self):
        self.cnc_panel = CNCPanel(duet_ip="192.168.185.2", parent=self)
        self.cnc_panel.show()

    def _drain_pos_queue(self):
        # Vacía la cola y aplica la última actualización disponible
        try:
            # Procesa todo lo pendiente en este tick
            got_any = False
            while True:
                positions = self.position_server.queue.get_nowait()
                got_any = True
                # Reusa tu método actual
                self.update_positions_table(positions)
        except Exception as e:
            # queue.Empty u otros: simplemente ignora
            pass

    def start_test(self):
        """
        0 -> C0 -> 0 -> C1 -> 0 ... -> C9 with firmware-side waits:
          - G90 (absolute)
          - M400 (wait moves done)
          - G4 P2000 (2s dwell in firmware)
        """
        # center idx==0
        center = None
        for p in self.points:
            if p.get('idx') == 0:
                center = (float(p['x']), float(p['y']))
                break
        if center is None:
            QMessageBox.warning(self, "Start test", "No existe un punto con índice 0 (centro).")
            return

        # circle points C0..C9
        circle_pts = {}
        for p in self.points:
            idx = p.get('idx')
            if isinstance(idx, str) and idx.startswith('C') and idx[1:].isdigit():
                k = int(idx[1:])
                if 0 <= k <= 9:
                    circle_pts[k] = (float(p['x']), float(p['y']))
        missing = [k for k in range(10) if k not in circle_pts]
        if missing:
            QMessageBox.warning(self, "Start test", f"Faltan puntos del círculo: {missing}. Crea el círculo primero.")
            return

        cx, cy = center
        seq = []
        seq.append("G90")  # absolute mode
        # Optional (uncomment only if you must move without soft-limit blocks):
        # seq.append("M564 S0")           # allow moves outside soft limits (use with care)
        # seq.append("G54")               # pick work offset if you rely on it
        # seq.append("G21")               # millimeters (if needed)

        # go to center, wait, dwell 2s
        seq += [
            f"G1 X{cx:.3f} Y{cy:.3f} F6000",
            "M400",
            "G4 P2000",
        ]

        for k in range(10):
            xk, yk = circle_pts[k]
            seq += [
                f"G1 X{xk:.3f} Y{yk:.3f} F6000",
                "M400",
                "G4 P2000",
                f"G1 X{cx:.3f} Y{cy:.3f} F6000",
                "M400",
                "G4 P2000",
            ]

        # Send via panel if open; else direct
        if hasattr(self, "cnc_panel") and self.cnc_panel is not None and self.cnc_panel.isVisible():
            ok = self.cnc_panel.send_sequence(seq)
            if ok:
                self.update_status("▶ Test iniciado vía panel: 0↔C0..C9")
            else:
                QMessageBox.warning(self, "Start test", "Error enviando la secuencia vía panel.")
            return

        # Direct to Duet using your _duet_send helper
        for i, c in enumerate(seq):
            ok, reply, data = self._duet_send(c, read_reply=True)
            if not ok:
                QMessageBox.warning(
                    self, "Start test",
                    f"Error en comando #{i + 1}:\n{c}\nReply: {reply}\nData: {data}"
                )
                return
            # gentle throttle to avoid HTTP bursts; the real waiting is M400/G4
            time.sleep(0.03)

        self.update_status("▶ Test iniciado: 0↔C0..C9")

    
    def poll_duet_ballcount(self):
        """
        Poll Duet via HTTP RR model to read global.ballCount, and also presence (sensors.gpIn[J].value).
        """
        # ballCount
        try:
            r = requests.get(f"http://{self.duet_ip}/rr_model", params={"key":"global.ballCount", "flags":"v"}, timeout=0.6)
            if r.status_code == 200:
                data = r.json()
                res = data.get("result")
                if isinstance(res, dict) and "ballCount" in res:
                    val = res["ballCount"]
                else:
                    val = res
                if isinstance(val, (int, float, str)):
                    self.ballcount_edit.setText(str(val))
        except Exception:
            pass

        # presence
        self.update_presence_from_duet()

    def update_presence_from_duet(self):
        try:
            idx = getattr(self, "duet_gpIn_index", 1)
            r = requests.get(f"http://{self.duet_ip}/rr_model",
                             params={"key": f"sensors.gpIn[{idx}].value", "flags": "v"},
                             timeout=0.6)
            if r.status_code == 200:
                data = r.json()
                val = data.get("result")
                if isinstance(val, dict) and "value" in val:
                    val = val["value"]
                present = False
                try:
                    present = bool(int(val))
                except Exception:
                    present = bool(val)
                self.presence_value.setText("Metal" if present else "No metal")
                self.presence_value.setStyleSheet("color: green;" if present else "color: gray;")
        except Exception:
            pass

    def reset_ballcount(self):
        """
        Reset the Duet-side global.ballCount to 0 using rr_gcode.
        """
        try:
            # Send meta command through HTTP
            rq = requests.get(f"http://{self.duet_ip}/rr_gcode", params={"gcode":"set global.ballCount = 0"}, timeout=0.8)
            if rq.status_code == 200:
                self.ballcount_edit.setText("0")
        except Exception:
            pass
def update_hsv_thresholds(self):
        hsv_values = [s.value() for s in self.sliders]
        self.gopro_manager.update_hsv_values(hsv_values)

    def close_program(self):
        reply = QMessageBox.question(self, 'Exit',
                                     'Are you sure you want to exit the program?',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.close()

    def closeEvent(self, event):
        self.gopro_manager.stop()
        self.position_server.stop()
        # No intentes importar cv2 en Windows (evita cargar DLLs de OpenCV)
        event.accept()

if __name__ == "__main__":
    # Este atributo debe ponerse ANTES de crear QApplication
    if platform.system() == "Windows":
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, True)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
