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
import json
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
    QMenu, QInputDialog, QListWidget
)
from PyQt5.QtGui import QImage, QPixmap, QIntValidator, QPainter, QPen, QBrush

class TestTypeDialog(QDialog):
    """Modal dialog to choose which test to run before showing the execution monitor."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select test type")
        self.setFixedSize(360, 180)

        v = QVBoxLayout(self)

        lbl = QLabel("Choose a test to run:")
        lbl.setStyleSheet("font-size: 14px;")
        v.addWidget(lbl)

        btn_row = QHBoxLayout()
        self.btn_structural_integrity_b = QPushButton("Structural integrity (Steel balls)")
        self.btn_structural_integrity_n = QPushButton("Structural integrity (Nails)")
        self.btn_throwing_objects = QPushButton("Throwing objects")
        self.btn_impact_test = QPushButton("Impact test")
        btn_row.addWidget(self.btn_structural_integrity_b)
        btn_row.addWidget(self.btn_structural_integrity_n)
        btn_row.addWidget(self.btn_throwing_objects)
        btn_row.addWidget(self.btn_impact_test)
        v.addLayout(btn_row)

        h = QHBoxLayout()
        h.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        h.addWidget(self.btn_cancel)
        v.addLayout(h)

        self.choice = None
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_structural_integrity_b.clicked.connect(lambda: self._choose("structural_integrity_b"))
        self.btn_structural_integrity_n.clicked.connect(lambda: self._choose("structural_integrity_n"))
        self.btn_throwing_objects.clicked.connect(lambda: self._choose("throwing_objects"))
        self.btn_impact_test.clicked.connect(lambda: self._choose("impact_test"))

    def _choose(self, val: str):
        self.choice = val
        self.accept()


# --- Plano 2D con imagen + puntos ---
class PointsCanvas(QWidget):
    hovered_index_changed = pyqtSignal(int)          # For possile future hooks
    request_assign_index = pyqtSignal(int, int)      # (idx of point, new assigned index)
    request_erase_point = pyqtSignal(int)            # idx of point in list
    request_goto_point = pyqtSignal(float, float)

    def __init__(self, parent=None, mm_size=(1000.0, 1000.0), image_name="Test plate.png"):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.mm_w, self.mm_h = mm_size
        # Points list: [{'x': float, 'y': float, 'idx': Optional[int]}]
        self.points = []
        # Loads image for canvas
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
    
    def _get_idx0_mm(self):
        """Return (x_mm, y_mm) for point with idx == 0, or None if not found."""
        for p in self.points:
            if p.get('idx') == 0:
                return float(p['x']), float(p['y'])
        return None

    def _screen_center(self):
        """Center of the drawn image rect in screen coords."""
        if self.img_rect.isNull():
            return QPointF(self.width()/2.0, self.height()/2.0)
        return QPointF(self.img_rect.left() + self.img_rect.width()/2.0,
                       self.img_rect.top()  + self.img_rect.height()/2.0)

    def _scale_per_mm(self):
        """Return (sx_per_mm, sy_per_mm) to convert mm deltas to screen pixels."""
        if self.mm_w == 0 or self.mm_h == 0:
            return 1.0, 1.0
        return (self.img_rect.width() / self.mm_w,
                self.img_rect.height() / self.mm_h)

    def _point_to_screen(self, p):
        """
        Map a point dict {'x','y','idx'} to screen coords:
          - idx==0 -> fixed at image center
          - idx like 'C0'..'C9' -> relative to idx0 using mm deltas
          - otherwise -> legacy absolute mapping via _mm_to_screen
        """
        # Fallback: no image
        if self.img.isNull():
            return QPointF(p['x'], p['y'])
        idx0 = self._get_idx0_mm()
        if idx0 is None:
            # No center defined: use legacy mapping for everything
            return self._mm_to_screen(p['x'], p['y'])

        # When center exists:
        idx = p.get('idx')
        if idx == 0:
            return self._screen_center()
        if isinstance(idx, str) and idx.startswith('C') and idx[1:].isdigit():
            # relative to center in mm
            sx_per_mm, sy_per_mm = self._scale_per_mm()
            dx_mm = float(p['x']) - idx0[0]
            dy_mm = float(p['y']) - idx0[1]
            sc = self._screen_center()
            return QPointF(sc.x() + dx_mm * sx_per_mm, sc.y() + dy_mm * sy_per_mm)
        # Any other point uses legacy mapping
        return self._mm_to_screen(p['x'], p['y'])
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
            sp = self._point_to_screen(p)
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
            sp = self._point_to_screen(p)
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
    def __init__(self, duet_ip="192.168.185.2", parent=None, jog_steps=None):
        super().__init__(parent)
        self.setWindowTitle("Manual CNC Control Panel")
        self.setFixedSize(400, 600)

        # Configuración
        self.duet_ip = duet_ip
        self.feeder_velocity = 600
        self.single_jump = 1.0  # Paso por defecto
        self._jog_steps = jog_steps or [0.01, 0.1, 1, 10, 100]

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
        for value in self._jog_steps:
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
        # Performance knobs (Raspberry Pi)
        self.emit_fps = 12
        self._last_emit_ts = 0.0
        self.process_scale = 0.5
        self.frame_skip = 1
        self.cap_props_set = False
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
        self.retry_limit = 1

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
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, self.emit_fps)
            except Exception:
                pass
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
        self.loop.run_until_complete(self._run_with_retries())

    async def _run_with_retries(self):
        tries = int(getattr(self, 'retry_limit', 1) or 1)
        attempt = 0
        while self._run_flag and attempt < tries:
            attempt += 1
            try:
                ok = await self.initialize()
                if not ok:
                    self.status_update.emit(f"⚠️ Connection attempt {attempt}/{tries} failed")
                    if attempt < tries:
                        await asyncio.sleep(2)
                    continue
                await self.start_stream()
                # Process frames until stream ends or stop requested
                while self._run_flag and self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    # Throttle UI FPS
                    import time as _time
                    now = _time.time()
                    if (now - self._last_emit_ts) < (1.0 / max(1, self.emit_fps)):
                        continue
                    self._last_emit_ts = now
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
                # Clean up after stream ends
                if self.show_filters:
                    cv2.destroyWindow("Red Mask")
                    cv2.destroyWindow("Cleaned Mask")
                if self.cap:
                    self.cap.release()
                if self.gopro:
                    try:
                        await self.gopro.streaming.stop_active_stream()
                    except Exception:
                        pass
                # If we reach here without stop flag, consider it a failed attempt and let loop retry
                if not self._run_flag:
                    break
                self.status_update.emit(f"⚠️ Stream ended (attempt {attempt}/{tries})")
            except Exception as e:
                self.status_update.emit(f"⚠️ GoPro run error (attempt {attempt}/{tries}): {e}")
            if attempt < tries and self._run_flag:
                self.status_update.emit("🔄 Retrying in 2s...")
                await asyncio.sleep(2)
        # Out of tries
        if attempt >= tries:
            self.status_update.emit("⛔ Could not connect to GoPro after retries.")
        if self.gopro:
            try:
                await self.gopro.close()
            except Exception:
                pass

    def process_frame(self, frame):
        x_start, x_end = 480, 1440
        y_start, y_end = 270, 810
        roi = frame[y_start:y_end, x_start:x_end]
        scale = float(getattr(self, 'process_scale', 1.0))
        if 0.2 <= scale < 1.0:
            roi_small = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            roi_small = roi
            scale = 1.0
        hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
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
        processed_frame = frame
        current_pos = None
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > (500 * (scale*scale)):
                x, y, w, h = cv2.boundingRect(largest_contour)
                x = int(x / scale); y = int(y / scale); w = int(w / scale); h = int(h / scale)
                x += x_start; y += y_start
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

class Preferences:
    """Tiny JSON-backed settings store."""
    def __init__(self, path):
        self.path = path
        self.defaults = {
            "calibration": {
                "coarse_step": 50.0,
                "coarse_F": 6000,
                "fine_step": 10.0,
                "fine_F": 3000,
                "max_iters": 400,
                "pixel_tolerance": 2
            },
            "manual": {
                "jog_steps": [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            "duet": {
                "output_pin": 3,
                "toggle_on_S": 1,
                "toggle_off_S": 0
            },
            "servo": {
                "channel": 0,
                "s_low": 0,
                "s_high": 3000,
                "dwell_seconds": 1
            },
            "test": {
                "block_repeats": 10,
                "struct_per_point": 10,
                "throw_per_point": 10,
                "throw_big_iters": 2
            },
            "servo_positions": {
                "collect": 0,
                "deposit": 3000
            },
            "nails": {
                "spindle_velocity": 1200
            },
            "calibration_dir": {
                "dir_x": 1,
                "dir_y": 1
            }
        }
        self.data = {}
        self.load()

    def load(self):
        try:
            with open(self.path, "r") as f:
                self.data = json.load(f)
        except Exception:
            self.data = json.loads(json.dumps(self.defaults))
            self.save()

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print("Preferences save error:", e)

    def get(self, kpath, fallback=None):
        ref = self.data
        for k in kpath.split("."):
            if isinstance(ref, dict) and k in ref:
                ref = ref[k]
            else:
                return fallback
        return ref

    def set(self, kpath, value):
        parts = kpath.split(".")
        ref = self.data
        for k in parts[:-1]:
            ref = ref.setdefault(k, {})
        ref[parts[-1]] = value
        self.save()

class EditPreferencesDialog(QDialog):
    def __init__(self, prefs: 'Preferences', parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit preferences")
        self.setMinimumSize(520, 420)
        self.prefs = prefs

        v = QVBoxLayout(self)

        rows = [
            ("Calibration coarse step (mm)", "calibration.coarse_step"),
            ("Calibration coarse F (mm/min)", "calibration.coarse_F"),
            ("Calibration fine step (mm)", "calibration.fine_step"),
            ("Calibration fine F (mm/min)", "calibration.fine_F"),
            ("Calibration pixel tolerance", "calibration.pixel_tolerance"),
            ("Calibration max iters", "calibration.max_iters"),
            ("Manual jog steps (comma-sep)", "manual.jog_steps"),
            ("Duet output pin (M42 P...)", "duet.output_pin"),
            ("Duet toggle ON (S...)", "duet.toggle_on_S"),
            ("Duet toggle OFF (S...)", "duet.toggle_off_S"),
            ("Servo channel (M280 P...)", "servo.channel"),
            ("Servo S low (M280 S...)", "servo.s_low"),
            ("Servo S high (M280 S...)", "servo.s_high"),
            ("Servo dwell seconds (G4 S...)", "servo.dwell_seconds"),
            ("Test block repeats", "test.block_repeats"),
            ("Struct. per point", "test.struct_per_point"),
            ("Throw per point", "test.throw_per_point"),
            ("Throw big iterations", "test.throw_big_iters"),
            ("Nails spindle velocity (F for E)", "nails.spindle_velocity"),
            ("Collect position (S)", "servo_positions.collect"),
            ("Deposit position (S)", "servo_positions.deposit"),
            ("Calibration dir X (+1/-1)", "calibration_dir.dir_x"),
            ("Calibration dir Y (+1/-1)", "calibration_dir.dir_y"),
        ]

        self.inputs = []
        for label, key in rows:
            h = QHBoxLayout()
            h.addWidget(QLabel(f"{label}:"))
            edit = QLineEdit()
            val = self.prefs.get(key)
            if isinstance(val, list):
                edit.setText(", ".join(str(x) for x in val))
            else:
                edit.setText(str(val))
            btn = QPushButton("Change")
            btn.clicked.connect(lambda _, e=edit, k=key: self._commit(k, e.text()))
            h.addWidget(edit, 1)
            h.addWidget(btn)
            v.addLayout(h)

            self.inputs.append((key, edit))

        close_row = QHBoxLayout()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        close_row.addStretch(1)
        close_row.addWidget(close)
        v.addLayout(close_row)

    def _commit(self, key, raw):
        try:
            if key == "manual.jog_steps":
                lst = [float(s) for s in raw.split(",") if s.strip() != ""]
                self.prefs.set(key, lst)
            elif key.endswith(("_step", "_F")):
                self.prefs.set(key, float(raw))
            elif key.endswith(("pixel_tolerance", "max_iters", "output_pin", "toggle_on_S", "toggle_off_S", "channel", "s_low", "s_high", "dwell_seconds", "struct_per_point", "throw_per_point", "throw_big_iters", "collect", "deposit", "dir_x", "dir_y")):
                self.prefs.set(key, int(float(raw)))
            else:
                self.prefs.set(key, raw)
            QMessageBox.information(self, "Saved", f"{key} updated.")
        except Exception as e:
            QMessageBox.warning(self, "Invalid value", f"Could not parse value for {key}:\n{e}")

class CalibrateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate")
        self.setFixedSize(360, 240)
        v = QVBoxLayout(self)

        def row(lbl):
            h = QHBoxLayout()
            h.addWidget(QLabel(lbl))
            e = QLineEdit()
            h.addWidget(e, 1)
            v.addLayout(h)

            return e

        self.x_off = row("Offset X (mm):")
        self.y_off = row("Offset Y (mm):")
        self.radius = row("Radius (mm):")
        self.msg = QLabel("")
        v.addWidget(self.msg)

        hbtn = QHBoxLayout()
        self.start_btn = QPushButton("Start calibration")
        self.cancel_btn = QPushButton("Cancel")
        hbtn.addStretch(1)
        hbtn.addWidget(self.start_btn)
        hbtn.addWidget(self.cancel_btn)
        v.addLayout(hbtn)

        self.cancel_btn.clicked.connect(self.reject)

    def values(self):
        return self.x_off.text().strip(), self.y_off.text().strip(), self.radius.text().strip()

class CalibrationWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, dict)

    def __init__(self, main: 'MainWindow', x_off: float, y_off: float, radius: float):
        super().__init__(main)
        self.main = main
        self.x_off = x_off
        self.y_off = y_off
        self.radius = radius
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            prefs = self.main.prefs
            coarse = float(prefs.get("calibration.coarse_step", 50.0))
            coarseF = int(prefs.get("calibration.coarse_F", 6000))
            fine = float(prefs.get("calibration.fine_step", 10.0))
            fineF = int(prefs.get("calibration.fine_F", 3000))
            max_iters = int(prefs.get("calibration.max_iters", 400))
            px_tol = int(prefs.get("calibration.pixel_tolerance", 2))

            # 1) Home using requested command
            self.progress.emit('Homing...')
            ok, reply, _ = self.main._duet_send('M559 P"homeall.g"', read_reply=True)
            if not ok:
                self.finished.emit(False, f"Homing failed (M559): {reply}", {})
                return

            def move_rel(dx=0.0, dy=0.0, F=3000):
                seq = ["G91"]
                if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                    seq.append(f"G1 X{dx:.3f} Y{dy:.3f} F{F}")
                seq += ["G90", "M400"]
                for c in seq:
                    ok, rep, _ = self.main._duet_send(c, read_reply=True)
                    if not ok:
                        raise RuntimeError(f"Gcode error: {c} → {rep}")
                    time.sleep(0.01)

            def have_pos():
                return self.main.current_position is not None

            def center_of_frame():
                pm = self.main.gopro_label.pixmap()
                if pm is None or pm.isNull():
                    return None
                return (pm.width()//2, pm.height()//2)

            def drive_axis_to_center(axis):
                iters = 0
                while iters < max_iters and not self._stop:
                    iters += 1
                    if not have_pos():
                        time.sleep(0.02)
                        continue
                    cp = self.main.current_position
                    cf = center_of_frame()
                    if cf is None:
                        time.sleep(0.02)
                        continue
                    errx = cp[0] - cf[0]
                    erry = cp[1] - cf[1]
                    if axis == 'X':
                        err = errx
                        sgn = -1.0
                        if abs(err) <= px_tol:
                            break
                        step = coarse if abs(err) > (2*px_tol) else fine
                        F = coarseF if step == coarse else fineF
                        move_rel(dx=sgn * (step if err > 0 else -step), dy=0.0, F=F)
                    else:
                        err = erry
                        sgn = -1.0
                        if abs(err) <= px_tol:
                            break
                        step = coarse if abs(err) > (2*px_tol) else fine
                        F = coarseF if step == coarse else fineF
                        move_rel(dx=0.0, dy=sgn * (step if err > 0 else -step), F=F)
                return iters < max_iters

            # 2) Coarse search for first X
            self.progress.emit("Searching first X (coarse)...")
            scan_iters = 0
            while not have_pos() and scan_iters < max_iters and not self._stop:
                move_rel(dx=coarse, dy=0.0, F=coarseF)
                scan_iters += 1
                if have_pos():
                    break
                if scan_iters % 10 == 0 and not have_pos():
                    move_rel(dx=0.0, dy=coarse, F=coarseF)

            if not have_pos():
                self.finished.emit(False, "Could not find first red X.", {})
                return

            # 3) Center first X (X then Y)
            self.progress.emit("Centering first X horizontally...")
            if not drive_axis_to_center('X'):
                self.finished.emit(False, "Centering X (first) failed.", {})
                return
            self.progress.emit("Centering first X vertically...")
            if not drive_axis_to_center('Y'):
                self.finished.emit(False, "Centering Y (first) failed.", {})
                return

            x1, y1 = self.main.get_current_position_from_duet()
            if x1 is None or y1 is None:
                self.finished.emit(False, "Failed to read Duet pos for M1.", {})
                return
            M1 = (float(x1), float(y1))
            self.progress.emit(f"M1 = {M1}")

            # 4) Find second mark
            self.progress.emit("Searching second X (coarse)...")
            found2 = False
            for i in range(max_iters // 4):
                move_rel(dx=coarse, dy=0.0, F=coarseF)
                if have_pos():
                    found2 = True
                    break
                if i % 10 == 0:
                    move_rel(dx=0.0, dy=coarse, F=coarseF)
                    if have_pos():
                        found2 = True
                        break
            if not found2:
                self.finished.emit(False, "Could not find second red X.", {})
                return

            self.progress.emit("Centering second X horizontally...")
            if not drive_axis_to_center('X'):
                self.finished.emit(False, "Centering X (second) failed.", {})
                return
            self.progress.emit("Centering second X vertically...")
            if not drive_axis_to_center('Y'):
                self.finished.emit(False, "Centering Y (second) failed.", {})
                return

            x2, y2 = self.main.get_current_position_from_duet()
            if x2 is None or y2 is None:
                self.finished.emit(False, "Failed to read Duet pos for M2.", {})
                return
            M2 = (float(x2), float(y2))
            self.progress.emit(f"M2 = {M2}")

            # Midpoint + offsets
            midx = (M1[0] + M2[0]) / 2.0 + self.x_off
            midy = (M1[1] + M2[1]) / 2.0 + self.y_off

            # Update points
            self.main.points = [p for p in self.main.points if not (p.get('idx') == 0)]
            self.main.points.append({'x': M1[0], 'y': M1[1], 'idx': "M1"})
            self.main.points.append({'x': M2[0], 'y': M2[1], 'idx': "M2"})
            self.main.points.append({'x': midx, 'y': midy, 'idx': 0})
            self.main.points_canvas.set_points(self.main.points)
            self.main.write_points_to_csv()

            # Draw circle
            self.main._create_circle_from_center_and_radius(center=(midx, midy), R=self.radius)

            self.finished.emit(True, "Calibration complete.", {"M1": M1, "M2": M2, "center": (midx, midy)})

        except Exception as e:
            self.finished.emit(False, f"Calibration error: {e}", {})

class TestRunnerDialog(QDialog):
    pause_clicked = pyqtSignal()
    resume_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()

    def __init__(self, total_commands:int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start test — Execution Monitor")
        self.setMinimumSize(700, 520)

        grid = QGridLayout(self)

        # Left: Emergency stop
        left = QVBoxLayout()
        self.emerg_btn = QPushButton("Emergency Stop")
        self.emerg_btn.setStyleSheet(
            "font-size:22px; background-color:#b71c1c; color:white; border-radius:12px; padding:24px;"
        )
        self.emerg_btn.setFixedHeight(160)
        left.addWidget(self.emerg_btn, 0)
        left.addStretch(1)

        # Right: buffer
        v = QVBoxLayout()

        # Current
        cur_group = QGroupBox("Current")
        cur_layout = QGridLayout()
        self.exec_label = QLabel('Executing command: —')
        self.state_label = QLabel('State: —')
        self.state_label.setStyleSheet("font-weight: bold;")
        cur_layout.addWidget(self.exec_label, 0, 0, 1, 2)
        cur_layout.addWidget(self.state_label, 1, 0, 1, 2)
        cur_group.setLayout(cur_layout)
        v.addWidget(cur_group)

        # Rod controls (hidden by default)
        self.rod_group = QGroupBox("Rod Control")
        self.rod_group.setVisible(False)
        rg = QHBoxLayout()
        self.insert_btn = QPushButton("Insert Steel Rod")
        self.retract_btn = QPushButton("Retract Steel Rod")
        self.insert_btn.setEnabled(False)
        self.retract_btn.setEnabled(False)
        rg.addWidget(self.insert_btn)
        rg.addWidget(self.retract_btn)
        self.rod_group.setLayout(rg)
        v.addWidget(self.rod_group)

        # Interim prompt (hidden by default)
        self.interim_group = QGroupBox("Interim evaluation")
        self.interim_group.setVisible(False)
        ig = QHBoxLayout()
        ig.addWidget(QLabel("Interim evaluation done?"))
        self.interim_done_btn = QPushButton("Done")
        self.interim_cancel_btn = QPushButton("Cancel operation")
        ig.addWidget(self.interim_done_btn)
        ig.addWidget(self.interim_cancel_btn)
        self.interim_group.setLayout(ig)
        v.addWidget(self.interim_group)

        # Nails group
        self.nails_group = QGroupBox("Nails control")
        self.nails_group.setVisible(False)
        ng = QHBoxLayout()
        self.shoot_nails_btn = QPushButton("Shoot nails")
        self.shoot_nails_btn.setEnabled(False)
        ng.addWidget(self.shoot_nails_btn)
        self.nails_group.setLayout(ng)
        v.addWidget(self.nails_group)

        # Next commands
        next_group = QGroupBox("Next commands (look-ahead 10)")
        next_layout = QVBoxLayout()
        self.next_list = QListWidget()
        next_layout.addWidget(self.next_list, 1)
        next_group.setLayout(next_layout)
        v.addWidget(next_group, 1)

        # Controls
        h = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.cancel_btn = QPushButton("Cancel Test")
        h.addStretch(1)
        h.addWidget(self.pause_btn)
        h.addWidget(self.resume_btn)
        h.addWidget(self.cancel_btn)
        v.addLayout(h)

        grid.addLayout(left, 0, 0)
        grid.addLayout(v, 0, 1)

        # Signals (buffer controls)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        self.emerg_btn.clicked.connect(lambda: parent._duet_send('M112', read_reply=False))
        self.resume_btn.clicked.connect(self.resume_clicked.emit)
        def _confirm_cancel():
            reply = QMessageBox.question(
                self, "Cancel test", "Are you sure you want to cancel the test?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.cancel_clicked.emit()
        self.cancel_btn.clicked.connect(_confirm_cancel)

        # Rod / interim flags (the runner will poll these)
        self._insert_clicked = False
        self._retract_clicked = False
        self._interim_done = False
        self._interim_cancel = False
        self.insert_btn.clicked.connect(lambda: setattr(self, "_insert_clicked", True))
        self.retract_btn.clicked.connect(lambda: setattr(self, "_retract_clicked", True))
        self.interim_done_btn.clicked.connect(lambda: setattr(self, "_interim_done", True))
        self.interim_cancel_btn.clicked.connect(lambda: setattr(self, "_interim_cancel", True))
        self._shoot_nails_clicked = False
        self.shoot_nails_btn.clicked.connect(lambda: setattr(self, "_shoot_nails_clicked", True))

        self.total_commands = total_commands

    # Helpers the runner can call
    def show_rod_controls(self, visible=True):
        self.rod_group.setVisible(visible)
        self.insert_btn.setEnabled(False)
        self.retract_btn.setEnabled(False)

    def enable_insert(self, enabled=True):
        self.insert_btn.setEnabled(enabled)

    def enable_retract(self, enabled=True):
        self.retract_btn.setEnabled(enabled)

    def show_nails_controls(self, visible=True):
        self.nails_group.setVisible(visible)
        self.shoot_nails_btn.setEnabled(False)

    def enable_shoot_nails(self, enabled=True):
        self.shoot_nails_btn.setEnabled(enabled)

    def show_interim_prompt(self, visible=True):
        self.interim_group.setVisible(visible)

    # Buffer updates
    def set_current(self, cmd: str, state: str):
        self.exec_label.setText(f'Executing command: "{cmd}"')
        st = state
        if state.lower() == "executing":
            self.state_label.setStyleSheet("color: #1565c0; font-weight: bold;")
        elif state.lower() == "executed":
            self.state_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        elif state.lower() == "paused":
            self.state_label.setStyleSheet("color: #f9a825; font-weight: bold;")
        elif state.lower() == "error":
            self.state_label.setStyleSheet("color: #c62828; font-weight: bold;")
        else:
            self.state_label.setStyleSheet("")
        self.state_label.setText(f"State: {st}")

    def set_next(self, commands_slice):
        self.next_list.clear()
        for c in commands_slice:
            self.next_list.addItem(c)

class CommandRunner(QThread):
    sig_executing = pyqtSignal(int, str)
    sig_executed  = pyqtSignal(int, str)
    sig_error     = pyqtSignal(int, str, str)
    sig_nextslice = pyqtSignal(list)
    # (keep any other signals you already have, e.g., sig_shot)

    def __init__(self, main: 'MainWindow', commands: list, wait_after_ok_sec: float = 1.0):
        super().__init__(main)
        from threading import Event
        self.main = main
        self.commands = commands or []
        # NEW: store the delay between successful commands (seconds)
        try:
            self.wait_after_ok_sec = float(wait_after_ok_sec)
        except Exception:
            self.wait_after_ok_sec = 1.0

        self._paused = Event()
        self._paused.clear()
        self._cancel = False
        self._index = 0
        # if you use this flag in your code, keep it
        self._repeat_on_resume = False
        # this will be set by _run_commands_with_monitor
        self.dialog = None

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def cancel(self):
        self._cancel = True

    # --- helpers for motion gating ---
    def _parse_target_xy(self, cmd: str):
        if not cmd or ('G0' not in cmd and 'G1' not in cmd):
            return None
        import re
        mX = re.search(r"X(-?\d+(?:\.\d+)?)", cmd)
        mY = re.search(r"Y(-?\d+(?:\.\d+)?)", cmd)
        if mX or mY:
            tx = float(mX.group(1)) if mX else None
            ty = float(mY.group(1)) if mY else None
            return (tx, ty)
        return None

    def _gui(self, fn):
        """Schedule a GUI call on the main thread."""
        if self.dialog:
            QTimer.singleShot(0, fn)

    def _wait_until_position(self, target, tol=0.05, max_ms=10000):
        if not target:
            return True
        tx, ty = target
        import time
        start = time.time()
        while (time.time() - start) * 1000 < max_ms and not self._cancel:
            x, y = self.main.get_current_position_from_duet()
            try:
                x = float(x) if x is not None else None
                y = float(y) if y is not None else None
            except:
                x = y = None
            okx = (tx is None) or (x is not None and abs(x - tx) <= tol)
            oky = (ty is None) or (y is not None and abs(y - ty) <= tol)
            if okx and oky:
                return True
            self.msleep(100)
        return False

    # --- pseudo-commands (UI/camera/sensor logic) ---
    def _handle_pseudo(self, cmd: str):
        try:
            if not cmd.startswith('::'):
                return False
            main = self.main
            token = cmd[2:].strip()

            # Sleep in seconds
            if token.startswith('SLEEP'):
                parts = token.split()
                sec = float(parts[1]) if len(parts) > 1 else 1.0
                ms = max(0, int(sec * 1000))
                elapsed = 0
                step = 100
                while elapsed < ms and not self._cancel and not self._paused.is_set():
                    self.msleep(min(step, ms - elapsed))
                    elapsed += step
                return True

            # Camera control
            if token == 'CAM_START':
                try:
                    if main.gopro_manager and main.gopro_manager.loop:
                        import asyncio
                        fut = asyncio.run_coroutine_threadsafe(
                            main.gopro_manager.http_command.set_shutter(shutter=0), main.gopro_manager.loop
                        )
                        fut.result(timeout=4)
                except Exception:
                    pass
                return True

            if token == 'CAM_STOP':
                try:
                    if main.gopro_manager and main.gopro_manager.loop:
                        import asyncio
                        fut = asyncio.run_coroutine_threadsafe(
                            main.gopro_manager.http_command.set_shutter(shutter=1), main.gopro_manager.loop
                        )
                        fut.result(timeout=4)
                except Exception:
                    pass
                return True

            # Structural cycle (sensor-driven). Format: ::STRUCT_CYCLE N=<int>
            if token.startswith('STRUCT_CYCLE_B'):
                target = 10
                try:
                    import re as _re
                    m = _re.search(r"N=(\d+)", token)
                    if m: target = int(m.group(1))
                except Exception:
                    pass
                self._structural_cycle_b(target)
                try:
                    self.main._duet_send("G90", read_reply=True)
                    self.main._duet_send("M400", read_reply=True)
                except Exception:
                    pass
                return True

            if token.startswith('STRUCT_CYCLE_N'):
                target = 20
                try:
                    import re as _re
                    m = _re.search(r"N=(\d+)", token)
                    if m: target = int(m.group(1))
                except Exception:
                    pass
                self._structural_cycle_n(target)
                self._structural_cycle_n(target)
                try:
                    self.main._duet_send("G90", read_reply=True)
                    self.main._duet_send("M400", read_reply=True)
                except Exception:
                    pass
                return True

            # Rod controls (Impact test)
            if token == 'BTN_SHOW_ROD':
                if self.dialog:
                    self.dialog.show_rod_controls(True)
                return True
            if token == 'BTN_ENABLE_INSERT':
                if self.dialog:
                    self.dialog.enable_insert(True)
                return True
            if token == 'WAIT_BTN_INSERT':
                if self.dialog:
                    while not getattr(self.dialog, "_insert_clicked", False) and not self._cancel:
                        if self._paused.is_set(): self.msleep(100); continue
                        self.msleep(80)
                return True
            if token == 'BTN_ENABLE_RETRACT':
                if self.dialog:
                    self.dialog.enable_retract(True)
                return True
            if token == 'WAIT_BTN_RETRACT':
                if self.dialog:
                    while not getattr(self.dialog, "_retract_clicked", False) and not self._cancel:
                        if self._paused.is_set(): self.msleep(100); continue
                        self.msleep(80)
                return True

            # Interim prompt (Throwing objects)
            if token == 'BTN_SHOW_NAILS':
                if self.dialog:
                    self.dialog.show_nails_controls(True)
                return True
            if token == 'BTN_ENABLE_SHOOT_NAILS':
                if self.dialog:
                    self.dialog.enable_shoot_nails(True)
                return True
            if token == 'WAIT_BTN_SHOOT_NAILS':
                if self.dialog:
                    # wait for operator press
                    while not getattr(self.dialog, '_shoot_nails_clicked', False) and not self._cancel:
                        if self._paused.is_set():
                            self.msleep(100); continue
                        self.msleep(80)
                    # reset flag so next shot waits again
                    try:
                        setattr(self.dialog, '_shoot_nails_clicked', False)
                    except Exception:
                        pass
                return True

            if token == 'INTERIM_PROMPT_SHOW':
                if self.dialog:
                    self.dialog.show_interim_prompt(True)
                return True
            if token == 'INTERIM_PROMPT_WAIT':
                if self.dialog:
                    while not (getattr(self.dialog, "_interim_done", False) or getattr(self.dialog, "_interim_cancel", False)) and not self._cancel:
                        if self._paused.is_set(): self.msleep(100); continue
                        self.msleep(80)
                    if getattr(self.dialog, "_interim_cancel", False):
                        self.sig_error.emit(self._index, "Interim cancelled by operator", "User cancelled")
                        self._paused.set()
                return True

            return False
        except Exception:
            return False

    def _read_sensor_ball(self, idx=None):
        try:
            return bool(self.main.read_gpin(idx or self.main.duet_gpIn_index))
        except Exception:
            return False

    def _structural_cycle_b(self, target: int):
        """Implements the PDF's loading/shooting loops until 'target' shots."""
        main = self.main
        pin = main.prefs.get("duet.output_pin", 3)
        servoP = main.prefs.get("servo.channel", 0)
        S_collect = main.prefs.get("servo_positions.collect", main.prefs.get("servo.s_low", 0))
        S_deposit = main.prefs.get("servo_positions.deposit", main.prefs.get("servo.s_high", 3000))
        dwell = int(main.prefs.get("servo.dwell_seconds", 1))
        temp = 0

        # 5) If ball already in chamber -> one shot
        if self._read_sensor_ball():
            main._duet_send(f"M42 P{pin} S1", read_reply=True); self.msleep(500)
            main._duet_send(f"M42 P{pin} S0", read_reply=True); self.msleep(500)
            temp += 1

        # 6) Wait
        self.msleep(500)

        # 7) While temp < target:
        while temp < target and not self._cancel:
            # 7.1) While no ball present -> load attempts
            while not self._read_sensor_ball() and not self._cancel:
                main._duet_send("M400 M83 G1 E20 F600", read_reply=True); self.msleep(2000)
                main._duet_send(f" M280 P{servoP} S{S_collect}", read_reply=True); self.msleep(1000)
                main._duet_send(f" M280 P{servoP} S{S_deposit}", read_reply=True);
                waited = 0
                while waited < 200 and not self._cancel:
                    if self._read_sensor_ball(): break
                    self.msleep(50); waited += 50

            # 7.2) While ball present -> shoot and increment
            self.msleep(100)
            while self._read_sensor_ball() and temp < target and not self._cancel:
                main._duet_send(f"M42 P{pin} S1", read_reply=True); temp += 1; self.msleep(500)
                main._duet_send(f"M42 P{pin} S0", read_reply=True); self.msleep(500)
                # Confirm chamber clear
                waited = 0
                while waited < 900 and not self._cancel:
                    if not self._read_sensor_ball(): break
                    self.msleep(50); waited += 50

    def _structural_cycle_n(self, target: int):
        """
        Nails routine:
          - Operator presses 'Shoot nails' -> button disables -> spindle spins -> valve fires -> reset
          - Wait again for operator -> repeat
          - Each shot counts as 10; e.g., target=20 => 2 shots in total.
        """
        main = self.main
        pin = int(main.prefs.get("duet.output_pin", 3))
        spindle_F = int(main.prefs.get("nails.spindle_velocity", 1200))  # F for E axis

        SHOT_WEIGHT = 10  # one operator-triggered shot equals 10 "units"

        # Show nails controls
        if self.dialog:
            self.dialog.show_nails_controls(True)
        self.msleep(300)

        shots_progress = 0
        while shots_progress <= target and not self._cancel:
            # --- Wait for operator ---
            if self.dialog:
                self.dialog.enable_shoot_nails(True)
            # block until click (or pause/cancel)
            while self.dialog and not getattr(self.dialog, "_shoot_nails_clicked", False) and not self._cancel:
                if self._paused.is_set():
                    self.msleep(100)
                    continue
                self.msleep(50)
            if self._cancel:
                break

            # Disable button immediately to avoid double-trigger
            if self.dialog:
                self.dialog.enable_shoot_nails(False)
                try:
                    setattr(self.dialog, "_shoot_nails_clicked", False)
                except Exception:
                    pass

            # --- Do one shot ---
            # (1) Spin spindle using E-axis at configured feed
            main._duet_send("M83", read_reply=True)  # relative extrusion mode
            main._duet_send("G91", read_reply=True)  # relative positioning
            main._duet_send(f"G1 E200 F{spindle_F}", read_reply=True)  # start rotation (short runup)
            main._duet_send("G90", read_reply=True)  # back to absolute to protect XY moves

            # (2) Fire valve
            main._duet_send(f"M42 P{pin} S1", read_reply=True)
            self.msleep(3000)  # pulse / dwell time (ms)
            main._duet_send(f"M42 P{pin} S0", read_reply=True)

            # (3) Reset extruder distance (keeps future E moves sane)
            main._duet_send("G92 E0", read_reply=True)

            # Count this shot
            shots_progress += SHOT_WEIGHT

        # Safety: ensure absolute mode before returning to XY motion
        try:
            main._duet_send("G90", read_reply=True)
            main._duet_send("M400", read_reply=True)
        except Exception:
            pass

        return not self._cancel

    def run(self):
        total = len(self.commands)
        while self._index < total and not self._cancel:
            while self._paused.is_set() and not self._cancel:
                self.msleep(100)
            if self._cancel:
                break

            cmd = self.commands[self._index]
            self.sig_executing.emit(self._index, cmd)

            # Pseudo-commands don't go to Duet
            if cmd.startswith('::'):
                handled = self._handle_pseudo(cmd)
                ok = True if handled else False
                reply_text, data = "", {}
            else:
                ok, reply_text, data = self.main._duet_send(cmd, read_reply=True, timeout=5.0)

            has_err_flag = (isinstance(data, dict) and data.get('err', 0) != 0)
            reply_lower = (reply_text or "").lower()
            looks_error = ("error" in reply_lower) or ("bad command" in reply_lower) or ("invalid" in reply_lower)
            # Ignore common Duet meta spam that does not block motion
            if "meta command: expected boolean operand" in reply_lower:
                looks_error = False
            if not ok or has_err_flag or looks_error:
                err_msg = reply_text if reply_text else (data if data else "Unknown error")
                self.sig_error.emit(self._index, cmd, str(err_msg))
                self._paused.set()
                while self._paused.is_set() and not self._cancel:
                    self.msleep(200)
                if self._cancel:
                    break
                continue

            self.sig_executed.emit(self._index, cmd)

            # Motion gating for G0/G1 with XY
            target = None
            try:
                target = self._parse_target_xy(cmd)
            except Exception:
                pass
            if target:
                self._wait_until_position(target)

            self.sleep(int(self.wait_after_ok_sec))
            self._index += 1

            start = self._index
            self.sig_nextslice.emit(self.commands[start:start+10])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Preferences store
        self.prefs_path = os.path.join(os.path.dirname(__file__), "preferences.json")
        self.prefs = Preferences(self.prefs_path)
        self.jog_steps_for_panel = self.prefs.get("manual.jog_steps", [0.01,0.1,1,10,100])
        self.setWindowTitle("Integrated Monitoring System for DEKRA Test Center Dresden by Fabian Romero")
        self.setGeometry(100, 100, 1400, 800)
        print("[INIT] MainWindow starting...")
        self.duet_ip = "192.168.185.2"  # ajusta si tu IP es otra
        import requests as _rq
        self.http = _rq.Session()
        self.http.headers.update({'Connection': 'keep-alive'})

        # --- Setup GoPro (siempre habilitado) ---
        self.enable_gopro = True

        self.gopro_manager = GoProManager()  # 1) crear
        self.gopro_manager.frame_ready.connect(self.update_gopro_image)  # 2) conectar señales
        self.gopro_manager.status_update.connect(self.update_status)
        self.gopro_manager.position_update.connect(self.handle_position_update)
        self.gopro_manager.recording_changed.connect(self.update_recording_label)

        
        # Current position tracking
        self.local_ball_count = 0  # local counter increments on each completed shot
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
        self._pos_timer.start(200)

        self.duet_gpIn_index = 1  # J1 -> sensors.gpIn[1]
        self.duet_gpIn2_index = 2  # J2 -> sensors.gpIn[2] (io1.in)
        self.ball_timer = QTimer(self)
        self.ball_timer.timeout.connect(self.poll_duet_ballcount)
        self.ball_timer.start(1200)
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

        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self.show_calibrate_dialog)

        self.manual_control_btn = QPushButton("Activate manual control")
        self.edit_prefs_btn = QPushButton("Edit preferences")
        self.edit_prefs_btn.clicked.connect(self.show_edit_preferences)
        self.manual_control_btn.clicked.connect(self.show_cnc_panel)

        self.exit_btn = QPushButton("Exit Program")
        self.exit_btn.setStyleSheet("background-color: #FF4444; color: white;")
        self.exit_btn.clicked.connect(self.close_program)

        control_layout.addWidget(self.record_btn, 0, 0)
        control_layout.addWidget(self.calibrate_btn, 0, 1)
        control_layout.addWidget(self.manual_control_btn, 1, 0)
        control_layout.addWidget(self.edit_prefs_btn, 1, 1)
        control_layout.addWidget(self.exit_btn, 3, 1)
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

        self.status_label = QLabel("GoPro: idle (not connected)")
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
        
        self.connect_gopro_btn = QPushButton("Connect GoPro")
        self.connect_gopro_btn.clicked.connect(self.connect_gopro)
        status_layout.addWidget(self.connect_gopro_btn)
        status_layout.addWidget(self.server_status_label)
        status_group.setLayout(status_layout)
        right_panel.addWidget(status_group)
        # --- Ball counter (Duet) ---
        ball_group = QGroupBox("Ball Counter")
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
        ball_layout.addSpacing(12)
        ball_layout.addWidget(QLabel("Detected:"))
        self.presence_value = QLabel("—")
        self.presence_value.setAlignment(Qt.AlignCenter)
        self.presence_value.setFixedWidth(90)
        ball_layout.addWidget(self.presence_value, 0)
        ball_group.setLayout(ball_layout)
        right_panel.addWidget(ball_group)

        # --- Second sensor (io1.in) ---
        sensor2_group = QGroupBox("Second Sensor (io1.in)")
        s2_layout = QHBoxLayout()
        s2_layout.addWidget(QLabel("Detected 2:"))
        self.presence2_value = QLabel("—")
        self.presence2_value.setAlignment(Qt.AlignCenter)
        self.presence2_value.setFixedWidth(90)
        s2_layout.addWidget(self.presence2_value, 0)
        sensor2_group.setLayout(s2_layout)
        right_panel.addWidget(sensor2_group)

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

    
    def connect_gopro(self):
        """Start GoPro connection routine with up to 3 attempts. Creates a fresh manager if needed."""
        try:
            # If a previous thread exists and is running, do nothing
            if hasattr(self, "gopro_manager") and self.gopro_manager is not None and self.gopro_manager.isRunning():
                QMessageBox.information(self, "GoPro", "GoPro connection is already running.")
                return
        except Exception:
            pass
        # (Re)create manager instance and wire signals
        try:
            self.gopro_manager = GoProManager()
            self.gopro_manager.frame_ready.connect(self.update_gopro_image)
            self.gopro_manager.status_update.connect(self.update_status)
            self.gopro_manager.position_update.connect(self.handle_position_update)
            self.gopro_manager.recording_changed.connect(self.update_recording_label)
            self.gopro_manager.retry_limit = 3
            self.update_status("🔌 Starting GoPro connection (up to 3 tries)...")
            self.gopro_manager.start()
        except Exception as e:
            QMessageBox.critical(self, "GoPro", f"Failed to start GoPro thread:\n{e}")
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

    def read_gpin(self, idx: int) -> bool:
        """Return True if sensors.gpIn[idx].value is truthy."""
        try:
            r = self.http.get(
                f"http://{self.duet_ip}/rr_model",
                params={"key": f"sensors.gpIn[{idx}].value", "flags": "v"},
                timeout=0.6
            )
            if r.status_code == 200:
                data = r.json()
                val = data.get("result")
                if isinstance(val, dict) and "value" in val:
                    val = val["value"]
                try:
                    return bool(int(val))
                except Exception:
                    return bool(val)
        except Exception:
            return False
        return False

    def _duet_send(self, gcode: str, read_reply: bool = True, timeout: float = 3.0):
        """
        Envía un G-code a la Duet. Considera éxito si:
          - HTTP 200, y
          - 'err' no existe (caso común con RRF) o existe y es 0.
        Devuelve (ok, reply_text, data_json)
        """
        try:
            r = self.http.get(
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
                    rr = self.http.get(f"http://{self.duet_ip}/rr_reply", timeout=timeout)
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

    def _on_shot_completed(self):
        """Increase local ball count and refresh counter field."""
        try:
            self.local_ball_count += 1
            self.ballcount_edit.setText(str(self.local_ball_count))
        except Exception:
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
        cmds = ["G90", f"G1 X{float(x):.3f} Y{float(y):.3f} F3000"]

        # Si el panel manual está abierto, usa su metodo (ya maneja la Duet y prints)
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
            cmd_move = f"G1 X{float(x):.3f} Y{float(y):.3f} F3000"
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
        self.gcode_input.setPlaceholderText("Write a G code (ej: G1 X10 Y20 F6000)")
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
            QMessageBox.warning(self, "Draw circle", "No point with index 0.")
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
            x = cx + R * math.sin(ang)
            y = cy + R * math.cos(ang)
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

    def show_edit_preferences(self):
            dlg = EditPreferencesDialog(self.prefs, self)
            dlg.exec_()
            self.jog_steps_for_panel = self.prefs.get("manual.jog_steps", self.jog_steps_for_panel)

    def get_current_position_from_duet(self):
            try:
                self.http.get(f"http://{self.duet_ip}/rr_gcode", params={"gcode": "M114"}, timeout=2)
                r = self.http.get(f"http://{self.duet_ip}/rr_reply", timeout=2)
                raw = r.text
                import re
                m = re.search(r"X:([\d\.-]+)\s+Y:([\d\.-]+)", raw)
                if m:
                    return m.group(1)[:6], m.group(2)[:6]
            except Exception as e:
                print("get_current_position_from_duet error:", e)
            return None, None

    def _create_circle_from_center_and_radius(self, R: float):
            try:
                cx, cy = 500,500
                new_points = [p for p in self.points if not (isinstance(p.get('idx'), str) and p['idx'].startswith('C') and p['idx'][1:].isdigit())]
                self.points = new_points
                import math
                for k in range(10):
                    ang = math.radians(36 * k)
                    x = cx + R * math.sin(ang)
                    y = cy + R * math.cos(ang)
                    self.points.append({'x': x, 'y': y, 'idx': f"C{k}"})
                self.points_canvas.set_points(self.points)
                self.write_points_to_csv()
                self.update_status(f"Circle created (R={R:.3f}) around new idx 0: C0..C9")
            except Exception as e:
                QMessageBox.warning(self, "Draw circle", f"Circle creation failed: {e}")

    def show_calibrate_dialog(self):
            dlg = CalibrateDialog(self)
            def start_clicked():
                xraw, yraw, rraw = dlg.values()
                def parse_float(s):
                    s = s.replace(",", ".")
                    return float(s)
                def valid_number(s):
                    try:
                        parse_float(s)
                        return True
                    except:
                        return False
                if not xraw or not valid_number(xraw) or abs(parse_float(xraw)) > 500:
                    dlg.msg.setText("Enter a valid X offset (|value| ≤ 500). Decimals allowed.")
                    return
                if not yraw or not valid_number(yraw) or abs(parse_float(yraw)) > 500:
                    dlg.msg.setText("Enter a valid Y offset (|value| ≤ 500). Decimals allowed.")
                    return
                if not rraw or not valid_number(rraw) or abs(parse_float(rraw)) > 5000:
                    dlg.msg.setText("Enter a valid Radius (|value| ≤ 5000).")
                    return
                dlg.accept()
                x_off = float(xraw.replace(",", "."))
                y_off = float(yraw.replace(",", "."))
                radius = float(rraw.replace(",", "."))
                self._run_calibration(x_off, y_off, radius)
            dlg.start_btn.clicked.connect(start_clicked)
            def cancel_and_home():
                dlg.reject()
                self._duet_send('M559 P"homeall.g"', read_reply=True)
            dlg.cancel_btn.clicked.connect(cancel_and_home)
            dlg.exec_()

    def _run_calibration(self, x_off, y_off, radius):
            self.calib_worker = CalibrationWorker(self, x_off, y_off, radius)
            self.calib_worker.progress.connect(self.update_status)
            self.calib_worker.finished.connect(lambda ok, m, res: self.update_status(m))
            self.calib_worker.start()

    def show_cnc_panel(self):
            self.cnc_panel = CNCPanel(duet_ip=self.duet_ip, parent=self, jog_steps=self.jog_steps_for_panel)
            self.cnc_panel.show()

    def _drain_pos_queue(self):
        # Vacía la cola y aplica la última actualización disponible
        try:
            # Process all in this iteration
            got_any = False
            while True:
                positions = self.position_server.queue.get_nowait()
                got_any = True
                # Reuses actual method
                self.update_positions_table(positions)
        except Exception as e:
            # queue.Empty u otros: simplemente ignora
            pass

    def _run_commands_with_monitor(self, cmds: list, title_suffix: str = ""):
        # create the monitor window
        dlg = TestRunnerDialog(total_commands=len(cmds), parent=self)
        if title_suffix:
            dlg.setWindowTitle(f"Start test — {title_suffix} Execution Monitor")

        # create the runner thread
        runner = CommandRunner(main=self, commands=cmds, wait_after_ok_sec=1.0)
        runner.dialog = dlg  # store reference (so pseudo-handlers can use it)

        # wire signals
        runner.sig_executing.connect(lambda i, c: dlg.set_current(c, "Executing"))
        runner.sig_executed.connect(lambda i, c: dlg.set_current(c, "Executed"))
        runner.sig_error.connect(lambda i, c, e: (dlg.set_current(c, "Error")))
        runner.sig_nextslice.connect(dlg.set_next)
        try:
            runner.sig_shot.connect(self._on_shot_completed)
        except Exception:
            pass

        # control buttons
        dlg.pause_clicked.connect(lambda: runner.pause())
        dlg.resume_clicked.connect(lambda: runner.resume())
        dlg.cancel_clicked.connect(lambda: (runner.cancel(), dlg.close()))

        # start thread and show window
        runner.start()
        dlg.exec_()

    def start_structural_integrity_b(self):
        try:
            # Gather geometry
            center = None
            circ = {}
            for p in self.points:
                idx = p.get('idx')
                if idx == 0:
                    center = (float(p['x']), float(p['y']))
                elif isinstance(idx, str) and idx.startswith('C') and idx[1:].isdigit():
                    circ[int(idx[1:])] = (float(p['x']), float(p['y']))
            if center is None:
                QMessageBox.warning(self, "Structural integrity", "No point with index 0 (center).")
                return
            missing = [k for k in range(10) if k not in circ]
            if missing:
                QMessageBox.warning(self, "Structural integrity", f"Missing C points: {missing}.")
                return

            servoP = self.prefs.get("servo.channel", 0)
            S_collect = self.prefs.get("servo_positions.collect", self.prefs.get("servo.s_low", 0))
            per_point = int(self.prefs.get("test.struct_per_point", 10))

            cmds = []
            # Safe startup
            cmds += [
                f"M280 P{servoP} S{S_collect}",
                "M42 P3 S0", "M42 P4 S0", "M42 P5 S0", "M42 P10 S0",
                "G92 E0", "G90"
            ]

            cx, cy = center
            for k in range(10):
                xk, yk = circ[k]
                # 1) 0 -> 2) Ck
                cmds += ["G90", f"G1 X{cx:.3f} Y{cy:.3f} F3000", "M400",
                         f"G1 X{xk:.3f} Y{yk:.3f} F3000", "M400"]
                # 3) start recording, 4) wait 1s
                cmds += ["::CAM_START", "::SLEEP 1"]
                # 5–7) structural cycle until temp_count == per_point
                cmds += [f"::STRUCT_CYCLE_B N={per_point}"]
                # 8) temp resets internally; 9) stop recording
                cmds += ["::CAM_STOP"]

            # 4) Back to point 0, completed
            cmds += ["G90", f"G1 X{cx:.3f} Y{cy:.3f} F3000", "M400"]
            self._run_commands_with_monitor(cmds, "Structural integrity")
        except Exception as e:
            QMessageBox.warning(self, "Structural integrity", str(e))

    def start_structural_integrity_n(self):
        try:
            # Gather geometry
            center = None
            circ = {}
            for p in self.points:
                idx = p.get('idx')
                if idx == 0:
                    center = (float(p['x']), float(p['y']))
                elif isinstance(idx, str) and idx.startswith('C') and idx[1:].isdigit():
                    circ[int(idx[1:])] = (float(p['x']), float(p['y']))
            if center is None:
                QMessageBox.warning(self, "Structural integrity", "No point with index 0 (center).")
                return
            missing = [k for k in range(10) if k not in circ]
            if missing:
                QMessageBox.warning(self, "Structural integrity", f"Missing C points: {missing}.")
                return

            servoP = self.prefs.get("servo.channel", 0)
            S_collect = self.prefs.get("servo_positions.collect", self.prefs.get("servo.s_low", 0))
            per_point = int(self.prefs.get("test.struct_per_point", 10))

            cmds = []
            # Safe startup
            cmds += [
                f"M280 P{servoP} S{S_collect}",
                "M42 P3 S0", "M42 P4 S0", "M42 P5 S0", "M42 P10 S0",
                "G92 E0", "G90"
            ]

            cx, cy = center
            for k in range(10):
                xk, yk = circ[k]
                # 1) 0 -> 2) Ck
                cmds += ["G90", f"G1 X{cx:.3f} Y{cy:.3f} F3000", "M400",
                         f"G1 X{xk:.3f} Y{yk:.3f} F3000", "M400"]
                # 3) start recording, 4) wait 1s
                cmds += ["::CAM_START", "::SLEEP 1"]
                # 5–7) structural cycle until temp_count == per_point
                cmds += ["::BTN_SHOW_NAILS", f"::STRUCT_CYCLE_N N={2}"]
                # 8) temp resets internally; 9) stop recording
                cmds += ["::CAM_STOP"]

            # 4) Back to point 0, completed
            cmds += ["G90", f"G1 X{cx:.3f} Y{cy:.3f} F3000", "M400"]
            self._run_commands_with_monitor(cmds, "Structural integrity")
        except Exception as e:
            QMessageBox.warning(self, "Structural integrity", str(e))

    def start_throwing_objects(self):
        try:
            center = None
            point1 = None
            for p in self.points:
                idx = p.get('idx')
                if idx == 0:
                    center = (float(p['x']), float(p['y']))
                    print(center)
                elif idx == 1:
                    point1 = (float(p['x']), float(p['y']))
                    print(point1)
            if center is None:
                QMessageBox.warning(self, "Throwing objects", "No point with index 0 (center).")
                return
            if point1 is None:
                QMessageBox.warning(self, "Throwing objects", "No point 1 stated.")
                return

            servoP = self.prefs.get("servo.channel", 0)
            S_collect = self.prefs.get("servo_positions.collect", self.prefs.get("servo.s_low", 0))
            per_point = int(self.prefs.get("test.throw_per_point", 10))
            big_iters = int(self.prefs.get("test.throw_big_iters", 5))

            cmds = []
            # Safe startup
            cmds += [
                f"M280 P{servoP} S{S_collect}",
                "M42 P3 S0", "M42 P4 S0", "M42 P5 S0", "M42 P10 S0",
                "G92 E0", "G90"
            ]
            cx, cy = center
            ax, ay = point1
            print(cx, cy, ax, ay)

            for b in range(big_iters):
                # 1) Go to 0 -> 2) Go to point 1
                cmds += [f"G1 X{cx:.3f} Y{cy:.3f} F3000", "M400",
                         f"G1 X{ax:.3f} Y{ay:.3f} F3000", "M400"]
                # 3) do structural cycle but target 100
                cmds += ["::CAM_START", "::SLEEP 1", "::STRUCT_CYCLE_B N=100"]
                # 4) interim prompt
                cmds += ["::INTERIM_PROMPT_SHOW", "::INTERIM_PROMPT_WAIT"]
                # Stop recording after each big iteration
                cmds += ["::CAM_STOP"]

            self._run_commands_with_monitor(cmds, "Throwing objects")
        except Exception as e:
            QMessageBox.warning(self, "Throwing objects", str(e))

    def start_impact_test(self):
        try:
            center = None
            for p in self.points:
                if p.get('idx') == 0:
                    center = (float(p['x']), float(p['y']))
                    break
            if center is None:
                QMessageBox.warning(self, "Impact test", "No point with index 0 (center).")
                return

            servoP = self.prefs.get("servo.channel", 0)
            S_collect = self.prefs.get("servo_positions.collect", self.prefs.get("servo.s_low", 0))

            cmds = []
            # Safe startup
            cmds += [
                f"M280 P{servoP} S{S_collect}",
                "M42 P3 S0", "M42 P4 S0", "M42 P5 S0", "M42 P10 S0",
                "G92 E0", "G90"
            ]
            cx, cy = center

            # Show rod controls on buffer
            cmds += ["::BTN_SHOW_ROD"]

            # 3) to point 0
            cmds += ["G90", f"G1 X{cx:.3f} Y{cy:.3f} F3000", "M400"]
            # 4) to (0,0), wait, 5) wait 2s
            cmds += ["G1 X0.000 Y0.000 F3000", "M400", "::SLEEP 2"]
            # 6) enable insert, 7) wait for operator
            cmds += ["::BTN_ENABLE_INSERT", "::WAIT_BTN_INSERT"]
            # 8–9) actuate rod
            cmds += ["M42 P4 S1", "M42 P5 S0"]
            # 10) enable retract, 11) wait for operator
            cmds += ["::BTN_ENABLE_RETRACT", "::WAIT_BTN_RETRACT"]
            # 12–13) retract rod
            cmds += ["M42 P4 S0", "M42 P5 S1"]

            self._run_commands_with_monitor(cmds, "Impact test")
        except Exception as e:
            QMessageBox.warning(self, "Impact test", str(e))

    def start_test(self):
        # Always resolve dynamically and guard exceptions
        cls = globals().get("TestTypeDialog")
        if cls is None:
            QMessageBox.critical(self, "Start test", "Internal error: TestTypeDialog class not found.")
            return
        try:
            sel = cls(self)
        except Exception as e:
            QMessageBox.critical(self, "Start test", f"Failed to open test selector:\n{e}")
            return

        if sel.exec_() != QDialog.Accepted or not sel.choice:
            return

        mapping = {
            "structural_integrity_b": self.start_structural_integrity_b,
            "structural_integrity_n": self.start_structural_integrity_n,
            "throwing_objects": self.start_throwing_objects,
            "impact_test": self.start_impact_test,
        }
        fn = mapping.get(sel.choice)
        if not fn:
            QMessageBox.warning(self, "Start test", f"Invalid selection: {sel.choice}")
            return
        fn()

    def poll_duet_ballcount(self):
            """Update sensors from Duet; ball count is local and shown from self.local_ball_count."""
            try:
                self.ballcount_edit.setText(str(self.local_ball_count))
            except Exception:
                pass
            self.update_presence_from_duet()
            self.update_presence2_from_duet()

    def update_presence_from_duet(self):
        """
        Read sensors.gpIn[J1].value to know metal presence (sensor 1).
        """
        try:
            idx = getattr(self, "duet_gpIn_index", 1)
            r = self.http.get(f"http://{self.duet_ip}/rr_model",
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

    def update_presence2_from_duet(self):
        """
        Read sensors.gpIn[J2].value (defaults to 2) to know second sensor state (io1.in).
        """
        try:
            idx = getattr(self, "duet_gpIn2_index", 2)
            r = self.http.get(f"http://{self.duet_ip}/rr_model",
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
                self.presence2_value.setText("Metal" if present else "No metal")
                self.presence2_value.setStyleSheet("color: green;" if present else "color: gray;")
        except Exception:
            pass

    def reset_ballcount(self):
        """Reset the local shot counter only."""
        self.local_ball_count = 0
        try:
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
