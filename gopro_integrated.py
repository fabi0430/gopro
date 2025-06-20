import sys
import cv2
import numpy as np
import asyncio
import aiohttp
import os
import socket
from threading import Thread
import threading
from open_gopro import WirelessGoPro
from open_gopro.models import constants, streaming
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem, QGroupBox,
                             QSlider, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import platform

# Configuration
STREAM_PORT = 8554
DOWNLOAD_DIR = "GoProDownloads"
POSITION_SERVER_PORT = 65432
RECONNECT_DELAY = 3

toggle_recording_int=0

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class PositionServer(QThread):
    position_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.host = '0.0.0.0'
        self.port = POSITION_SERVER_PORT
        self.positions = {f"pos{i}": ["0.0", "0.0"] for i in range(1, 11)}
        self.server_socket = None
        self.running = False

    def run(self):
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Position server listening on {self.host}:{self.port}")

        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                Thread(target=self.handle_client, args=(conn,), daemon=True).start()
            except OSError:
                break

    def handle_client(self, conn):
        with conn:
            while self.running:
                try:
                    data = conn.recv(1024).decode('utf-8')
                    if not data:
                        break

                    # Process received data
                    for line in data.split('\n'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            if key.startswith('pos') and key[3:].isdigit():
                                pos_num = int(key[3:])
                                if 1 <= pos_num <= 10:
                                    x, y = value.split(',')
                                    self.positions[key] = [x, y]
                                    self.position_received.emit(self.positions)
                except (ConnectionResetError, ValueError) as e:
                    print(f"Connection error: {e}")
                    break

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()


class MediaHttpGetter:
    pass


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
        self.filenameint = 0

    async def initialize(self):
        try:
            self.gopro = WirelessGoPro()
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

            state_resp = await self.gopro.http_command.get_camera_state()
            # state_resp.data es un dict con keys de constantes StatusId
            encoding = state_resp.data.get(constants.StatusId.ENCODING, 0)
            recording_now = bool(encoding)
            print("Estado actual real (ENCODING):", encoding)

            toggle = constants.Toggle.DISABLE if recording_now else constants.Toggle.ENABLE
            print(f"Toggle value: {toggle}")

            resp = await self.gopro.http_command.set_shutter(shutter=toggle)
            print("HTTP set_shutter response:", resp)

            if not resp.ok:
                raise RuntimeError(f"GoPro error: {resp.status}")

            self.recording = not recording_now
            status = "🔴 Recording started" if encoding else "⏹️ Recording stopped"
            self.status_update.emit(status)
            print(">>> Recording state toggled successfully")

            if not self.recording:
                print(">>> Downloading video after stop...")
                await asyncio.sleep(2)
                await self.download_and_log()

        except Exception as e:
            error_msg = f"Recording: error → {e}"
            print("❌", error_msg)
            self.status_update.emit(error_msg)

    async def download_and_log(self):
        print("Rutina de guardado de archivo iniciada")
        try:
            media_resp = await self.gopro.http_command.get_media_list()
            files = media_resp.data.files
            print("DEBUG media count:", len(files))
            if not files:
                raise RuntimeError("No media files found")

            last = max(files, key=lambda x: x.created)
            fname = last.filename
            save_path = os.path.join(DOWNLOAD_DIR, fname)
            async with aiohttp.ClientSession() as sess:
                resp = await sess.get(last.download_url)
                print("DEBUG download URL:", last.download_url)
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                with open(save_path, "wb") as f:
                    f.write(await resp.read())
            self.status_update.emit(f"✅ Video downloaded: {fname}")
        except Exception as e:
            self.status_update.emit(f"⚠️ Download error: {e}")

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

                        # Convert to QImage for GUI
                        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        self.frame_ready.emit(qt_image)

                        # Show filter windows if enabled
                        if self.show_filters:
                            red_mask, cleaned_mask = masks
                            cv2.imshow("Red Mask", red_mask)
                            cv2.imshow("Cleaned Mask", cleaned_mask)
                            cv2.waitKey(1)
                    else:
                        break

                # Cleanup
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

        # Final cleanup
        if self.gopro:
            await self.gopro.close()

    def process_frame(self, frame):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define red color ranges
        lower_red1 = np.array([self.hsv_values[0], self.hsv_values[4], self.hsv_values[6]])
        upper_red1 = np.array([self.hsv_values[1], self.hsv_values[5], self.hsv_values[7]])
        lower_red2 = np.array([self.hsv_values[2], self.hsv_values[4], self.hsv_values[6]])
        upper_red2 = np.array([self.hsv_values[3], self.hsv_values[5], self.hsv_values[7]])

        # Threshold the HSV image
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        processed_frame = frame.copy()
        current_pos = None

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                current_pos = (x + w // 2, y + h // 2)

                # Draw rectangle and center
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(processed_frame, current_pos, 5, (0, 255, 0), -1)
                cv2.putText(processed_frame, 'X', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Show coordinates
                cv2.putText(processed_frame, f"({current_pos[0]}, {current_pos[1]})",
                            (current_pos[0] + 10, current_pos[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Emit current position
                self.position_update.emit(current_pos)

        # Draw reference position if exists
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

        # Setup position server
        self.position_server = PositionServer()
        self.position_server.position_received.connect(self.update_positions_table)
        self.position_server.start()

        # Setup GoPro
        self.gopro_manager = GoProManager()
        self.gopro_manager.frame_ready.connect(self.update_gopro_image)
        self.gopro_manager.status_update.connect(self.update_status)
        self.gopro_manager.position_update.connect(self.handle_position_update)
        self.gopro_manager.recording_changed.connect(self.update_recording_label)

        # Current position tracking
        self.current_position = None

        # Initialize UI
        self.init_ui()

        # Start GoPro
        self.gopro_manager.start()

        # Crear el bucle dedicado para GoPro
        self.gopro_loop = asyncio.new_event_loop()

        def start_gopro_loop():
            asyncio.set_event_loop(self.gopro_loop)
            self.gopro_loop.run_forever()

        threading.Thread(target=start_gopro_loop, daemon=True).start()

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

        self.record_btn = QPushButton("Toggle Recording")
        self.record_btn.clicked.connect(self.handle_record_click)

        self.set_ref_btn = QPushButton("Set Reference Position")
        self.set_ref_btn.clicked.connect(self.set_reference_position)

        self.show_filters_btn = QPushButton("Show/Hide Filters")
        self.show_filters_btn.clicked.connect(self.toggle_filters)

        self.exit_btn = QPushButton("Exit Program")
        self.exit_btn.setStyleSheet("background-color: #FF4444; color: white;")
        self.exit_btn.clicked.connect(self.close_program)

        control_layout.addWidget(self.record_btn, 0, 0)
        control_layout.addWidget(self.set_ref_btn, 0, 1)
        control_layout.addWidget(self.show_filters_btn, 1, 0)
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

        # Right panel (status, positions and table)
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

        # Current position group
        current_pos_group = QGroupBox("Current Position")
        current_pos_layout = QVBoxLayout()

        self.current_pos_table = QTableWidget()
        self.current_pos_table.setColumnCount(2)
        self.current_pos_table.setHorizontalHeaderLabels(["Type", "Position (X,Y)"])
        self.current_pos_table.setRowCount(2)

        self.current_pos_table.setItem(0, 0, QTableWidgetItem("Current"))
        self.current_pos_table.setItem(0, 1, QTableWidgetItem("N/A"))
        self.current_pos_table.setItem(1, 0, QTableWidgetItem("Reference"))
        self.current_pos_table.setItem(1, 1, QTableWidgetItem("N/A"))

        current_pos_layout.addWidget(self.current_pos_table)
        current_pos_group.setLayout(current_pos_layout)
        right_panel.addWidget(current_pos_group)

        # Tablet positions group
        tablet_pos_group = QGroupBox("Received Positions (Tablet)")
        tablet_pos_layout = QVBoxLayout()

        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(3)
        self.positions_table.setHorizontalHeaderLabels(["Position", "X", "Y"])
        self.positions_table.setRowCount(10)

        for i in range(10):
            self.positions_table.setItem(i, 0, QTableWidgetItem(f"Position {i + 1}"))
            self.positions_table.setItem(i, 1, QTableWidgetItem("0.0"))
            self.positions_table.setItem(i, 2, QTableWidgetItem("0.0"))

        tablet_pos_layout.addWidget(self.positions_table)
        tablet_pos_group.setLayout(tablet_pos_layout)
        right_panel.addWidget(tablet_pos_group)

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
        self.current_pos_table.item(0, 1).setText(f"({position[0]}, {position[1]})")

    def update_positions_table(self, positions):
        for i in range(1, 11):
            pos_key = f"pos{i}"
            x, y = positions.get(pos_key, ["0.0", "0.0"])
            self.positions_table.item(i - 1, 1).setText(x)
            self.positions_table.item(i - 1, 2).setText(y)

    def set_reference_position(self):
        if self.current_position:
            self.gopro_manager.set_reference_position(self.current_position)
            self.current_pos_table.item(1, 1).setText(f"({self.current_position[0]}, {self.current_position[1]})")
            self.ref_pos_label.setText(f"Reference position: ({self.current_position[0]}, {self.current_position[1]})")
        else:
            QMessageBox.warning(self, "Warning", "No X position detected")

    def handle_record_click(self):
        print("Botón presionado, enviando coroutine")

        future = asyncio.run_coroutine_threadsafe(
            self.gopro_manager.toggle_recording(), self.gopro_loop
        )

        def callback(fut):
            try:
                result = fut.result()
                print("Grabación finalizada:", result)
            except Exception as e:
                print("Error en grabación:", e)

        future.add_done_callback(callback)

    def toggle_filters(self):
        self.gopro_manager.toggle_filters()

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
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())