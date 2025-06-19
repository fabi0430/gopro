import asyncio
import cv2
import time
import os
import aiohttp
import numpy as np
from open_gopro import WirelessGoPro
from open_gopro.models import constants, streaming
import platform

# Set Windows event loop policy if on Windows
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

STREAM_PORT = 8554
RECONNECT_DELAY = 3
DOWNLOAD_DIR = "GoProDownloads"

# Initialize red X detection variables at module level
reference_position = None
error_range = None
position_window = None
threshold_window_created = False

# HSV thresholds (default values)
h_low1, h_high1 = 0, 10
h_low2, h_high2 = 170, 180
s_low, s_high = 120, 255
v_low, v_high = 70, 255

def create_threshold_trackbars():
    global threshold_window_created
    if not threshold_window_created:
        cv2.namedWindow("Threshold Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Threshold Controls", 400, 400)
        cv2.createTrackbar("H1 Low", "Threshold Controls", h_low1, 180, lambda x: None)
        cv2.createTrackbar("H1 High", "Threshold Controls", h_high1, 180, lambda x: None)
        cv2.createTrackbar("H2 Low", "Threshold Controls", h_low2, 180, lambda x: None)
        cv2.createTrackbar("H2 High", "Threshold Controls", h_high2, 180, lambda x: None)
        cv2.createTrackbar("S Low", "Threshold Controls", s_low, 255, lambda x: None)
        cv2.createTrackbar("S High", "Threshold Controls", s_high, 255, lambda x: None)
        cv2.createTrackbar("V Low", "Threshold Controls", v_low, 255, lambda x: None)
        cv2.createTrackbar("V High", "Threshold Controls", v_high, 255, lambda x: None)
        threshold_window_created = True

def detect_red_x(frame):
    global reference_position, error_range, position_window

    # Check if threshold window exists
    if not threshold_window_created:
        return frame, False, None

    try:
        # Get trackbar values
        h1_low = cv2.getTrackbarPos("H1 Low", "Threshold Controls")
        h1_high = cv2.getTrackbarPos("H1 High", "Threshold Controls")
        h2_low = cv2.getTrackbarPos("H2 Low", "Threshold Controls")
        h2_high = cv2.getTrackbarPos("H2 High", "Threshold Controls")
        s_lo = cv2.getTrackbarPos("S Low", "Threshold Controls")
        s_hi = cv2.getTrackbarPos("S High", "Threshold Controls")
        v_lo = cv2.getTrackbarPos("V Low", "Threshold Controls")
        v_hi = cv2.getTrackbarPos("V High", "Threshold Controls")

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range for red color
        lower_red1 = np.array([h1_low, s_lo, v_lo])
        upper_red1 = np.array([h1_high, s_hi, v_hi])
        lower_red2 = np.array([h2_low, s_lo, v_lo])
        upper_red2 = np.array([h2_high, s_hi, v_hi])

        # Threshold the HSV image
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        x_detected = False
        current_position = None
        processed_frame = frame.copy()

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                current_position = (x + w // 2, y + h // 2)
                x_detected = True

                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(processed_frame, current_position, 5, (0, 255, 0), -1)
                cv2.putText(processed_frame, 'X', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if reference_position is not None:
                    ref_x, ref_y = reference_position
                    err_x, err_y = error_range

                    cv2.circle(processed_frame, reference_position, 5, (255, 0, 0), -1)
                    cv2.rectangle(processed_frame,
                                (ref_x - err_x, ref_y - err_y),
                                (ref_x + err_x, ref_y + err_y),
                                (255, 255, 0), 2)

                    if position_window is None:
                        position_window = np.zeros((100, 400, 3), dtype=np.uint8)

                    position_window[:] = (0, 0, 0)  # Clear the window

                    if current_position is not None and \
                            abs(current_position[0] - ref_x) <= err_x and \
                            abs(current_position[1] - ref_y) <= err_y:
                        cv2.putText(position_window, "X is in position", (50, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(position_window, "X is NOT in position", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.imshow('Position Status', position_window)

        return processed_frame, x_detected, current_position
    except:
        return frame, False, None

async def download_latest_video(gopro):
    try:
        media = await gopro.http_command.list_media()
        if not media.media:
            print("⚠️ No media found on camera")
            return

        latest_folder = media.media[-1]
        latest_file = latest_folder.files[-1]
        filename = latest_file.name
        full_url = f"http://10.5.5.9:8080/videos/DCIM/{latest_folder.d}/{filename}"

        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        save_path = os.path.join(DOWNLOAD_DIR, filename)

        print(f"⬇️ Downloading {filename} to {save_path}")
        async with aiohttp.ClientSession() as session:
            async with session.get(full_url) as resp:
                if resp.status == 200:
                    with open(save_path, "wb") as f:
                        f.write(await resp.read())
                    print(f"✅ Downloaded: {save_path}")
                else:
                    print(f"❌ Failed to download: HTTP {resp.status}")
    except Exception as e:
        print(f"⚠️ Error downloading video: {e}")

async def start_preview(gopro):
    await gopro.streaming.start_stream(
        streaming.StreamType.PREVIEW,
        streaming.PreviewStreamOptions(port=STREAM_PORT)
    )
    return gopro.streaming.url

async def main():
    global reference_position, error_range
    
    # Initialize reference position with default values
    reference_position = None
    error_range = None
    
    try:
        # First establish BLE connection before any GUI windows
        gopro = WirelessGoPro()
        await gopro.open()
        print("✅ Connected to GoPro")

        # Now create GUI windows
        create_threshold_trackbars()

        recording = False
        should_exit = False

        while not should_exit:
            try:
                url = await start_preview(gopro)
                print(f"📡 Livestream at {url}")
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    raise RuntimeError("Could not open VideoCapture!")

                print("🎥 Controls:")
                print("- 'r' toggle recording")
                print("- 'x' set reference position")
                print("- 'q' quit")

                last_error = None
                while not should_exit:
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError("Frame read failed")

                    # Process frame for red X detection
                    processed_frame, x_detected, current_position = detect_red_x(frame)
                    cv2.imshow("GoPro Livestream - X Detection", processed_frame)

                    # Create status window
                    status_frame = np.zeros((100, 400, 3), dtype=np.uint8)
                    status_text = "RED X DETECTED" if x_detected else "NO RED X DETECTED"
                    color = (0, 255, 0) if x_detected else (0, 0, 255)
                    cv2.putText(status_frame, status_text, (50, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    if reference_position is not None:
                        ref_status = f"Reference: ({reference_position[0]}, {reference_position[1]})"
                        cv2.putText(status_frame, ref_status, (50, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.imshow('Detection Status', status_frame)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        print("🛑 Quit requested")
                        if recording:
                            print("⏹️ Stopping recording before exit...")
                            await gopro.http_command.set_shutter(shutter=constants.Toggle.DISABLE)
                            recording = False
                            await asyncio.sleep(2)
                            await download_latest_video(gopro)
                        should_exit = True
                        break

                    elif key == ord('x') and current_position is not None:
                        reference_position = current_position
                        height, width = frame.shape[:2]
                        error_range = (int(width * 0.01), int(height * 0.01))
                        print(f"Reference position set to {reference_position} with error range ±{error_range}")

                    elif key == ord('r'):
                        toggle = constants.Toggle.ENABLE if not recording else constants.Toggle.DISABLE
                        await gopro.http_command.set_shutter(shutter=toggle)
                        recording = not recording
                        print("🔴 Recording started" if recording else "⏹️ Recording stopped")

                        if not recording:
                            await asyncio.sleep(2)
                            await download_latest_video(gopro)

            except Exception as ex:
                print(f"⚠️ {ex}")
                last_error = time.time()
            finally:
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
                cv2.destroyWindow("GoPro Livestream - X Detection")
                cv2.destroyWindow("Detection Status")
                if 'position_window' in globals() and position_window is not None:
                    cv2.destroyWindow("Position Status")
                await gopro.streaming.stop_active_stream()
                if not should_exit and last_error:
                    print(f"🔄 Reconnecting in {RECONNECT_DELAY} s…")
                    await asyncio.sleep(RECONNECT_DELAY)

    except Exception as e:
        print(f"⚠️ Failed to connect to GoPro: {e}")
    finally:
        # Clean up all windows when exiting
        cv2.destroyAllWindows()
        if 'gopro' in locals():
            await gopro.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Program stopped by user")
        cv2.destroyAllWindows()