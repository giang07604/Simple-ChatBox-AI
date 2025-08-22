# -*- coding: utf-8 -*-
"""
Universal HMcaptcha Solver (exclusive mouse)
- Auto-detect panel & type: slide / rotate / object selection
- Call HMcaptcha API
- Auto act with pyautogui

Requirements:
  pip install mss pillow opencv-python numpy requests pyautogui
"""

import os
import io
import time
import math
import json
import base64
import re
import platform
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import cv2
import mss
import pyautogui as pag
from PIL import Image
import requests

# OCR for text detection
try:
    import pytesseract

    # Auto-configure Tesseract for cross-platform
    def configure_tesseract():
        system = platform.system()

        if system == "Windows":
            # Common Windows installation paths
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(
                    os.getenv('USERNAME', '')),
                r'C:\tesseract\tesseract.exe'
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"✅ Tesseract found at: {path}")
                    return True

            print("⚠️ Tesseract not found on Windows. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            return False

        elif system == "Darwin":  # macOS
            # Check if tesseract is in PATH (usually installed via brew)
            try:
                import subprocess
                result = subprocess.run(
                    ['which', 'tesseract'], capture_output=True, text=True)
                if result.returncode == 0:
                    tesseract_path = result.stdout.strip()
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    print(f"✅ Tesseract found at: {tesseract_path}")
                    return True
                else:
                    print(
                        "⚠️ Tesseract not found on macOS. Install with: brew install tesseract")
                    return False
            except Exception:
                print("⚠️ Could not locate tesseract on macOS")
                return False

        else:  # Linux
            # Usually in PATH
            print("✅ Using system tesseract (Linux)")
            return True

    TESSERACT_AVAILABLE = configure_tesseract()

    if TESSERACT_AVAILABLE:
        print("✅ pytesseract đã sẵn sàng")
    else:
        print("⚠️ pytesseract không hoạt động, chỉ dùng API detection")

except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ pytesseract không có, chỉ dùng API detection")

# ================= CONFIG =================
HM_API_KEY = os.getenv(
    "HMCAPTCHA_KEY", "giangdev-KmpjQPCO2kzrCqtoAud1TiSi4OsxVfxN")
HM_BASE = "https://hmcaptcha.com"

# Telegram config for sending screenshots
TELEGRAM_BOT_TOKEN = "7431244567:AAEqvYa58hSXbfJaQ4ZiIv3MR1SHHoVmMmA"
TELEGRAM_CHAT_ID = "1934845201"

# HMcaptcha API endpoints
API_ENDPOINTS = {
    "create_task": "/Recognition?wait=1",
    "get_result": "/getResult"
}

# Captcha type mapping to HMcaptcha types
CAPTCHA_TYPES = {
    "slide": "ALL_CAPTCHA_SLIDE",
    "rotate_app": "TIKTOK_ROTATE_APP",
    "rotate_web": "TIKTOK_ROTATE_WEB",
    "object": "TIKTOK_OBJ"
}

# Fixed captcha region settings (percentage of screen for compatibility)
FIXED_CAPTCHA_REGION = {
    "enabled": True,  # Set to False to use auto-detection
    "x_percent": 0.3,     # 30% from left edge (thu hẹp từ 25%)
    "y_percent": 0.15,    # 15% from top edge (thu hẹp từ 10%)
    "width_percent": 0.4,  # 40% of screen width (thu hẹp từ 50%)
    "height_percent": 0.7  # 70% of screen height (thu hẹp từ 80%)
}

# Statistics tracking
STATS = {
    "attempts": 0,
    "successes": 0,
    "failures": 0,
    "cost_per_thousand": 0.5  # $0.5 per 1000 successful solves
}

# Auto-detection fallback settings
AUTO_DETECTION = {
    "enabled": False,  # Disable auto-detection, only use fixed region
    "max_panels": 1   # Max panels to check in auto-detection
}

# Save screenshots for debugging
SCREENSHOT_DIR = "screenshots"
SCAN_INTERVAL = 10  # seconds (faster scanning)

# Create screenshot directory
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Cross-platform mouse settings
SYSTEM = platform.system()
print(f"🖥️ Hệ điều hành: {SYSTEM}")

# pyautogui settings for cross-platform compatibility
pag.FAILSAFE = True
pag.PAUSE = 0.1

# macOS specific settings
if SYSTEM == "Darwin":  # macOS
    print("🍎 Cấu hình cho macOS...")
    # Disable pyautogui safety features that might interfere
    pag.FAILSAFE = False
    pag.PAUSE = 0.05
elif SYSTEM == "Windows":
    print("🪟 Cấu hình cho Windows...")
    pag.PAUSE = 0.1


@dataclass
class ROI:
    left: int
    top: int
    width: int
    height: int

# ================= UTILITIES =================


def screenshot_full():
    """Take full screenshot using mss"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Main monitor
        screenshot = sct.grab(monitor)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")


def screenshot_fixed_region():
    """Take screenshot of fixed captcha region using percentage"""
    if not FIXED_CAPTCHA_REGION["enabled"]:
        return None

    monitor_info = get_monitor_info()

    # Calculate absolute coordinates from percentages
    x = int(monitor_info["width"] * FIXED_CAPTCHA_REGION["x_percent"])
    y = int(monitor_info["height"] * FIXED_CAPTCHA_REGION["y_percent"])
    width = int(monitor_info["width"] * FIXED_CAPTCHA_REGION["width_percent"])
    height = int(monitor_info["height"] *
                 FIXED_CAPTCHA_REGION["height_percent"])

    region = {
        "left": x,
        "top": y,
        "width": width,
        "height": height
    }

    with mss.mss() as sct:
        try:
            screenshot = sct.grab(region)
            img = Image.frombytes("RGB", screenshot.size,
                                  screenshot.bgra, "raw", "BGRX")
            return img, (x, y, width, height)
        except Exception as e:
            print(f"❌ Lỗi chụp vùng cố định: {e}")
            return None, None


def get_fixed_region_coords():
    """Get absolute coordinates of fixed region"""
    monitor_info = get_monitor_info()
    x = int(monitor_info["width"] * FIXED_CAPTCHA_REGION["x_percent"])
    y = int(monitor_info["height"] * FIXED_CAPTCHA_REGION["y_percent"])
    width = int(monitor_info["width"] * FIXED_CAPTCHA_REGION["width_percent"])
    height = int(monitor_info["height"] *
                 FIXED_CAPTCHA_REGION["height_percent"])
    return (x, y, width, height)


def get_monitor_info():
    """Get monitor resolution info for coordinate calculation with scaling detection"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Main monitor

        # Get physical monitor size
        physical_width = monitor["width"]
        physical_height = monitor["height"]

        # Detect if this is a high-DPI/Retina display
        scaling_factor = detect_display_scaling()

        return {
            "width": physical_width,
            "height": physical_height,
            "left": monitor["left"],
            "top": monitor["top"],
            "scaling_factor": scaling_factor,
            "is_retina": scaling_factor > 1.5
        }


def detect_display_scaling():
    """Detect display scaling factor for different monitor types - Enhanced version"""
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            physical_width = monitor["width"]
            physical_height = monitor["height"]

        if SYSTEM == "Darwin":  # macOS
            # More accurate Retina detection for macOS
            retina_patterns = [
                # Width-based detection for common Retina displays
                (2880, 2.0),  # MacBook Pro 15" Retina
                (2560, 2.0),  # MacBook Pro 13" Retina, iMac 27"
                (2304, 2.0),  # MacBook 12" Retina
                (5120, 2.0),  # iMac 27" 5K
                (4096, 2.0),  # iMac 21.5" 4K
                (3456, 2.0),  # MacBook Pro 14" M1
                (3024, 2.0),  # MacBook Pro 16" M1
                (1920, 1.0),  # Standard 1080p
                (1680, 1.0),  # Standard resolution
            ]

            for width_threshold, scaling in retina_patterns:
                if physical_width >= width_threshold * 0.95:  # 5% tolerance
                    print(
                        f"🖥️ Detected macOS display: {physical_width}x{physical_height} → scaling={scaling}")
                    return scaling

        elif SYSTEM == "Windows":
            # Enhanced Windows high-DPI detection
            try:
                import ctypes
                user32 = ctypes.windll.user32

                # Try to get DPI
                try:
                    user32.SetProcessDPIAware()
                    hdc = user32.GetDC(0)
                    dpi_x = ctypes.windll.gdi32.GetDeviceCaps(
                        hdc, 88)  # LOGPIXELSX
                    user32.ReleaseDC(0, hdc)

                    scaling = dpi_x / 96.0  # Standard DPI is 96
                    print(
                        f"🖥️ Windows DPI detection: {dpi_x} DPI → scaling={scaling:.1f}")

                    if scaling >= 1.8:
                        return 2.0
                    elif scaling >= 1.4:
                        return 1.5
                    elif scaling >= 1.1:
                        return 1.25
                    else:
                        return 1.0

                except Exception:
                    # Fallback to resolution-based detection
                    pass

                # Resolution-based fallback for Windows
                if physical_width >= 3200:  # 4K+
                    return 2.0
                elif physical_width >= 2560:  # QHD
                    return 1.5
                elif physical_width >= 1920:  # FHD
                    return 1.0
                else:
                    return 1.0

            except Exception:
                # Ultimate fallback
                if physical_width >= 2560:
                    return 1.5
                else:
                    return 1.0

        else:  # Linux
            # Linux scaling detection
            if physical_width >= 3200:
                return 2.0
            elif physical_width >= 2560:
                return 1.5
            else:
                return 1.0

    except Exception as e:
        print(f"⚠️ Could not detect scaling: {e}")
        return 1.0


def save_screenshot(img, prefix="debug"):
    """Gửi ảnh qua Telegram thay vì lưu file"""
    try:
        # Convert PIL image to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        # Prepare Telegram API call
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

        # Create caption with timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        caption = f"🤖 {prefix} - {timestamp}"

        # Prepare files and data
        files = {
            'photo': ('screenshot.png', buffer, 'image/png')
        }
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'caption': caption
        }

        # Send to Telegram
        # response = requests.post(telegram_url, files=files, data=data, timeout=10)

        # if response.status_code == 200:
        #     print(f"📸 Đã gửi ảnh qua Telegram: {prefix}")
        # else:
        #     print(f"❌ Lỗi gửi Telegram: {response.status_code}")

    except Exception as e:
        print(f"❌ Lỗi gửi ảnh Telegram: {e}")
        # Fallback: save to file if Telegram fails
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.png"
            filepath = os.path.join(SCREENSHOT_DIR, filename)
            img.save(filepath)
            print(f"💾 Fallback - Saved: {filepath}")
        except Exception as save_error:
            print(f"❌ Lỗi lưu file: {save_error}")


def pil_to_cv(pil_img):
    """Convert PIL to OpenCV"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img):
    """Convert OpenCV to PIL"""
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def crop_pil(pil_img, roi: ROI):
    """Crop PIL image using ROI"""
    return pil_img.crop((roi.left, roi.top, roi.left + roi.width, roi.top + roi.height))


def pil_to_b64(pil_img):
    """Convert PIL image to base64"""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# ================= API FUNCTIONS =================


def post_json(captcha_type, image_b64):
    """Call HMcaptcha API with task creation and polling"""
    try:
        # Map captcha type to HMcaptcha type
        hm_type = CAPTCHA_TYPES.get(captcha_type)
        if not hm_type:
            return {"error": f"Unsupported captcha type: {captcha_type}"}

        # Create task
        create_url = f"{HM_BASE}{API_ENDPOINTS['create_task']}"

        payload = {
            "Type": hm_type,
            "Image": image_b64,
            "Apikey": HM_API_KEY
        }

        response = requests.post(create_url, json=payload, timeout=30)

        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text}"}

        result = response.json()

        if result.get("Status") == "SUCCESS":
            return result
        elif result.get("Status") == "PENDING":
            task_id = result.get("TaskId")
            if task_id:
                return poll_result(task_id)
            else:
                return {"error": "No TaskId returned"}
        else:
            return {"error": f"Task creation failed: {result}"}

    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}


def poll_result(task_id, max_attempts=20):
    """Poll for task result"""
    result_url = f"{HM_BASE}{API_ENDPOINTS['get_result']}"

    for attempt in range(max_attempts):
        try:
            response = requests.get(
                result_url, params={"apikey": HM_API_KEY, "taskId": task_id}, timeout=15)

            if response.status_code == 200:
                result = response.json()

                if result.get("Status") == "SUCCESS":
                    return result
                elif result.get("Status") == "PENDING":
                    time.sleep(2)  # Wait before next poll
                    continue
                else:
                    return {"error": f"Task failed: {result}"}
            else:
                time.sleep(2)
                continue

        except Exception as e:
            time.sleep(2)
            continue

    return {"error": "Polling timeout"}

# ================= PANEL DETECTION =================


def find_panels_cv(full_cv):
    """Find potential captcha panels"""
    gray = cv2.cvtColor(full_cv, cv2.COLOR_BGR2GRAY)

    # Multi-scale template matching approach
    panels = []

    # Method 1: Contour-based detection
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 50000 < area < 500000:  # Reasonable captcha size
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            # Captcha panels are usually rectangular
            if 0.8 < aspect_ratio < 3.0 and w > 300 and h > 200:
                score = area / 10000  # Simple scoring
                panels.append((score, (x, y, w, h)))

    # Method 2: Rectangle detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours2, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours2:
        area = cv2.contourArea(contour)
        if 40000 < area < 600000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            if 0.7 < aspect_ratio < 4.0 and w > 250 and h > 150:
                score = area / 8000
                panels.append((score, (x, y, w, h)))

    # Remove duplicates and sort by score
    unique_panels = []
    for score, (x, y, w, h) in panels:
        is_duplicate = False
        for _, (x2, y2, w2, h2) in unique_panels:
            if abs(x - x2) < 50 and abs(y - y2) < 50:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_panels.append((score, (x, y, w, h)))

    return sorted(unique_panels, key=lambda x: x[0], reverse=True)


def refine_panel_bbox(panel):
    """Refine panel bounding box to focus on captcha content"""
    ph, pw = panel.shape[:2]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    # Find the main content area
    edges = cv2.Canny(gray, 30, 100)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get bounding box of all significant contours
        all_points = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                all_points.extend([(x, y), (x+w, y+h)])

        if all_points:
            xs, ys = zip(*all_points)
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Add some padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(pw, x_max + padding)
            y_max = min(ph, y_max + padding)

            return (x_min, y_min, x_max - x_min, y_max - y_min)

    return None

# ================= CAPTCHA TYPE DETECTION =================


def detect_captcha_type_improved(panel: np.ndarray) -> Tuple[str, float]:
    """
    Cải thiện thuật toán nhận diện loại captcha với debug info
    Returns: (type, confidence_score)
    """
    ph, pw = panel.shape[:2]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    # Kiểm tra kích thước minimum
    if pw < 200 or ph < 100:
        return "unknown", 0.0

    # Tính confidence cho từng loại
    slide_confidence = detect_slide_captcha_improved(panel)
    rotate_confidence = detect_rotate_captcha_improved(panel)
    object_confidence = detect_object_selection(panel)

    print(
        f"🧩 Detection scores: slide={slide_confidence:.2f}, rotate={rotate_confidence:.2f}, object={object_confidence:.2f}")

    # Tìm confidence cao nhất
    scores = [
        ("slide", slide_confidence),
        ("rotate_web", rotate_confidence),
        ("object", object_confidence)
    ]

    best_type, best_confidence = max(scores, key=lambda x: x[1])

    # Nếu là rotate và confidence đủ cao, phân biệt APP vs WEB
    if best_type == "rotate_web" and best_confidence > 0.5:
        if detect_rotate_app_pattern(panel):
            best_type = "rotate_app"

    print(f"🏆 Best match: {best_type} (confidence: {best_confidence:.2f})")

    # Chỉ trả về nếu confidence đủ cao
    if best_confidence >= 0.5:  # Threshold để test
        return best_type, best_confidence
    else:
        print(f"❌ All confidences too low, returning unknown")
        return "unknown", 0.0


def detect_slide_captcha_improved(panel: np.ndarray) -> float:
    """Cải thiện detection cho slide captcha - nhận diện puzzle pieces trên background"""
    ph, pw = panel.shape[:2]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    # BLACKLIST FILTER: Loại bỏ VSCode/IDE UI
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    dominant_color = np.argmax(hist)
    dominant_ratio = hist[dominant_color] / (ph * pw)

    # VSCode thường có nền màu xám đồng nhất > 40%
    if dominant_ratio > 0.4 and 80 < dominant_color < 180:
        return 0.0  # Likely IDE interface

    mean_brightness = np.mean(gray)
    if mean_brightness > 240 or mean_brightness < 30:  # Quá sáng/tối -> không phải captcha
        return 0.0

    confidence = 0.0

    # 1. Tìm missing piece patterns - puzzle pieces thường có hình dạng đặc biệt
    edges = cv2.Canny(gray, 40, 120)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    puzzle_pieces = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Đủ lớn để là puzzle piece
            # Kiểm tra độ phức tạp của contour (puzzle pieces có hình dạng phức tạp)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)

                # Puzzle pieces có compactness thấp (hình dạng phức tạp)
                if 0.1 < compactness < 0.6:
                    puzzle_pieces += 1

                    # Kiểm tra có tabs/blanks đặc trưng của puzzle không
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    if solidity < 0.9:  # Có indentations
                        confidence += 0.2

    if puzzle_pieces >= 1:
        confidence += 0.3
        if puzzle_pieces >= 2:  # Có nhiều pieces
            confidence += 0.2

    # 2. Tìm thanh trượt ở dưới - QUAN TRỌNG cho slide captcha
    bottom_area = gray[int(ph*0.75):, :]  # Vùng dưới 25%
    if bottom_area.size > 0:
        bottom_edges = cv2.Canny(bottom_area, 30, 100)

        # Tìm đường ngang dài (slide track)
        lines = cv2.HoughLinesP(bottom_edges, 1, np.pi/180, threshold=max(15, pw//15),
                                minLineLength=max(pw//3, 80), maxLineGap=10)

        strong_horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180/np.pi)
                if angle < 15 and length > pw*0.3:  # Thực sự ngang và dài
                    strong_horizontal_lines += 1
                    confidence += 0.25

        # Tìm nút trượt (slider button)
        circles = cv2.HoughCircles(bottom_area, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=50, param2=25, minRadius=8, maxRadius=40)
        if circles is not None and len(circles[0]) > 0:
            confidence += 0.2

    # 3. Kiểm tra có text instruction ("Kéo mảnh ghép vào vị trí")
    text_regions = [gray[0:int(ph*0.3), :], gray[int(ph*0.7):, :]]

    for text_area in text_regions:
        if text_area.size > 0:
            # Tìm text bằng morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (pw//15, 3))
            text_thresh = cv2.threshold(
                text_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            dilated = cv2.dilate(text_thresh, kernel)
            text_contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in text_contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > pw*0.3 and 15 < h < ph*0.2:  # Text region hợp lý
                    confidence += 0.15
                    break

    # 4. Kiểm tra background complexity (slide captcha thường có background phong cảnh)
    texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if texture_var > 50:  # Background có texture
        confidence += 0.1

    # 5. Yêu cầu kích thước tối thiểu
    if ph < 100 or pw < 200:
        confidence *= 0.5

    return min(confidence, 1.0)


def detect_rotate_captcha_improved(panel: np.ndarray) -> float:
    """Cải thiện detection cho rotate captcha - nhận diện puzzle xoay tròn"""
    ph, pw = panel.shape[:2]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    confidence = 0.0

    # 1. Tìm vòng tròn chính ở giữa - ĐẶC TRƯNG chính của rotate captcha
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=80, param2=40, minRadius=min(ph, pw)//8, maxRadius=min(ph, pw)//2)

    best_circle = None
    if circles is not None:
        circles = circles[0, :]

        for x, y, r in circles:
            # Kiểm tra vị trí gần trung tâm
            center_dist = np.sqrt((x - pw/2)**2 + (y - ph/2)**2)
            if center_dist < min(pw, ph) * 0.25:  # Gần trung tâm

                # Kiểm tra có content phức tạp bên trong vòng tròn không
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
                masked_region = cv2.bitwise_and(gray, mask)

                roi = masked_region[max(0, int(y-r)):min(ph, int(y+r)),
                                    max(0, int(x-r)):min(pw, int(x+r))]

                if roi.size > 0:
                    # Kiểm tra texture complexity bên trong
                    texture = cv2.Laplacian(roi, cv2.CV_64F).var()
                    edge_content = cv2.Canny(roi, 50, 150).sum()

                    if texture > 100 and edge_content > 1000:  # Có pattern phức tạp
                        best_circle = (x, y, r)
                        confidence += 0.5

                        # Kiểm tra có pattern sector/segment không
                        center_region = roi[max(0, int(r*0.3)):min(roi.shape[0], int(r*1.7)),
                                            max(0, int(r*0.3)):min(roi.shape[1], int(r*1.7))]

                        if center_region.size > 0:
                            edges = cv2.Canny(center_region, 40, 120)
                            lines = cv2.HoughLines(
                                edges, 1, np.pi/180, threshold=20)

                            if lines is not None and len(lines) >= 3:
                                confidence += 0.2  # Có pattern radial
                        break

    # 2. Tìm thanh trượt dưới cho rotate-by-slider
    bottom_area = gray[int(ph*0.8):, :]
    if bottom_area.size > 0:
        bottom_edges = cv2.Canny(bottom_area, 30, 100)
        lines = cv2.HoughLinesP(bottom_edges, 1, np.pi/180, threshold=15,
                                minLineLength=pw//4, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180/np.pi)
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if angle < 15 and length > pw*0.25:  # Horizontal slider
                    confidence += 0.2
                    break

        # Tìm slider button
        slider_circles = cv2.HoughCircles(bottom_area, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                          param1=50, param2=25, minRadius=8, maxRadius=30)
        if slider_circles is not None:
            confidence += 0.15

    # 3. Kiểm tra có text instruction ("Kéo thanh trượt để ghép hình")
    text_areas = [gray[0:int(ph*0.25), :], gray[int(ph*0.85):, :]]

    for text_area in text_areas:
        if text_area.size > 0:
            # Morphology để detect text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (pw//20, 3))
            text_thresh = cv2.threshold(
                text_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            dilated = cv2.dilate(text_thresh, kernel)
            text_contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in text_contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > pw*0.3 and 15 < h < ph*0.15:  # Text region hợp lý
                    confidence += 0.15
                    break

    # 4. Kiểm tra màu sắc đa dạng (rotate puzzle thường colorful)
    if panel.shape[2] == 3:  # Color image
        hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]

        # Đếm số màu khác nhau trong vòng tròn chính
        if best_circle:
            x, y, r = best_circle
            mask = np.zeros(h_channel.shape, dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            masked_hue = cv2.bitwise_and(h_channel, mask)

            unique_hues = len(np.unique(masked_hue[masked_hue > 0]))
            if unique_hues > 20:  # Nhiều màu sắc
                confidence += 0.1

    # 5. Aspect ratio check - rotate captcha thường gần vuông
    aspect_ratio = pw / ph
    if 0.6 < aspect_ratio < 1.5:  # Không quá dài/rộng
        confidence += 0.1

    return min(confidence, 1.0)


def detect_object_selection(panel: np.ndarray) -> float:
    """Detection cho object selection (chọn 2 đối tượng giống nhau) - cải thiện cho ảnh thật"""
    ph, pw = panel.shape[:2]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    confidence = 0.0

    # 1. Kiểm tra background sáng (object selection có nền trắng/sáng)
    mean_brightness = np.mean(gray)
    if mean_brightness > 200:  # Nền sáng
        confidence += 0.2

    # 2. Tìm các object riêng biệt bằng threshold adaptive
    # Object selection có các items tối trên nền sáng
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Loại bỏ noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc objects có kích thước hợp lý
    valid_objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Đủ lớn để là object
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Object không quá dài/rộng và không ở biên
            if (0.2 < aspect_ratio < 5.0 and
                x > pw*0.05 and y > ph*0.1 and
                    x+w < pw*0.95 and y+h < ph*0.9):
                valid_objects.append((x, y, w, h, area))

    # Object selection thường có 4-8 objects
    object_count = len(valid_objects)
    if 4 <= object_count <= 10:
        confidence += 0.4

        # Bonus nếu objects được phân bố đều
        if object_count >= 4:
            # Kiểm tra có layout dạng grid không
            x_coords = [obj[0] + obj[2] //
                        2 for obj in valid_objects]  # center x
            y_coords = [obj[1] + obj[3] //
                        2 for obj in valid_objects]  # center y

            # Đếm số hàng và cột
            unique_rows = len(set([int(y//(ph//4)) for y in y_coords]))
            unique_cols = len(set([int(x//(pw//4)) for x in x_coords]))

            if unique_rows >= 2 and unique_cols >= 2:  # Grid layout
                confidence += 0.3

    # 3. Kiểm tra có text instruction phía trên
    top_area = gray[0:int(ph*0.4), :]
    if top_area.size > 0:
        # Tìm vùng text bằng morphology
        kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (pw//20, 5))
        text_thresh = cv2.threshold(
            top_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        text_dilated = cv2.dilate(text_thresh, kernel_text)

        text_contours, _ = cv2.findContours(
            text_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in text_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > pw*0.4 and 20 < h < ph*0.2:  # Text region hợp lý
                confidence += 0.2
                break

    # 4. Loại trừ slide/rotate patterns
    # Không có vòng tròn lớn ở giữa
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=80, param2=40, minRadius=min(ph, pw)//6, maxRadius=min(ph, pw)//3)
    if circles is None:
        confidence += 0.1

    # Không có thanh slide dưới
    bottom_area = gray[int(ph*0.8):, :]
    if bottom_area.size > 0:
        bottom_edges = cv2.Canny(bottom_area, 30, 100)
        slide_lines = cv2.HoughLinesP(bottom_edges, 1, np.pi/180,
                                      threshold=20, minLineLength=pw//3, maxLineGap=10)
        if slide_lines is None:
            confidence += 0.1

    return min(confidence, 1.0)


def detect_rotate_app_pattern(panel: np.ndarray) -> bool:
    """Phân biệt rotate APP vs WEB"""
    ph, pw = panel.shape[:2]

    # APP thường có:
    # - Vòng tròn nhỏ hơn so với tổng thể
    # - Có thanh slide hoặc button ở dưới
    # - Tỷ lệ vòng tròn/panel < 0.4

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=80, param2=40, minRadius=20, maxRadius=min(ph, pw)//3)

    if circles is not None:
        largest_r = max(circles[0, :, 2])
        circle_ratio = largest_r / min(ph, pw)

        # APP: ratio < 0.4, WEB: ratio >= 0.4
        return circle_ratio < 0.4

    return True  # Default to APP

# ================= MOUSE AUTOMATION =================


def safe_mouse_move(x, y, duration=0.3):
    """Cross-platform safe mouse movement"""
    try:
        # print(f"🖱️ Di chuyển chuột đến ({x}, {y}) trong {duration}s")

        if SYSTEM == "Darwin":  # macOS
            # Use faster movement for macOS
            pag.moveTo(x, y, duration=duration)
            time.sleep(0.1)  # Small delay after movement
        else:  # Windows/Linux
            pag.moveTo(x, y, duration=duration)

        # Verify position
        current_pos = pag.position()
        # print(f"✅ Vị trí chuột hiện tại: {current_pos}")
        return True
    except Exception as e:
        print(f"❌ Lỗi di chuyển chuột: {e}")
        return False


def safe_mouse_click(x, y, button='left', clicks=1, duration=0.2):
    """Cross-platform safe mouse click"""
    try:
        if not safe_mouse_move(x, y, duration):
            return False

        time.sleep(0.2)

        if SYSTEM == "Darwin":  # macOS
            pag.mouseDown(x, y, button=button)
            time.sleep(0.1)
            pag.mouseUp(x, y, button=button)
            time.sleep(0.2)
        else:  # Windows/Linux
            pag.click(x, y, clicks=clicks, interval=0.1, button=button)
            time.sleep(0.2)

        # print(f"✅ Đã click tại ({x}, {y})")
        return True
    except Exception as e:
        print(f"❌ Lỗi click chuột: {e}")
        return False


def safe_mouse_drag(start_x, start_y, end_x, end_y, duration=0.5):
    """Cross-platform safe mouse drag"""
    try:
        if not safe_mouse_move(start_x, start_y, 0.2):
            return False

        time.sleep(0.1)

        if SYSTEM == "Darwin":  # macOS
            pag.mouseDown(start_x, start_y, button='left')
            time.sleep(0.1)
            pag.moveTo(end_x, end_y, duration=duration)
            time.sleep(0.1)
            pag.mouseUp(end_x, end_y, button='left')
        else:  # Windows/Linux
            pag.drag(end_x, end_y, duration, button='left')

        print(f"✅ Đã kéo từ ({start_x}, {start_y}) đến ({end_x}, {end_y})")
        return True
    except Exception as e:
        print(f"❌ Lỗi kéo chuột: {e}")
        return False


def drag_slider_abs(start_pos, end_pos):
    """Drag slider from start to end position"""
    start_x, start_y = start_pos
    end_x, end_y = end_pos

    return safe_mouse_drag(start_x, start_y, end_x, end_y, duration=1.0)


def rotate_by_angle(center_pos, angle, radius):
    """Rotate by dragging in a circular motion"""
    cx, cy = center_pos

    # Start position
    start_x = cx + radius
    start_y = cy

    # Calculate end position based on angle
    angle_rad = math.radians(angle)
    end_x = cx + radius * math.cos(angle_rad)
    end_y = cy + radius * math.sin(angle_rad)

    return safe_mouse_drag(start_x, start_y, end_x, end_y, duration=1.5)


def click_points_abs(points):
    """Click multiple points in sequence"""
    success = True
    for i, (x, y) in enumerate(points):
        # print(f"🖱️ Click điểm {i+1}/{len(points)}: ({x}, {y})")
        if not safe_mouse_click(x, y, duration=0.3):
            success = False
            break
        time.sleep(0.5)  # Delay giữa các click

    if success:
        print(f"✅ Hoàn thành click {len(points)} điểm")
    return success

# ================= SOLVER FUNCTIONS =================


def solve_slide_captcha(roi_abs, roi_img, region_coords):
    """Giải slide captcha bằng HMcaptcha API"""
    print("↗️ HMcaptcha slide captcha...")

    res = post_json("slide", pil_to_b64(roi_img))
    if "error" in res:
        print(f"❌ API error: {res['error']}")
        return False

    print("🟦 Kết quả:", json.dumps(res, indent=2, ensure_ascii=False))

    if res.get("Status") == "SUCCESS" and "Data" in res:
        data = res["Data"]

        # Lấy offset và tọa độ từ API
        offset = data.get("offset", 0)
        x_api = data.get("x", 0)  # Tọa độ trong ảnh
        y_api = data.get("y", 0)

        # Chuyển đổi tọa độ tương đối thành tuyệt đối
        region_x, region_y, region_w, region_h = region_coords
        start_x = region_x + x_api
        start_y = region_y + y_api
        end_x = start_x + offset
        end_y = start_y

        print(
            f"🎯 Kéo từ ({start_x}, {start_y}) đến ({end_x}, {end_y}) (offset: {offset}px)")
        time.sleep(0.9)
        drag_slider_abs((start_x, start_y), (end_x, end_y))
        print("✅ DONE slide captcha")
        return True

    return False


def solve_rotate_captcha(roi_abs, roi_img, rotate_type, region_coords):
    """Giải rotate captcha bằng HMcaptcha API"""
    print(f"↗️ HMcaptcha {rotate_type} captcha...")

    res = post_json(rotate_type, pil_to_b64(roi_img))
    if "error" in res:
        print(f"❌ API error: {res['error']}")
        return False

    print("🟦 Kết quả:", json.dumps(res, indent=2, ensure_ascii=False))

    if res.get("Status") == "SUCCESS" and "Data" in res:
        data = res["Data"]
        angle = data.get("angle", 0)

        if rotate_type == "rotate_app":
            # APP: sử dụng point_slide
            point_slide = data.get("point_slide", {})
            x_api = point_slide.get("x", 0)
            y_api = point_slide.get("y", 0)

            # Chuyển đổi tọa độ
            region_x, region_y, region_w, region_h = region_coords
            start_x = region_x + x_api
            start_y = region_y + y_api

            # Tính offset theo công thức: offset = angle * (width_slide/180)
            width_slide = region_w  # Sử dụng width của region
            offset = angle * (width_slide / 180)

            end_x = start_x + int(offset)
            end_y = start_y

            print(f"🎯 APP rotate: angle={angle}°, offset={offset:.1f}px")
            print(f"🎯 Kéo từ ({start_x}, {start_y}) đến ({end_x}, {end_y})")
            time.sleep(0.9)
            drag_slider_abs((start_x, start_y), (end_x, end_y))

        else:  # rotate_web
            # WEB: xoay theo góc tại trung tâm
            region_x, region_y, region_w, region_h = region_coords
            cx_abs = region_x + region_w//2
            cy_abs = region_y + region_h//2
            radius = int(min(region_w, region_h)*0.3)

            print(f"🎯 WEB rotate: xoay {angle}° tại ({cx_abs}, {cy_abs})")
            time.sleep(0.9)
            rotate_by_angle((cx_abs, cy_abs), angle, radius)

        print("✅ DONE rotate captcha")
        return True

    return False


def solve_object_captcha(roi_abs, roi_img, region_coords):
    """Giải object selection captcha bằng HMcaptcha API"""
    print("↗️ HMcaptcha object selection...")

    res = post_json("object", pil_to_b64(roi_img))
    if "error" in res:
        print(f"❌ API error: {res['error']}")
        return False

    print("🟦 Kết quả:", json.dumps(res, indent=2, ensure_ascii=False))

    if res.get("Status") == "SUCCESS" and "Data" in res:
        data = res["Data"]
        raw_coords = data.get("raw", "")

        if raw_coords:
            # Parse "x1,y1|x2,y2" format (tỷ lệ 0-1)
            clicks = []
            region_x, region_y, region_w, region_h = region_coords

            for coord_pair in raw_coords.split('|'):
                if ',' in coord_pair:
                    x_prop, y_prop = coord_pair.split(',')
                    x_prop, y_prop = float(x_prop), float(y_prop)

                    # Chuyển tỷ lệ thành tọa độ tuyệt đối
                    x_abs = region_x + int(x_prop * region_w)
                    y_abs = region_y + int(y_prop * region_h)
                    clicks.append((x_abs, y_abs))

            if clicks:
                print(f"🎯 Click tại {len(clicks)} điểm: {clicks}")
                time.sleep(0.6)
                click_points_abs(clicks)
                print("✅ DONE object selection")
                return True

    return False


def crop_captcha_roi(panel_cv, captcha_type):
    """Cắt chính xác vùng captcha dựa trên loại"""
    ph, pw = panel_cv.shape[:2]

    if captcha_type == "slide":
        # Slide captcha: chỉ lấy vùng puzzle, bỏ thanh trượt và text
        # Thường puzzle ở 20-80% chiều cao
        top_margin = int(ph * 0.15)  # Bỏ text instruction phía trên
        bottom_margin = int(ph * 0.85)  # Bỏ slider track phía dưới
        left_margin = int(pw * 0.05)
        right_margin = int(pw * 0.95)

        return panel_cv[top_margin:bottom_margin, left_margin:right_margin]

    elif captcha_type in ("rotate_app", "rotate_web"):
        # Rotate captcha: lấy vùng có vòng tròn chính
        gray = cv2.cvtColor(panel_cv, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=80, param2=40, minRadius=min(ph, pw)//8, maxRadius=min(ph, pw)//2)

        if circles is not None:
            # Tìm vòng tròn lớn nhất gần trung tâm
            best_circle = None
            for x, y, r in circles[0, :]:
                center_dist = np.sqrt((x - pw/2)**2 + (y - ph/2)**2)
                if center_dist < min(pw, ph) * 0.25:
                    best_circle = (x, y, r)
                    break

            if best_circle:
                x, y, r = best_circle
                # Cắt vuông quanh vòng tròn với padding
                padding = int(r * 0.3)
                x1 = max(0, int(x - r - padding))
                y1 = max(0, int(y - r - padding))
                x2 = min(pw, int(x + r + padding))
                y2 = min(ph, int(y + r + padding))

                return panel_cv[y1:y2, x1:x2]

        # Fallback: crop giữa
        margin = int(min(pw, ph) * 0.1)
        return panel_cv[margin:ph-margin, margin:pw-margin]

    elif captcha_type == "object":
        # Object selection: bỏ text instruction, chỉ lấy vùng objects
        top_margin = int(ph * 0.25)  # Bỏ text instruction
        bottom_margin = int(ph * 0.95)
        left_margin = int(pw * 0.05)
        right_margin = int(pw * 0.95)

        return panel_cv[top_margin:bottom_margin, left_margin:right_margin]

    else:
        # Default: crop nhẹ các biên
        margin = 20
        return panel_cv[margin:ph-margin, margin:pw-margin]


def detect_captcha_type_by_text(image):
    """Nhận diện loại captcha dựa trên text trong ảnh"""
    if not TESSERACT_AVAILABLE:
        print("⚠️ Tesseract không khả dụng, sử dụng fallback detection")
        return "unknown"

    try:
        import pytesseract

        # Convert PIL to OpenCV nếu cần
        if isinstance(image, Image.Image):
            img_cv = pil_to_cv(image)
        else:
            img_cv = image

        # Preprocess ảnh để OCR tốt hơn
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Thử nhiều preprocessing khác nhau
        texts = []

        # Method 1: Ảnh gốc
        text1 = pytesseract.image_to_string(
            gray, lang='vie+eng', config='--psm 6')
        texts.append(text1.lower())

        # Method 2: Threshold
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text2 = pytesseract.image_to_string(
            thresh, lang='vie+eng', config='--psm 6')
        texts.append(text2.lower())

        # Method 3: Chỉ lấy vùng trên (thường có text instruction)
        height = gray.shape[0]
        top_region = gray[0:int(height*0.3), :]
        text3 = pytesseract.image_to_string(
            top_region, lang='vie+eng', config='--psm 6')
        texts.append(text3.lower())

        # Kết hợp tất cả text
        all_text = ' '.join(texts)
        # print(f"🔤 Text phát hiện: {all_text[:100]}...")

        # Phân loại dựa trên từ khóa
        if any(keyword in all_text for keyword in ['chọn 2 đối tượng', 'chon 2 doi tuong', 'select 2 objects', 'giống nhau']):
            return "object"
        elif any(keyword in all_text for keyword in ['kéo mảnh ghép', 'keo manh ghep', 'drag the puzzle', 'vào vị trí', 'trượt']):
            return "slide"
        elif any(keyword in all_text for keyword in ['kéo thanh trượt', 'keo thanh truot', 'drag the slider', 'ghép hình']):
            return "rotate_app"
        else:
            print("🚫 Không phát hiện Captcha")
            return "unknown"

    except ImportError:
        print("⚠️ Cần cài đặt pytesseract: pip install pytesseract")
        return "unknown"
    except Exception as e:
        print(f"⚠️ Lỗi OCR: {e}")
        return "unknown"


def try_all_captcha_types(region_img, region_coords):
    """Thử nhận diện captcha bằng text trước, nếu không được thì thử cả 3 loại"""

    actual_image_size = region_img.size if region_img else None
    # Pass captured image size to coordinate calculation
    captured_image_size = actual_image_size

    # Bước 1: Thử nhận diện bằng text
    detected_type = detect_captcha_type_by_text(region_img)

    if detected_type != "unknown":
        print(f"🎯 Phát hiện: {detected_type}")
        res = post_json(detected_type, pil_to_b64(region_img))
        if "error" not in res and res.get("Status") == "SUCCESS":
            return execute_captcha_action(detected_type, res["Data"], region_coords, actual_image_size, captured_image_size)
    else:
        # Không phát hiện được loại captcha, bỏ qua để tránh tiêu tốn tài nguyên
        print("⏭️ Bỏ qua captcha này - không thử fallback")
        return False
    captcha_types = ["slide", "rotate_app", "object"]

    for captcha_type in captcha_types:
        res = post_json(captcha_type, pil_to_b64(region_img))
        if "error" not in res and res.get("Status") == "SUCCESS" and "Data" in res:
            print(f"✅ Thành công: {captcha_type}")
            return execute_captcha_action(captcha_type, res["Data"], region_coords, actual_image_size, captured_image_size)

    return False


def calculate_simple_coordinates(x_api, y_api, region_info, captured_image_size=None):
    """
    Simple coordinate calculation based on captured image size
    Logic: If captured image > 1000px width, it means 2x scaling, divide by 2
    """
    region_x, region_y, region_w, region_h = region_info

    print(f"� Simple coord calc: API=({x_api},{y_api})")
    print(f"📐 Region: ({region_x},{region_y}) size={region_w}x{region_h}")

    # Check if we have captured image size info
    if captured_image_size:
        captured_w, captured_h = captured_image_size
        print(f"📷 Captured image size: {captured_w}x{captured_h}")

        # If captured image width > 1000, it means 2x scaling on Retina
        if captured_w > 1000:
            print(f"🔍 Detected 2x scaling (captured_w={captured_w} > 1000)")
            # Scale down API coordinates by 2
            effective_x = x_api / 2.0
            effective_y = y_api / 2.0
            print(
                f"📉 Scaled API coords: ({effective_x:.1f},{effective_y:.1f})")
        else:
            print(f"🔍 No scaling detected (captured_w={captured_w} <= 1000)")
            effective_x = x_api
            effective_y = y_api
    else:
        print(f"⚠️ No captured image size, using API coords directly")
        effective_x = x_api
        effective_y = y_api

    # Convert to screen coordinates using simple proportion
    # Assume API coordinates are in 716x784 space (your default capture size)
    api_reference_w = 716
    api_reference_h = 784

    x_ratio = effective_x / api_reference_w
    y_ratio = effective_y / api_reference_h

    x_screen = region_x + int(x_ratio * region_w)
    y_screen = region_y + int(y_ratio * region_h)

    print(
        f"📍 Final mapping: ratio=({x_ratio:.3f},{y_ratio:.3f}) → screen=({x_screen},{y_screen})")

    return x_screen, y_screen


def execute_captcha_action(captcha_type, data, region_coords, actual_image_size=None, captured_image_size=None):
    """Simple captcha action with image-size-based coordinate calculation"""
    region_x, region_y, region_w, region_h = region_coords

    if captcha_type == "slide":
        # Lấy tham số từ API
        x_api = data.get("x", 0)
        y_api = data.get("y", 0)
        offset = data.get("offset", 0)

        print(
            f"🎯 Slide captcha: API coords=({x_api}, {y_api}), offset={offset}")
        print(
            f"📐 Region: ({region_x}, {region_y}) size=({region_w}x{region_h})")

        # Chọn kích thước tham chiếu "có gì dùng nấy"
        if actual_image_size:
            ref_w, ref_h = actual_image_size
            print(f"📏 Tham chiếu: actual_image_size = {ref_w}x{ref_h}")
        elif captured_image_size:
            ref_w, ref_h = captured_image_size
            print(f"📏 Tham chiếu: captured_image_size = {ref_w}x{ref_h}")
        else:
            # Không có kích thước ảnh: coi như tọa độ đã cùng hệ với region
            ref_w, ref_h = region_w, region_h
            print(f"📏 Tham chiếu: dùng luôn region size = {ref_w}x{ref_h}")

        # Map tọa độ API sang tọa độ màn hình theo tỉ lệ
        # (nếu ref == region thì tỉ lệ = 1:1)
        # Tránh chia cho 0
        ref_w = max(1, int(ref_w))
        ref_h = max(1, int(ref_h))

        x_ratio = x_api / ref_w
        y_ratio = y_api / ref_h

        x1 = region_x + int(x_ratio * region_w)
        y1 = region_y + int(y_ratio * region_h)

        print(
            f"📍 Mapping: ratio=({x_ratio:.3f},{y_ratio:.3f}) → screen=({x1},{y1})")

        # Tính offset kéo:
        # Nếu có ref_w (actual/captured), quy đổi offset theo tỉ lệ sang region_w
        if offset and (actual_image_size or captured_image_size):
            offset_screen = int((offset / ref_w) * region_w)
            print(
                f"➡️ Offset theo tỉ lệ: {offset} / {ref_w} * {region_w} = {offset_screen}px")
        elif offset:
            # Không có kích thước ảnh: coi offset là pixel trực tiếp
            offset_screen = int(offset)
            print(f"➡️ Offset trực tiếp (px): {offset_screen}px")
        else:
            # Offset không có: đặt mặc định
            offset_screen = region_w // 3
            print(f"➡️ Offset mặc định: {offset_screen}px")

        x2 = x1 + offset_screen
        y2 = y1

        # Giới hạn trong vùng
        margin = 10
        max_x = region_x + region_w - margin
        min_x = region_x + margin
        if x2 > max_x:
            x2 = max_x
            print(f"⚠️ Điều chỉnh x2 để không vượt phải: {x2}")
        elif x2 < min_x:
            x2 = min_x
            print(f"⚠️ Điều chỉnh x2 để không vượt trái: {x2}")

        print(
            f"🖱️ Slide drag: from ({x1}, {y1}) to ({x2}, {y2}) (offset_screen: {x2 - x1}px)")

        time.sleep(0.5)
        success = safe_mouse_drag(x1, y1, x2, y2, duration=1.0)
        return success

    elif captcha_type == "rotate_app":

        angle = data.get("angle", 0)
        point_slide = data.get("point_slide", {})
        x_api = point_slide.get("x", 0)
        y_api = point_slide.get("y", 0)

        print(f"🔄 Rotate APP: angle={angle}°, API coords=({x_api}, {y_api})")
        print(f"📐 Region: ({region_x}, {region_y}) size=({region_w}x{region_h})")

        # Chọn kích thước tham chiếu "có gì dùng nấy"
        if actual_image_size:
            ref_w, ref_h = actual_image_size
            print(f"📏 Tham chiếu: actual_image_size = {ref_w}x{ref_h}")
        elif captured_image_size:
            ref_w, ref_h = captured_image_size
            print(f"📏 Tham chiếu: captured_image_size = {ref_w}x{ref_h}")
        else:
            ref_w, ref_h = region_w, region_h
            print(f"📏 Tham chiếu: dùng luôn region size = {ref_w}x{ref_h}")

        # Tránh chia 0
        ref_w = max(1, int(ref_w))
        ref_h = max(1, int(ref_h))

        # Map tọa độ theo tỉ lệ
        x_ratio = x_api / ref_w
        y_ratio = y_api / ref_h

        # Điều chỉnh Y để trỏ lên trên thanh xoay một chút
        y_adjustment_pixels = 30

        x1 = region_x + int(x_ratio * region_w)
        y1 = region_y + int(y_ratio * region_h) - y_adjustment_pixels

        # Không vượt ra ngoài vùng theo trục Y
        if y1 < region_y:
            y1 = region_y + 5
            print(f"⚠️ Y adjusted to stay within region: {y1}")

        print(
            f"📍 Mapping: ratio=({x_ratio:.3f},{y_ratio:.3f}) → screen=({x1},{y1})")
        print(f"🔧 Applied Y adjustment: -{y_adjustment_pixels}px")

        # Ước lượng bề rộng slider (80% chiều rộng region)
        actual_slider_width = region_w * 0.8
        print(f"📐 Slider width (est): {actual_slider_width:.0f}px")

        # Offset kéo theo công thức: offset = angle * (slider_width / 180)
        offset = angle * (actual_slider_width / 180.0)
        print(
            f"📏 Offset: angle={angle}° × ({actual_slider_width:.0f}/180) = {offset:.1f}px")

        # Tính điểm kết thúc và ràng buộc trong vùng slider
        x2 = x1 + int(offset)
        y2 = y1

        margin = 20
        max_x = region_x + int(region_w * 0.85)  # giới hạn phải ~85% vùng
        min_x = region_x + margin

        if x2 > max_x:
            x2 = max_x
            print(f"⚠️ Adjusted x2 to stay within slider bounds: {x2}")
        elif x2 < min_x:
            x2 = min_x
            print(f"⚠️ Adjusted x2 to stay within slider bounds: {x2}")

        # Giới hạn độ dài kéo tối đa 90% bề rộng slider
        drag_distance = abs(x2 - x1)
        max_reasonable_drag = actual_slider_width * 0.9
        if drag_distance > max_reasonable_drag:
            scale_factor = max_reasonable_drag / drag_distance
            new_offset = int((x2 - x1) * scale_factor)
            x2 = x1 + new_offset
            print(
                f"🔧 Scaled drag: {drag_distance:.1f}px → {new_offset}px (factor: {scale_factor:.2f})")

        print(f"🖱️ Rotate drag: from ({x1}, {y1}) to ({x2}, {y2}) "
            f"[drag={abs(x2-x1)}px, slider≈{actual_slider_width:.0f}px]")

        time.sleep(0.5)
        success = safe_mouse_drag(x1, y1, x2, y2, duration=1.5)
        return success

    elif captcha_type == "object":
        raw_coords = data.get("raw", "")

        print(f"🎯 Object selection: raw_coords='{raw_coords}'")
        print(
            f"📐 Region: ({region_x}, {region_y}) size=({region_w}x{region_h})")

        # Get monitor info for scaling detection
        monitor_info = get_monitor_info()
        scaling_factor = monitor_info.get("scaling_factor", 1.0)
        is_retina = monitor_info.get("is_retina", False)

        print(f"🖥️ Display: scaling={scaling_factor}, retina={is_retina}")

        if raw_coords:
            clicks = []
            for coord_pair in raw_coords.split('|'):
                if ',' in coord_pair:
                    x_str, y_str = coord_pair.split(',')
                    x_api, y_api = float(x_str), float(y_str)

                    # Check if API coordinates are ratios (0-1) or absolute
                    if x_api <= 1.0 and y_api <= 1.0:
                        # API sends ratios from 0 to 1, convert to absolute within region
                        x_screen = region_x + int(x_api * region_w)
                        y_screen = region_y + int(y_api * region_h)
                        print(
                            f"🎯 Object click (ratio): API=({x_api:.1f},{y_api:.1f}) → screen=({x_screen}, {y_screen})")
                    else:
                        # API sends absolute coordinates, use coordinate mapping
                        x_screen, y_screen = calculate_simple_coordinates(
                            x_api, y_api,
                            (region_x, region_y, region_w, region_h),
                            captured_image_size
                        )
                        print(
                            f"🎯 Object click (absolute): API=({x_api:.1f},{y_api:.1f}) → screen=({x_screen}, {y_screen})")

                    clicks.append((x_screen, y_screen))

            if clicks:
                print(f"🖱️ Will click {len(clicks)} points")
                time.sleep(0.5)
                success = click_points_abs(clicks)
                return success

    return False

# ================= MAIN SOLVER =================


def solve_captcha_once():
    """Quét captcha một lần - logic đơn giản hóa"""
    print("� Bắt đầu quét captcha...")

    if not FIXED_CAPTCHA_REGION["enabled"]:
        print(
            "❌ Vùng cố định chưa được bật. Vui lòng bật FIXED_CAPTCHA_REGION['enabled'] = True")
        return False

    # Lấy thông tin vùng cố định
    region_coords = get_fixed_region_coords()
    print(
        f"📍 Chụp vùng giữa màn hình: {region_coords[2]}x{region_coords[3]} tại ({region_coords[0]}, {region_coords[1]})")

    # Chụp vùng cố định
    result = screenshot_fixed_region()
    if result[0] is None:
        print("❌ Không thể chụp vùng cố định")
        return False

    region_img, region_coords = result

    # Lưu ảnh debug (chỉ 1 ảnh)
    save_screenshot(region_img, "vung_captcha")

    # Thử cả 3 loại captcha
    return try_all_captcha_types(region_img, region_coords)


def setup_fixed_coordinates():
    """Helper function để setup tọa độ cố định cho màn hình hiện tại"""
    monitor_info = get_monitor_info()
    region_coords = get_fixed_region_coords()

    print(
        f"🖥️ Monitor hiện tại: {monitor_info['width']}x{monitor_info['height']}")
    print(
        f"📍 Vùng captcha hiện tại: ({region_coords[0]}, {region_coords[1]}) {region_coords[2]}x{region_coords[3]}")
    print(
        f"📊 Tỷ lệ hiện tại: x={FIXED_CAPTCHA_REGION['x_percent']:.1%}, y={FIXED_CAPTCHA_REGION['y_percent']:.1%}")
    print(
        f"📊 Kích thước: w={FIXED_CAPTCHA_REGION['width_percent']:.1%}, h={FIXED_CAPTCHA_REGION['height_percent']:.1%}")

    print("\n🔧 Để cập nhật vùng cố định, sửa FIXED_CAPTCHA_REGION trong code:")
    print("   - x_percent, y_percent: tỷ lệ % vị trí so với màn hình")
    print("   - width_percent, height_percent: tỷ lệ % kích thước so với màn hình")
    print("   - enabled: True để dùng vùng cố định, False để auto-detect")

    # Test chụp vùng hiện tại
    result = screenshot_fixed_region()
    if result[0] is not None:
        test_img, coords = result
        save_screenshot(test_img, "test_fixed_region")
        print(f"✅ Đã lưu ảnh test: screenshots/test_fixed_region_*.png")

        # Test detect captcha
        test_cv = pil_to_cv(test_img)
        ctype, confidence = detect_captcha_type_improved(test_cv)
        print(f"🔍 Test detection: {ctype} (confidence: {confidence:.2f})")
    else:
        print("❌ Không thể chụp vùng cố định với tọa độ hiện tại")


def main():
    if not HM_API_KEY or HM_API_KEY == "":
        raise SystemExit("⚠️ Chưa cấu hình HM_API_KEY.")

    # Check for setup mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_fixed_coordinates()
        return

    print("🚀 Hệ thống giải Captcha tự động!")

    if FIXED_CAPTCHA_REGION["enabled"]:
        region_coords = get_fixed_region_coords()
        print(
            f"📍 Chế độ: Vùng cố định {region_coords[2]}x{region_coords[3]} ở giữa màn hình")
    else:
        print("❌ Vui lòng bật vùng cố định trong config")
        return

    print(f"⏰ Thời gian quét: mỗi {SCAN_INTERVAL}s")
    # print(f"📁 Ảnh debug lưu tại: {SCREENSHOT_DIR}/")
    print("⛔ Nhấn Ctrl+C để dừng")
    print("💡 Chạy 'python init.py setup' để cấu hình tọa độ")
    print("-" * 50)

    scan_count = 0
    success_count = 0

    try:
        while True:
            STATS["attempts"] += 1

            try:
                if solve_captcha_once():
                    STATS["successes"] += 1
                    print(f"✅ Thành công")
                else:
                    STATS["failures"] += 1
                    print(f"❌ Thất bại")

            except Exception as e:
                STATS["failures"] += 1
                print(f"❌ Lỗi: {e}")

            # Show stats
            success_rate = (STATS["successes"] / STATS["attempts"]
                            ) * 100 if STATS["attempts"] > 0 else 0
            print(
                f"📊 {STATS['successes']}/{STATS['attempts']} ({success_rate:.1f}%)")

            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        # Calculate final stats and cost
        total_attempts = STATS["attempts"]
        total_successes = STATS["successes"]
        total_failures = STATS["failures"]
        success_rate = (total_successes / total_attempts) * \
            100 if total_attempts > 0 else 0
        estimated_cost = (total_successes / 1000) * STATS["cost_per_thousand"]

        print(f"\n🛑 Đã dừng")
        print(f"📊 Thống kê:")
        print(f"   Tổng số lần: {total_attempts}")
        print(f"   Thành công: {total_successes}")
        print(f"   Thất bại: {total_failures}")
        print(f"   Tỷ lệ thành công: {success_rate:.1f}%")
        print(
            f"💰 Chi phí ước tính: ${estimated_cost:.4f} (${STATS['cost_per_thousand']}/1000 lần thành công)")


if __name__ == "__main__":
    main()
