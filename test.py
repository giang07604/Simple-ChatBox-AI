# -*- coding: utf-8 -*-
"""
Simple HMcaptcha Solver - Fixed Region, Telegram Debug If Captcha Detected, OCR-only Captcha Type Detection,
Auto-configure Tesseract (Win/macOS/Linux), Full CAPTCHA Types (slide, rotate_app, object).
"""

import os
import time
import math
import io
import base64
import platform

import numpy as np
import cv2
import mss
import pyautogui as pag
from PIL import Image
import requests

# ================= CONFIG =================
HM_API_KEY = os.getenv("HMCAPTCHA_KEY", "giangdev-KmpjQPCO2kzrCqtoAud1TiSi4OsxVfxN")
HM_BASE = "https://hmcaptcha.com"

API_ENDPOINTS = {
    "create_task": "/Recognition?wait=1"
}

CAPTCHA_TYPES = {
    "slide": "ALL_CAPTCHA_SLIDE",
    "rotate_app": "TIKTOK_ROTATE_APP",
    "object": "TIKTOK_OBJ"
}

FIXED_CAPTCHA_REGION = {
    "enabled": True,
    "x_percent": 0.3,
    "y_percent": 0.15,
    "width_percent": 0.4,
    "height_percent": 0.7
}

SCAN_INTERVAL = 10  # seconds

# Telegram config for sending screenshots
TELEGRAM_BOT_TOKEN = "7431244567:AAEqvYa58hSXbfJaQ4ZiIv3MR1SHHoVmMmA"
TELEGRAM_CHAT_ID = "1934845201"

pag.FAILSAFE = False
pag.PAUSE = 0.1

# ================= TESSERACT CONFIG =================
try:
    import pytesseract

    def configure_tesseract():
        system = platform.system()

        if system == "Windows":
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
                    print("⚠️ Tesseract not found on macOS. Install with: brew install tesseract")
                    return False
            except Exception:
                print("⚠️ Could not locate tesseract on macOS")
                return False

        else:  # Linux
            print("✅ Using system tesseract (Linux)")
            return True

    TESSERACT_AVAILABLE = configure_tesseract()
except ImportError:
    TESSERACT_AVAILABLE = False

# ================= UTILITIES =================

def get_monitor_info():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        return {
            "width": monitor["width"],
            "height": monitor["height"],
            "left": monitor["left"],
            "top": monitor["top"]
        }

def get_fixed_region_coords():
    monitor_info = get_monitor_info()
    x = int(monitor_info["width"] * FIXED_CAPTCHA_REGION["x_percent"])
    y = int(monitor_info["height"] * FIXED_CAPTCHA_REGION["y_percent"])
    width = int(monitor_info["width"] * FIXED_CAPTCHA_REGION["width_percent"])
    height = int(monitor_info["height"] * FIXED_CAPTCHA_REGION["height_percent"])
    return (x, y, width, height)

def screenshot_fixed_region():
    x, y, width, height = get_fixed_region_coords()
    region = {"left": x, "top": y, "width": width, "height": height}
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        # Khắc phục retina (nếu có), resize về kích thước logic pixel
        if img.width == width * 2 and img.height == height * 2:
            img = img.resize((width, height), Image.LANCZOS)
        return img, (x, y, width, height)

def pil_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def pil_to_b64(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def send_telegram_image(pil_img, caption="Debug"):
    try:
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'caption': caption
        }
        files = {
            'photo': ('captcha_debug.png', buffer, 'image/png')
        }
        response = requests.post(telegram_url, files=files, data=data, timeout=10)
        if response.status_code == 200:
            print("📸 Đã gửi ảnh debug qua Telegram.")
        else:
            print(f"❌ Lỗi gửi Telegram: {response.status_code}")
    except Exception as e:
        print(f"❌ Lỗi gửi ảnh Telegram: {e}")

# ================= CAPTCHA TYPE DETECTION =================

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

# ================= API =================

def post_json(captcha_type, image_b64):
    hm_type = CAPTCHA_TYPES.get(captcha_type)
    if not hm_type:
        return {"error": f"Unsupported captcha type: {captcha_type}"}
    create_url = f"{HM_BASE}{API_ENDPOINTS['create_task']}"
    payload = {
        "Type": hm_type,
        "Image": image_b64,
        "Apikey": HM_API_KEY
    }
    try:
        response = requests.post(create_url, json=payload, timeout=30)
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ================= MOUSE AUTOMATION =================

def safe_mouse_move(x, y, duration=0.3):
    try:
        pag.moveTo(x, y, duration=duration)
        return True
    except Exception:
        return False

def safe_mouse_drag(start_x, start_y, end_x, end_y, duration=1.0):
    try:
        safe_mouse_move(start_x, start_y, 0.2)
        pag.mouseDown(start_x, start_y, button='left')
        pag.moveTo(end_x, end_y, duration=duration)
        pag.mouseUp(end_x, end_y, button='left')
        return True
    except Exception:
        return False

def click_points_abs(points):
    success = True
    for i, (x, y) in enumerate(points):
        print(f"🖱️ Click điểm {i+1}/{len(points)}: ({x}, {y})")
        if not safe_mouse_move(x, y, 0.3):
            success = False
            break
        pag.click(x, y, duration=0.2)
        time.sleep(0.4)
    if success:
        print(f"✅ Hoàn thành click {len(points)} điểm")
    return success

# ================= SOLVER =================

def solve_slide_captcha(region_coords, api_data):
    x, y, w, h = region_coords
    x_api = api_data.get("x", 0)
    y_api = api_data.get("y", 0)
    offset = api_data.get("offset", 0)
    start_x = x + x_api
    start_y = y + y_api
    end_x = start_x + offset
    end_y = start_y
    print(f"🖱️ Slide: kéo từ ({start_x},{start_y}) đến ({end_x},{end_y})")
    time.sleep(0.5)
    safe_mouse_drag(start_x, start_y, end_x, end_y, duration=1.0)

def solve_rotate_captcha(region_coords, api_data):
    x, y, w, h = region_coords
    angle = api_data.get("angle", 0)
    cx = x + w // 2
    cy = y + h // 2
    radius = int(min(w, h) * 0.3)
    start_x = cx + radius
    start_y = cy
    angle_rad = math.radians(angle)
    end_x = int(cx + radius * math.cos(angle_rad))
    end_y = int(cy + radius * math.sin(angle_rad))
    print(f"🖱️ Rotate: kéo vòng tròn từ ({start_x},{start_y}) đến ({end_x},{end_y}) (góc {angle}°)")
    time.sleep(0.5)
    safe_mouse_drag(start_x, start_y, end_x, end_y, duration=1.5)

def solve_object_captcha(region_coords, api_data):
    region_x, region_y, region_w, region_h = region_coords
    raw_coords = api_data.get("raw", "")
    print(f"🎯 Object DEBUG:")
    print(f"   📊 Raw coords: '{raw_coords}'")
    print(f"   🖼️ Region: ({region_x},{region_y}) size={region_w}x{region_h}")
    clicks = []
    if raw_coords:
        for coord_pair in raw_coords.split('|'):
            if ',' in coord_pair:
                x_str, y_str = coord_pair.split(',')
                x_api, y_api = float(x_str), float(y_str)
                if x_api <= 1.0 and y_api <= 1.0:
                    x_screen = region_x + int(x_api * region_w)
                    y_screen = region_y + int(y_api * region_h)
                    print(f"   👆 Object (ratio): API({x_api:.2f},{y_api:.2f}) → ({x_screen},{y_screen})")
                else:
                    x_screen = region_x + int(x_api)
                    y_screen = region_y + int(y_api)
                    print(f"   👆 Object (abs): API({x_api:.0f},{y_api:.0f}) + Region({region_x},{region_y}) = ({x_screen},{y_screen})")
                clicks.append((x_screen, y_screen))
    if clicks:
        print(f"   🖱️ Clicking {len(clicks)} points: {clicks}")
        time.sleep(0.5)
        click_points_abs(clicks)
        print(f"✅ Đã thao tác object selection.")
        return True
    print("❌ Không có toạ độ object để click.")
    return False

# ================= MAIN =================

def main():
    while True:
        print("🟢 Quét captcha...")

        # In ra kích thước màn hình
        monitor_info = get_monitor_info()
        print(f"🖥️ Kích thước màn hình: {monitor_info['width']}x{monitor_info['height']}")

        region_img, region_coords = screenshot_fixed_region()

        # In ra kích thước ảnh vùng chụp và toạ độ điểm bắt đầu
        region_x, region_y, region_w, region_h = region_coords
        print(f"📸 Kích thước vùng chụp: {region_img.width}x{region_img.height} tại ({region_x},{region_y})")
        # Nếu vùng chụp quá lớn, resize nhỏ lại trước khi gửi API

        captcha_type = detect_captcha_type_by_text(region_img)
        print(f"Nhận diện captcha: {captcha_type}")

        if captcha_type not in CAPTCHA_TYPES:
            print("⏭️ Không nhận diện được loại captcha, bỏ qua.")
            time.sleep(SCAN_INTERVAL)
            continue

        # CHỈ gửi ảnh debug nếu có captcha
        send_telegram_image(region_img, caption=f"Captcha ({captcha_type}) debug")

        image_b64 = pil_to_b64(region_img)
        res = post_json(captcha_type, image_b64)
        if res.get("Status") == "SUCCESS" and "Data" in res:
            data = res["Data"]
            if captcha_type == "slide":
                solve_slide_captcha(region_coords, data)
            elif captcha_type == "rotate_app":
                solve_rotate_captcha(region_coords, data)
            elif captcha_type == "object":
                solve_object_captcha(region_coords, data)
        else:
            print("❌ Không giải được captcha hoặc API lỗi.")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
