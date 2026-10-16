"""Hand-made features of the component classifier (random forest) - copied unchanged from
درخت/درخت/final_33.ipynb (the notebook that reached 92.6 % on 1195 crops), plus a small wrapper.

extract_features_advanced() accepts a file path OR a BGR image (a crop cut from a board photo).
"""
import os
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")   # tiny crops give empty regions -> 'Mean of empty slice' etc. (harmless, value 0)
np.seterr(all="ignore")

IMAGE_SIZE = 256


def resize_with_aspect_ratio(image, size=IMAGE_SIZE):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh)//2
    x0 = (size - nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

COLOR_RANGES = {
    'yellow': ((20, 100, 100), (35, 255, 255)),
    'cream':  ((15, 20, 180),  (40, 120, 255)),
    'brown':  ((5, 80, 20),    (20, 255, 180)),
    'orange': ((5, 120, 120),  (30, 255, 255)),
    'black':  ((0, 0, 0),      (180, 255, 50)),
    'gray':   ((0, 0, 50),     (180, 40, 200)),
    'silver': ((0, 0, 200),    (180, 50, 255)),
    'green':  ((40, 50, 50),   (80, 255, 255)),
    'red1':   ((0, 100, 100),  (10, 255, 255)),
    'red2':   ((170, 100, 100),(180, 255, 255)),
}

# قبل از توابع، COLOR_MAP را آپدیت کن:
COLOR_MAP = {
    'black': 0, 'brown': 1, 'cream': 2, 'gray': 3, 'silver': 4,
    'yellow': 5, 'orange': 6, 'green': 7, 'red': 8,
    # اضافه کردن رنگ‌های جدید
    'blue': 9, 'white': 10, 'unknown': -1
}



def classify_color(image):
    """
    تشخیص رنگ با دقت بالاتر - بهبود یافته
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # توسعه محدوده رنگ‌ها برای دقت بیشتر
    EXPANDED_COLOR_RANGES = {
        'yellow': [
            ((20, 100, 100), (35, 255, 255)),  # زرد استاندارد
            ((15, 150, 150), (30, 255, 255)),  # زرد روشن
            ((25, 80, 120), (40, 255, 255)),   # زرد متمایل به نارنجی
        ],
        'cream': [
            ((15, 20, 180), (25, 100, 255)),   # کرم روشن
            ((20, 10, 190), (30, 80, 255)),    # کرم
            ((10, 15, 170), (35, 90, 240)),    # کرم تیره
        ],
        'brown': [
            ((5, 80, 20), (20, 255, 180)),     # قهوه‌ای روشن
            ((0, 100, 30), (20, 255, 150)),    # قهوه‌ای
            ((5, 120, 40), (25, 255, 120)),    # قهوه‌ای تیره
        ],
        'orange': [
            ((5, 120, 120), (25, 255, 255)),   # نارنجی پررنگ
            ((10, 100, 150), (30, 255, 255)),  # نارنجی
            ((0, 100, 130), (20, 255, 230)),   # نارنجی روشن
        ],
        'black': [
            ((0, 0, 0), (180, 255, 50)),       # سیاه استاندارد
            ((0, 0, 0), (180, 100, 30)),       # سیاه خالص
            ((0, 0, 0), (180, 150, 40)),       # سیاه متمایل به خاکستری
        ],
        'gray': [
            ((0, 0, 50), (180, 50, 200)),      # خاکستری روشن
            ((0, 0, 30), (180, 30, 150)),      # خاکستری
            ((0, 0, 70), (180, 40, 180)),      # خاکستری تیره
        ],
        'silver': [
            ((0, 0, 180), (180, 50, 255)),     # نقره‌ای
            ((0, 0, 150), (180, 30, 230)),     # نقره‌ای روشن
            ((0, 0, 200), (180, 40, 255)),     # نقره‌ای درخشان
        ],
        'green': [
            ((40, 50, 50), (80, 255, 255)),    # سبز الکترونیکی
            ((36, 25, 25), (70, 255, 200)),    # سبز روشن
            ((45, 100, 50), (85, 255, 180)),   # سبز تیره
        ],
        'red': [  # ترکیب دو محدوده قرمز
            ((0, 100, 100), (10, 255, 255)),   # قرمز روشن
            ((170, 100, 100), (180, 255, 255)), # قرمز تیره
        ],
        'blue': [  # اضافه کردن آبی برای قطعات الکترونیکی
            ((90, 50, 50), (130, 255, 255)),   # آبی
            ((100, 60, 60), (140, 255, 200)),  # آبی روشن
        ],
        'white': [  # اضافه کردن سفید
            ((0, 0, 200), (180, 30, 255)),     # سفید
        ]
    }
    
    # مناطق نمونه‌گیری استراتژیک
    height, width = IMAGE_SIZE, IMAGE_SIZE
    sample_regions = [
        # مرکز (مهمترین منطقه)
        hsv[height//2-30:height//2+30, width//2-30:width//2+30],
        # چهار گوشه
        hsv[:height//3, :width//3],
        hsv[:height//3, 2*width//3:],
        hsv[2*height//3:, :width//3],
        hsv[2*height//3:, 2*width//3:],
        # لبه‌ها (برای تشخیص الکترود)
        hsv[:, :width//6],
        hsv[:, 5*width//6:],
    ]
    
    region_weights = [1.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7]  # وزن مناطق
    
    color_votes = {}
    color_scores = {}
    
    for idx, region in enumerate(sample_regions):
        if region.size == 0:
            continue
            
        weight = region_weights[idx]
        
        # نمونه‌گیری تصادفی از پیکسل‌ها در منطقه
        if region.size > 100:
            flat_region = region.reshape(-1, 3)
            num_samples = min(50, len(flat_region))
            indices = np.random.choice(len(flat_region), num_samples, replace=False)
            samples = flat_region[indices]
        else:
            samples = region.reshape(-1, 3)
        
        for h, s, v in samples:
            # فیلتر پیکسل‌های غیرمعمول
            if v < 20 or v > 240:  # خیلی تیره یا خیلی روشن
                continue
                
            # تشخیص رنگ بر اساس HSV
            matched = False
            for color_name, ranges in EXPANDED_COLOR_RANGES.items():
                for (lower, upper) in ranges:
                    if (lower[0] <= h <= upper[0]) and \
                       (lower[1] <= s <= upper[1]) and \
                       (lower[2] <= v <= upper[2]):
                        
                        # محاسبه امتیاز بر اساس فاصله از مرکز محدوده
                        h_center = (lower[0] + upper[0]) / 2
                        s_center = (lower[1] + upper[1]) / 2
                        v_center = (lower[2] + upper[2]) / 2
                        
                        h_distance = 1 - min(abs(h - h_center) / 90, 1)
                        s_distance = 1 - min(abs(s - s_center) / 255, 1)
                        v_distance = 1 - min(abs(v - v_center) / 255, 1)
                        
                        score = (h_distance * 0.4 + s_distance * 0.3 + v_distance * 0.3) * weight
                        
                        # رأی برای رنگ
                        color_votes[color_name] = color_votes.get(color_name, 0) + 1
                        color_scores[color_name] = color_scores.get(color_name, 0) + score
                        matched = True
                        break
                if matched:
                    break
            
            # اگر رنگی پیدا نشد و اشباع کم است، احتمالاً خاکستری/سفید/سیاه
            if not matched and s < 40:
                if v > 200:
                    color_name = 'white'
                elif v > 100:
                    color_name = 'gray'
                else:
                    color_name = 'black'
                
                color_votes[color_name] = color_votes.get(color_name, 0) + 1
                color_scores[color_name] = color_scores.get(color_name, 0) + weight
    
    # اگر هیچ رنگی پیدا نشد
    if not color_scores:
        # تحلیل بر اساس میانگین HSV
        mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        h, s, v = mean_hsv
        
        if s < 30:
            if v > 180:
                return 'white'
            elif v > 80:
                return 'gray'
            else:
                return 'black'
        elif 15 <= h <= 35:
            return 'yellow'
        elif 5 <= h <= 25:
            if s > 80 and v > 120:
                return 'orange'
            else:
                return 'brown'
        elif 36 <= h <= 85:
            return 'green'
        elif 90 <= h <= 130:
            return 'blue'
        elif h <= 10 or h >= 170:
            return 'red'
        else:
            return 'unknown'
    
    # ترکیب رأی و امتیاز
    final_scores = {}
    for color in set(color_votes.keys()) | set(color_scores.keys()):
        votes = color_votes.get(color, 0)
        score = color_scores.get(color, 0)
        # میانگین وزنی از رأی و امتیاز
        final_scores[color] = (votes * 0.4) + (score * 0.6)
    
    # انتخاب رنگ برنده
    best_color = max(final_scores, key=final_scores.get)
    
    # اگر اختلاف امتیاز کم است، دقت بیشتری انجام دهیم
    sorted_colors = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_colors) > 1:
        first_score = sorted_colors[0][1]
        second_score = sorted_colors[1][1]
        
        # اگر اختلاف کمتر از 10٪ است، تحلیل بیشتری انجام دهیم
        if first_score > 0 and (first_score - second_score) / first_score < 0.1:
            # تحلیل فضای LAB برای دقت بیشتر
            lab_center = lab[height//2-20:height//2+20, width//2-20:width//2+20]
            if lab_center.size > 0:
                L_mean = np.mean(lab_center[:, :, 0])
                a_mean = np.mean(lab_center[:, :, 1])
                b_mean = np.mean(lab_center[:, :, 2])
                
                # تشخیص سفید/خاکستری بر اساس کانال L
                if L_mean > 190:
                    best_color = 'white'
                elif L_mean > 120:
                    # بین gray و silver
                    if abs(a_mean) < 10 and abs(b_mean) < 10:
                        best_color = 'gray'
                    else:
                        best_color = 'silver'
    
    # تطبیق با رنگ‌های موجود در COLOR_MAP
    if best_color not in COLOR_MAP and best_color != 'unknown':
        # نگاشت به نزدیکترین رنگ موجود
        if best_color in ['white', 'blue']:
            best_color = 'unknown'
    
    return best_color


def detect_main_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    gray_blur = cv2.GaussianBlur(gray, (9,9), 2)
    
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=int(0.6*min(h,w)),
        param1=100,
        param2=35,
        minRadius=int(0.2*min(h,w)),
        maxRadius=int(0.48*min(h,w))
    )
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circle_found = 0
    if circles is not None:
        circles = np.uint16(np.around(circles.reshape(-1, 3)))
        x, y, r = max(circles, key=lambda c: c[2])
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        filled_ratio = cv2.countNonZero(mask) / (np.pi*r*r)
        
        if filled_ratio > 0.5 and r > min(h, w) * 0.15:
            circle_found = 1
    
    if not circle_found and contours:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (h * w * 0.1):
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.85:
                    circle_found = 1
                    break
    
    return circle_found


def count_legs_advanced(image):
    """
    شمارش پایه با تحلیل پروفایل روشنایی در چهار نوار کناری.
    خروجی: یک عدد صحیح (تعداد کل پایه‌ها).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    strip_w = int(W * 0.12)
    strip_h = int(H * 0.12)

    left_strip  = gray[:, :strip_w]
    right_strip = gray[:, W-strip_w:]
    top_strip   = gray[:strip_h, :]
    bottom_strip= gray[H-strip_h:, :]

    def count_legs_in_strip(strip, axis):
        sh, sw = strip.shape
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        strip_eq = clahe.apply(strip)
        blurred = cv2.GaussianBlur(strip_eq, (5, 5), 0)

        if axis == 0:
            profile = np.mean(blurred, axis=1).astype(np.float32)
        else:
            profile = np.mean(blurred, axis=0).astype(np.float32)

        kernel_size = max(3, len(profile) // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1
        profile_smooth = cv2.GaussianBlur(
            profile.reshape(-1,1), (1, kernel_size), 0
        ).flatten()

        mean_val = np.mean(profile_smooth)
        binary = (profile_smooth > mean_val).astype(np.uint8)

        transitions = np.diff(binary.astype(np.int8))
        starts = np.where(transitions == 1)[0]
        ends   = np.where(transitions == -1)[0]

        if len(starts) == 0 or len(ends) == 0:
            return 0

        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:len(ends)]

        min_width = len(profile) * 0.01
        max_width = len(profile) * 0.25
        std_val = np.std(profile_smooth)
        min_amplitude = std_val * 0.3

        valid_legs = []
        for s, e in zip(starts, ends):
            width = e - s
            block_mean = np.mean(profile_smooth[s:e+1])
            amplitude = block_mean - mean_val
            if min_width < width < max_width and amplitude > min_amplitude:
                valid_legs.append((s, e))

        if len(valid_legs) > 1:
            centers = [(s + e) / 2 for s, e in valid_legs]
            min_distance = len(profile) * 0.02
            filtered_legs = [valid_legs[0]]
            for i in range(1, len(valid_legs)):
                if centers[i] - centers[i-1] >= min_distance:
                    filtered_legs.append(valid_legs[i])
            valid_legs = filtered_legs

        return len(valid_legs)

    left_count   = count_legs_in_strip(left_strip, 0)
    right_count  = count_legs_in_strip(right_strip, 0)
    top_count    = count_legs_in_strip(top_strip, 1)
    bottom_count = count_legs_in_strip(bottom_strip, 1)

    # تقارن
    if left_count == 0: right_count = 0
    if right_count == 0: left_count = 0
    if top_count == 0: bottom_count = 0
    if bottom_count == 0: top_count = 0

    return left_count + right_count + top_count + bottom_count



def detect_electrodes(image):
    """
    تشخیص الکترود در چهار طرف با امتیازدهی بر اساس:
    - اختلاف روشنایی نسبت به میانگین کل
    - تراکم لبه
    - خطوط عمودی/افقی
    - یکنواختی (انحراف معیار کم)
    خروجی: 1 اگر در دو طرف مقابل الکترود تشخیص داده شود، در غیر این صورت 0.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    margin_ratio = 0.30
    left_margin   = int(W * margin_ratio)
    right_margin  = int(W * margin_ratio)
    top_margin    = int(H * margin_ratio)
    bottom_margin = int(H * margin_ratio)

    regions = {
        'left':   gray[:, :left_margin],
        'right':  gray[:, W-right_margin:],
        'top':    gray[:top_margin, :],
        'bottom': gray[H-bottom_margin:, :]
    }

    global_mean = np.mean(gray)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                            minLineLength=15, maxLineGap=5)

    def evaluate_side(region, side):
        mean_val = np.mean(region)
        diff = mean_val - global_mean

        # Edge density
        if side == 'left':
            edge_region = edges[:, :left_margin]
        elif side == 'right':
            edge_region = edges[:, W-right_margin:]
        elif side == 'top':
            edge_region = edges[:top_margin, :]
        else:  # bottom
            edge_region = edges[H-bottom_margin:, :]
        edge_density = np.sum(edge_region > 0) / edge_region.size

        # Vertical/horizontal lines
        v_lines = 0
        h_lines = 0
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                if abs(x2 - x1) < 5 and abs(y2 - y1) > 10:  # vertical line
                    if side == 'left' and x1 < left_margin:
                        v_lines += 1
                    elif side == 'right' and x1 > W - right_margin:
                        v_lines += 1
                    elif side == 'top' and y1 < top_margin:
                        h_lines += 1
                    elif side == 'bottom' and y1 > H - bottom_margin:
                        h_lines += 1

        std_val = np.std(region)

        # Scoring
        score = 0
        if diff > 5: score += 1
        if edge_density > 0.05: score += 1
        if (side in ('left','right') and v_lines >= 1) or \
           (side in ('top','bottom') and h_lines >= 1):
            score += 1
        if std_val < 30: score += 1

        return score >= 3   # الکترود محسوب می‌شود

    left_ok   = evaluate_side(regions['left'], 'left')
    right_ok  = evaluate_side(regions['right'], 'right')
    top_ok    = evaluate_side(regions['top'], 'top')
    bottom_ok = evaluate_side(regions['bottom'], 'bottom')

    # شرط نهایی
    return 1 if (left_ok and right_ok) or (top_ok and bottom_ok) else 0   

def detect_written_advanced(gray):
    """
    تشخیص بلوک‌های متن (نه کاراکترهای جداگانه).
    خروجی:
        - تعداد بلوک‌های معتبر (int)
        - مجموع پیکسل‌های متن (int)
        - وجود متن (1 اگر حداقل یک بلوک باشد) (int)
        - چگالی متن (نسبت پیکسل‌های متن به کل تصویر) (float)
    """
    height, width = gray.shape

    # 1. بهبود کنتراست
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. حذف نویز
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10,
                                        templateWindowSize=7, searchWindowSize=21)

    # 3. آستانه‌گذاری‌های مختلف
    adaptive1 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    adaptive2 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY_INV, 15, 3)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. ترکیب و مورفولوژی
    combined = cv2.bitwise_or(adaptive1, adaptive2)
    combined = cv2.bitwise_or(combined, otsu)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    morphed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. آنالیز اجزای متصل
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morphed, connectivity=8)

    MIN_AREA = height * width * 0.01   # حداقل 1٪ تصویر
    MAX_AREA = height * width * 0.8    # حداکثر 80٪ تصویر
    total_text_pixels = 0
    valid_blocks = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_AREA <= area <= MAX_AREA:
            valid_blocks += 1
            total_text_pixels += area

    has_text = 1 if valid_blocks > 0 else 0
    text_density = total_text_pixels / (height * width)

    return valid_blocks, total_text_pixels, has_text, text_density


def extract_shape_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, 0, 0, 1.0
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    aspect_ratio = w / h if h > 0 else 0
    
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    return circularity, aspect_ratio, solidity, area



def infer_category(filename):
    f = filename.lower()
    if 'resistor' in f: return 'resistors'
    if 'capacitor' in f: return 'capacitors'
    if 'diode' in f: return 'diodes'
    if 'inductor' in f: return 'inductors'
    if 'transistor' in f: return 'transistors'
    if 'ic' in f or 'integrated' in f: return 'ICs'
    return 'unknown'

    
def extract_features_advanced(path):

    img = cv2.imread(path) if isinstance(path, str) else path
    if img is None:
        return None
    
    img = resize_with_aspect_ratio(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # ویژگی‌های اصلی با توابع بهبود یافته
    color = classify_color(img)  # تابع بهبود یافته
    is_circle = detect_main_circle(img)
    electrodes = detect_electrodes(img)
    pin_count = count_legs_advanced(img)
    three_legs = 1 if pin_count >= 3 else 0
    
    # تشخیص متن با تابع بهبود یافته
    chars, area, has_text, text_density = detect_written_advanced(gray)
    
    # ویژگی‌های شکل
    circularity, aspect_ratio, solidity, contour_area = extract_shape_features(img)
    
    # ویژگی‌های شدت نور
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # ویژگی‌های جدید برای بهبود دقت
    # 1. تحلیل گرادیان (لبه‌ها)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_mean = np.mean(gradient_magnitude)
    gradient_std = np.std(gradient_magnitude)
    
    # 2. تحلیل بافت ساده (واریانس محلی)
    from scipy.ndimage import uniform_filter
    gray_float = gray.astype(float)
    local_mean = uniform_filter(gray_float, size=5)
    local_sq_mean = uniform_filter(gray_float**2, size=5)
    local_variance = local_sq_mean - local_mean**2
    texture_mean = np.mean(local_variance)
    
    # 3. ویژگی‌های رنگ RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r_mean, g_mean, b_mean = np.mean(rgb_img, axis=(0, 1))
    r_std, g_std, b_std = np.std(rgb_img, axis=(0, 1))
    
    # 4. نسبت رنگ‌ها
    sum_rgb = r_mean + g_mean + b_mean + 1e-10
    color_ratio_r = r_mean / sum_rgb
    color_ratio_g = g_mean / sum_rgb
    color_ratio_b = b_mean / sum_rgb
    
    # 5. ویژگی‌های هیستوگرام
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    hist_mean = np.mean(hist)
    hist_std = np.std(hist)
    
    # 6. تراکم لبه‌ها
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (height * width)
    
    # 7. ویژگی‌های موقعیت متن
    text_in_center = 1 if chars > 0 and text_density > 0.005 else 0
    
    # 8. تنوع رنگ در مرکز
    center_region = img[height//4:3*height//4, width//4:3*width//4]
    if center_region.size > 0:
        center_hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        center_saturation = np.mean(center_hsv[:, :, 1])
        center_value = np.mean(center_hsv[:, :, 2])
    else:
        center_saturation = 0
        center_value = 0
    
    return {
        'filename': os.path.basename(path) if isinstance(path, str) else 'crop',
        'Category': infer_category(path) if isinstance(path, str) else 'unknown',
        'color_name': color,
        'color_code': COLOR_MAP.get(color, -1),
        'is_circle': is_circle,
        'electrodes_both_ends': electrodes,
        'pin_count': pin_count,
        'three_legs': three_legs,
        'written_chars': chars,
        'written_area_sum': area,
        'has_text': has_text,
        'text_density': text_density,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'contour_area': contour_area,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        
        # ویژگی‌های جدید (اضافه شده)
        'gradient_mean': float(gradient_mean),
        'gradient_std': float(gradient_std),
        'texture_mean': float(texture_mean),
        'r_mean': float(r_mean),
        'g_mean': float(g_mean),
        'b_mean': float(b_mean),
        'r_std': float(r_std),
        'g_std': float(g_std),
        'b_std': float(b_std),
        'color_ratio_r': float(color_ratio_r),
        'color_ratio_g': float(color_ratio_g),
        'color_ratio_b': float(color_ratio_b),
        'hist_mean': float(hist_mean),
        'hist_std': float(hist_std),
        'edge_density': float(edge_density),
        'text_in_center': text_in_center,
        'center_saturation': float(center_saturation),
        'center_value': float(center_value),
    }


# ------------------------------------------------------------------ wrapper used by the test notebook
FEATURE_COLS = [
    'color_code', 'is_circle', 'electrodes_both_ends', 'three_legs',
    'pin_count', 'written_chars', 'written_area_sum', 'has_text',
    'text_density', 'circularity', 'aspect_ratio', 'solidity',
    'contour_area', 'mean_intensity', 'std_intensity',
    'gradient_mean', 'gradient_std', 'texture_mean',
    'r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std',
    'color_ratio_r', 'color_ratio_g', 'color_ratio_b',
    'hist_mean', 'hist_std', 'edge_density',
    'text_in_center', 'center_saturation', 'center_value',
]
# tree class -> name used by the YOLO 'fine' model
TO_YOLO = {'resistors': 'Resistor', 'capacitors': 'Capacitor', 'ICs': 'IC', 'diodes': 'Diode',
           'inductors': 'Inductor', 'transistors': 'Transistor'}


def features_row(img):
    """Feature vector of one BGR crop (same order as FEATURE_COLS). classify_color samples pixels at random:
    the seed makes the answer repeatable."""
    np.random.seed(0)
    f = extract_features_advanced(img)
    return [float(f[c]) for c in FEATURE_COLS]


def features_many(crops, workers=None):
    """Features of many crops in parallel threads (OpenCV releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor
    if not crops:
        return np.zeros((0, len(FEATURE_COLS)))
    with ThreadPoolExecutor(workers or min(8, os.cpu_count() or 2)) as ex:
        return np.array(list(ex.map(features_row, crops)))


def train_forest(csv_path):
    """Same model as final_33.ipynb: RandomForest(100 trees, depth 10), trained on the saved feature table."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv(csv_path)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(df[FEATURE_COLS].fillna(0).values, df['Category'].values)
    return rf


def load_forest(joblib_path, csv_path=None):
    """Load the saved forest; if this scikit-learn cannot read it, rebuild it from the feature table (a few seconds)."""
    import warnings
    try:
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rf = joblib.load(joblib_path)
        rf.predict(np.zeros((1, len(FEATURE_COLS))))
        return rf, 'loaded'
    except Exception as e:
        if csv_path and os.path.exists(csv_path):
            return train_forest(csv_path), f'rebuilt from {os.path.basename(csv_path)} ({type(e).__name__})'
        raise


def classify(rf, crops, workers=None):
    """crops: list of BGR images -> list of (tree_class, probability)."""
    X = features_many(crops, workers)
    if not len(X):
        return []
    P = rf.predict_proba(X)
    return [(rf.classes_[i], float(p[i])) for p, i in zip(P, P.argmax(1))]
