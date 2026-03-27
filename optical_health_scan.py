import cv2
import numpy as np
import time
import os
import tempfile

# --- DEPENDENCY GUARD ---
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

# --- CONFIGURATION (Legacy Fallback) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# --- NEURAL NETWORK INITIALIZATION (If Available) ---
if HAS_MEDIAPIPE:
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh_instance = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
    except (AttributeError, RuntimeError) as e:
        print(f"MediaPipe initialization failed: {e}")
        HAS_MEDIAPIPE = False
        mp_face_mesh = None
        face_mesh_instance = None

# ── SHARED UTILITIES ───────────────────────────────────────────

def process_eye_diagnostics(img):
    """Deep analysis of eye crop for clarity and color."""
    if img is None or img.size == 0:
        return 0.0, 1.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clarity_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Sclera-Based Aging Index
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    lens_bgr = cv2.mean(img)
    
    if np.sum(thresh) > 100:
        sclera_bgr = cv2.mean(img, mask=thresh)
        w_ratio = (sclera_bgr[2] + sclera_bgr[1]) / (2 * sclera_bgr[0] + 1e-6)
        l_ratio = (lens_bgr[2] + lens_bgr[1]) / (2 * lens_bgr[0] + 1e-6)
        aging_index = l_ratio / (w_ratio + 1e-6)
    else:
        aging_index = (lens_bgr[2] + lens_bgr[1]) / (2 * lens_bgr[0] + 1e-6)
        
    return float(clarity_val), float(aging_index)

def get_tight_eye_crop(frame, lms=None, roi_target=None):
    """Isolates only the eye area for results (Heatmap)."""
    h, w = frame.shape[:2]
    
    # 1. Neural Tighter Crop (MediaPipe)
    if lms:
        eye_indices = [33, 133, 157, 158, 159, 160, 161, 246]
        pts = []
        for idx in eye_indices:
            lm = lms.landmark[idx]
            pts.append((int(lm.x * w), int(lm.y * h)))
        pts = np.array(pts)
        ex, ey, ew, eh = cv2.boundingRect(pts)
        
        # Add 25% padding for better visual context
        pad_w = int(ew * 0.25)
        pad_h = int(eh * 0.25)
        ex = max(0, ex - pad_w)
        ey = max(0, ey - pad_h)
        ew = min(w - ex, ew + pad_w * 2)
        eh = min(h - ey, eh + pad_h * 2)
        return frame[ey:ey+eh, ex:ex+ew]
    
    # 2. Manual Fallback Tighter Crop (Cascade)
    elif roi_target:
        rx, ry, rw, rh = roi_target
        roi_img = frame[ry:ry+rh, rx:rx+rw]
        eyes = eye_cascade.detectMultiScale(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY), 1.1, 5)
        if len(eyes) > 0:
            ex, ey, ew, eh = eyes[0]
            # Center the crop and add padding
            pad = 15
            ex2 = max(0, ex - pad)
            ey2 = max(0, ey - pad)
            ew2 = min(rw - ex2, ew + pad * 2)
            eh2 = min(rh - ey2, eh + pad * 2)
            return roi_img[ey2:ey2+eh2, ex2:ex2+ew2]
            
    # Total Fallback: Return 180x180 central
    tx, ty, tw, th = roi_target if roi_target else (w//2-90, h//2-90, 180, 180)
    return frame[ty:ty+th, tx:tx+tw]

# ── MAIN SYSTEM DRIVER ─────────────────────────────────────────

def run_unified_scanner():
    """Streamlined Precision Neural Eye Scanner (Eye-Only Metrics)."""
    cap = cv2.VideoCapture(0)
    window_name = 'Nayana Precision Ocular Diagnostic'
    cv2.namedWindow(window_name)

    mode_label = "NEURAL (MP)" if HAS_MEDIAPIPE else "ROBUST (CONT)"
    stage = "ALIGNMENT" 
    alignment_timer = 0
    test_frames = []
    results = None
    
    ci, fs, yi = 0, 0, 0
    heatmap_path = None
    last_lms = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        raw_frame = cv2.flip(frame, 1)
        display = raw_frame.copy() 
        h, w = raw_frame.shape[:2]
        
        roi_target = (w//2-90, h//2-90, 180, 180)
        roi_color = (130, 130, 130)
        detected = False
        key = cv2.waitKey(1) & 0xFF

        # --- NEURAL / LOGIC PROCESSING ---
        if HAS_MEDIAPIPE:
            rgb_f = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            m_res = face_mesh_instance.process(rgb_f)
            if m_res.multi_face_landmarks:
                last_lms = m_res.multi_face_landmarks[0]
                cx, cy = int(last_lms.landmark[468].x * w), int(last_lms.landmark[468].y * h)
                cv2.circle(display, (cx, cy), 2, (0, 255, 0), -1)
                if (roi_target[0] < cx < roi_target[0]+roi_target[2]) and (roi_target[1] < cy < roi_target[1]+roi_target[3]):
                    detected = True
        else:
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(display, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 1)
                fcx, fcy = fx + fw//2, fy + fh//3
                if (roi_target[0] < fcx < roi_target[0]+roi_target[2]) and (roi_target[1] < fcy < roi_target[1]+roi_target[3]):
                    detected = True
            
        # --- STAGE ROUTING ---
        if stage == "ALIGNMENT":
            if key == ord(' '): detected = True; alignment_timer = 50
            if detected:
                roi_color = (0, 255, 0)
                alignment_timer += 1
                progress = min(alignment_timer / 40.0, 1.0)
                cv2.rectangle(display, (roi_target[0], roi_target[1]+roi_target[3]+5), (roi_target[0] + int(roi_target[2]*progress), roi_target[1]+roi_target[3]+15), (0, 255, 0), -1)
                if alignment_timer >= 40:
                    stage = "LENS_SCAN"
                    alignment_timer = 0
            else:
                alignment_timer = max(0, alignment_timer - 1)
                cv2.putText(display, "ALIGN EYE (OR SPACEBAR)", (roi_target[0], roi_target[1]-10), 1, 1, (255, 255, 255), 1)

        elif stage == "LENS_SCAN":
            cv2.putText(display, "🚀 SCANNING EYE...", (roi_target[0], roi_target[1]-10), 1, 1.2, (0, 255, 0), 2)
            roi_color = (0, 255, 0)
            
            # Use tighter precision crop
            tight_eye = get_tight_eye_crop(raw_frame, last_lms if HAS_MEDIAPIPE else None, roi_target)
            test_frames.append(tight_eye)
            
            if len(test_frames) >= 25:
                final_eye = test_frames[-1]
                # Diagnostic Calculations
                ci, yi = process_eye_diagnostics(final_eye)
                frame_means = [np.mean(f) for f in test_frames]
                fs = (np.std(frame_means) / (np.mean(frame_means) + 1e-6)) * 100
                
                # High-Resolution Heatmap (Eye Only)
                h_map = cv2.applyColorMap(cv2.resize(cv2.cvtColor(final_eye, cv2.COLOR_BGR2GRAY), (300, 200)), cv2.COLORMAP_HOT)
                temp_dir = tempfile.gettempdir()
                heatmap_path = os.path.join(temp_dir, f"nayana_diagnostic_{int(time.time())}.png")
                cv2.imwrite(heatmap_path, h_map)
                stage = "FINISH"

        elif stage == "FINISH":
            results = {
                "clarity": float(ci), "stability": float(fs), "age_index": float(yi),
                "heatmap_path": heatmap_path, "timestamp": time.ctime()
            }
            break

        # UI Final Render
        cv2.putText(display, f"MODE: {mode_label}", (20, h-40), 1, 1, (255, 255, 255), 1)
        cv2.rectangle(display, roi_target[:2], (roi_target[0]+roi_target[2], roi_target[1]+roi_target[3]), roi_color, 2)
        cv2.imshow(window_name, display)
        if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    return results

if __name__ == "__main__":
    res = run_unified_scanner()
    print("PRECISION EYE RESULTS:", res)
