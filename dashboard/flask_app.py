import os
import sys
import cv2
import time
import threading
from flask import Flask, render_template, Response, jsonify

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config
from src.inference.pipeline import InferencePipeline

app = Flask(__name__)

# Global state for web APIs
LATEST_FRAME = None
LATEST_STATS = {
    "students": [],
    "frame_info": {},
    "active_alerts": [],
    "fps": 0.0
}
LOCK = threading.Lock()

def inference_worker():
    global LATEST_FRAME, LATEST_STATS, LOCK
    
    config = load_config("config/config.yaml")
    # Quick defaults for robustness
    checkpoint_path = "checkpoints/best_model.pth"
    
    print(f"Loading AI Model from {checkpoint_path}...")
    pipeline = InferencePipeline(config=config, checkpoint_path=checkpoint_path)
    
    # Init capture (Camera 0)
    cap = cv2.VideoCapture(0)
    
    while True:
        # Drain buffer for absolute zero-latency feed
        for _ in range(3): cap.grab()
        
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            # Reconnect camera if failed
            cap.release()
            cap = cv2.VideoCapture(0)
            continue
            
        t_start = time.time()
        result = pipeline.process_frame(frame)
        
        # Calculate pipeline FPS
        processing_time = result["frame_info"]["processing_time_ms"] / 1000.0
        fps = 1.0 / max(processing_time, 0.01)
        
        # We don't need CV2 string annotations on top of faces, we can use raw boxes, but let's keep annotated for quick setup.
        annotated_frame = result["annotated_frame"]
        
        # Compress to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret: continue
        
        frame_bytes = buffer.tobytes()
        
        # Extract frontend stats (avoiding non-serializable objects)
        alerts = [s for s in result["students"] if s["risk"].get("alert_active")]
        safe_students = []
        for s in result["students"]:
            safe_students.append({
                "student_id": s["student_id"],
                "engagement_class": s["predictions"].get("engagement", 0),
                "attention_score": s["attention"].get("attention_score", 0),
                "risk_level": s["risk"].get("risk_level", "low"),
                "risk_score": s["risk"].get("risk_score", 0),
                "risk_history": [h["smoothed_score"] for h in s.get("risk_history", [])][-30:] # Last 30 points
            })
            
        with LOCK:
            LATEST_FRAME = frame_bytes
            LATEST_STATS = {
                "students": safe_students,
                "frame_info": result["frame_info"],
                "active_alerts": [{"id": a["student_id"], "msg": a["risk"].get("alert_message", "Pattern change")} for a in alerts],
                "fps": round(fps, 1)
            }
            
        time.sleep(0.01) # give threads breathing room

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global LATEST_FRAME, LOCK
    while True:
        frame_data = None
        with LOCK:
            frame_data = LATEST_FRAME
        
        if frame_data is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    global LATEST_STATS, LOCK
    with LOCK:
        return jsonify(LATEST_STATS)

if __name__ == '__main__':
    # Start background inference loop
    threading.Thread(target=inference_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
