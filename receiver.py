from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from PIL import Image

# 1. Import your AI function
from app import analyze_front_eye

# 2. Import your existing Authentication logic
from auth import login_patient, login_doctor, login_admin

# 3. Import your existing Database logic
from database import load_cases

app = Flask(__name__)
CORS(app) # Allows your JS frontend to talk to this Python backend

SHARED_DIR = "shared_data"
os.makedirs(SHARED_DIR, exist_ok=True)

# -------------------------------------------------------------------
# ROUTE 1: The Hardware & AI Pipeline (Used by index.html & capture.py)
# -------------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image sent"}), 400
        
    file = request.files['image']
    filepath = os.path.join(SHARED_DIR, "latest_eye.jpg")
    file.save(filepath)
    
    try:
        img = Image.open(filepath)
        results = analyze_front_eye(img)
        return jsonify({
            "status": "success",
            "predictions": results
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# ROUTE 2: Authentication (Used by your future login.html)
# -------------------------------------------------------------------
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'patient') # Defaults to patient if not specified
    
    # Route the login attempt to the correct function from auth.py
    if role == 'doctor':
        success, user_data, msg = login_doctor(email, password)
    elif role == 'admin':
        success, msg = login_admin(email, password)
        user_data = {"email": email} if success else None
    else:
        success, user_data, msg = login_patient(email, password)
        
    if success:
        return jsonify({"status": "success", "user": user_data, "role": role, "message": msg}), 200
    else:
        return jsonify({"status": "error", "error": msg}), 401

# -------------------------------------------------------------------
# ROUTE 3: Database Records (Used by your future dashboard.html)
# -------------------------------------------------------------------
@app.route('/records', methods=['GET'])
def records():
    try:
        # Fetch all cases from the encrypted database.py logic
        cases = load_cases()
        return jsonify({
            "status": "success", 
            "cases": cases
        }), 200
    except Exception as e:
         return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    # host='0.0.0.0' is critical so the Pi can still reach it
    app.run(host='0.0.0.0', port=5000)