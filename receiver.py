from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────
SAVE_DIR = "./received_images"
os.makedirs(SAVE_DIR, exist_ok=True)

LATEST_PATH = os.path.join(SAVE_DIR, "latest_eye.jpg")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    img = request.files["image"]
    img.save(LATEST_PATH)
    print(f"[receiver] Image saved to {LATEST_PATH}")

    return jsonify({"status": "ok", "saved_to": LATEST_PATH}), 200

@app.route("/latest", methods=["GET"])
def latest():
    """Nayana Streamlit app can GET this to check for a new image."""
    if os.path.exists(LATEST_PATH):
        return jsonify({"available": True, "path": LATEST_PATH}), 200
    return jsonify({"available": False}), 200

if __name__ == "__main__":
    print("=== Nayana receiver running on port 5000 ===")
    print(f"    Images will be saved to: {os.path.abspath(SAVE_DIR)}")
    app.run(host="0.0.0.0", port=5000, debug=False)
