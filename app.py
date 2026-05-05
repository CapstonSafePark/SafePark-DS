from lane_detector import run_lane_detection
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/ds/line-detect", methods=["POST"])
def line_detect():
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    result = run_lane_detection(save_path)

    if "error" in result:
        return jsonify(result), 500
    result["image_path"] = save_path

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)