Nayana.ai

A smartphone-based tele-ophthalmology AI screening system for early detection of eye diseases. Built with Raspberry Pi hardware integration, PyTorch computer vision models, and Streamlit interface.

Overview

Nayana.ai is a 3-step screening app (Symptoms → Eye Photos → Results) that combines edge hardware, deep learning, and blockchain-secured record-keeping for remote eye disease detection.

Core Features:



Raspberry Pi-based image capture with Camera Module V2 + LED flash

EfficientNet-B0 model trained on ODIR-5K dataset

OpenCV-powered anterior eye analysis

GradCAM heatmap visualization for interpretability

Multilingual voice input and PDF report generation

Blockchain-based result immutability



Tech Stack

Hardware: Raspberry Pi 4 (Trixie 64-bit), Pi Camera Module V2, custom LED flash circuit (330Ω resistor)

Backend: Flask server, Python 3.10, PyTorch, timm, OpenCV, ReportLab

Frontend: Streamlit

Security: AES encryption, SHA-256 blockchain ledger

Models:



eye\_model\_v2\_best.pth (primary classifier)

eye\_model\_clean\_best.pth (backup model)



Project Structure

nayana-eye-screening/

├── capture.py              # Raspberry Pi camera trigger + HTTP transmission

├── receiver.py             # Flask server (port 5000) for receiving images

├── app.py                  # Main Streamlit UI

├── doctor\_dashboard.py     # Doctor interface

├── patient\_records.py      # Patient record management

├── blockchain.py           # SHA-256 result hashing

├── encryption.py           # AES encryption utilities

├── database.py             # SQLite database logic

├── auth.py                 # Authentication handlers

├── chatbot\_flow.py         # Chatbot integration

├── voice\_input.py          # Voice input processing

├── report\_generator.py     # PDF report generation

└── nayana.key             # Encryption key (DO NOT COMMIT)

Prerequisites

Raspberry Pi Setup:



Raspberry Pi 4 with Trixie 64-bit OS

Pi Camera Module V2 enabled (sudo raspi-config)

Python 3.7+ with picamera2, requests, RPi.GPIO



Laptop/Server Setup:



Python 3.10+

CUDA-capable GPU (recommended, not required)

4GB+ RAM



Installation

1\. Clone the Repository

bashgit clone https://github.com/AnaghaBL/nayana-eye-screening.git

cd nayana-eye-screening

2\. Install Dependencies

bashpip install -r requirements.txt

3\. Raspberry Pi Setup

SSH into your Pi and install the capture script dependencies:

bashpip install picamera2 requests RPi.GPIO

Edit capture.py to set your laptop's local IP:

pythonSERVER\_URL = "http://<YOUR\_LAPTOP\_IP>:5000/upload"  # Replace with actual IP

4\. Download Models

Place the trained PyTorch models in the project root:



eye\_model\_v2\_best.pth

eye\_model\_clean\_best.pth



Usage

Running the Full Pipeline

Step 1: Start the Image Receiver (Laptop)

bashpython receiver.py

The Flask server will listen on http://0.0.0.0:5000.

Step 2: Trigger Image Capture (Raspberry Pi)

bashpython capture.py

The Pi will capture an image, flash the LED, and POST it to the receiver.

Step 3: Launch the Streamlit App (Laptop)

bashstreamlit run app.py

Navigate to http://localhost:8501 in your browser.

Optional: Doctor Dashboard

bashstreamlit run doctor\_dashboard.py

Hardware Wiring

LED Flash Circuit:



GPIO Pin 17 → 330Ω resistor → LED anode

LED cathode → GND



Camera Module:



Connect Pi Camera Module V2 to CSI port

Enable camera in raspi-config



Security Notes



DO NOT commit nayana.key to version control

Add nayana.key to .gitignore

Patient data is AES-encrypted at rest

Screening results are hashed to blockchain for tamper detection



Troubleshooting

Pi Camera Not Detected:

bashsudo raspi-config

\# Interface Options → Camera → Enable

sudo reboot

Connection Refused (receiver.py):



Check firewall settings on laptop

Verify Pi and laptop are on same network

Test with curl http://<LAPTOP\_IP>:5000



Model Loading Error:



Ensure PyTorch version matches training environment

Check model file paths in app.py



Team



Anagha (Hardware Integration)

Medha (Raspberry Pi Setup, Optical Assembly)

Khushi (Frontend \& UI)



License

MIT License (or specify your license)

