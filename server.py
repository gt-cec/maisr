import json
import os
import time

from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO, emit, disconnect
import atexit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

COMPLETED_FILE = "completed_participants.json"
completed_participants = set()

def load_completed_participants():
    global completed_participants
    if os.path.exists(COMPLETED_FILE):
        with open(COMPLETED_FILE, "r") as f:
            try:
                completed_participants = set(json.load(f))
                print(f"Loaded {len(completed_participants)} completed participants from file.")
            except Exception as e:
                print(f"Error loading completed participants: {e}")
                completed_participants = set()
    else:
        completed_participants = set()

def save_completed_participants():
    with open(COMPLETED_FILE, "w") as f:
        json.dump(list(completed_participants), f)
        print(f"Saved {len(completed_participants)} completed participants to file.")

@socketio.on('connect', namespace='/')
def handle_connect():
    client_ip = request.remote_addr
    if client_ip in completed_participants and client_ip not in ['192.168.1.183']:
        print(f"Rejected repeat connection from {client_ip}")
        # Disconnect the client
        return False
    print(f"Client connected: {client_ip}")

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/pako.min.js')
def serve_pako():
    return send_from_directory('.', 'pako.min.js')


@socketio.on('frame',namespace='/')
def handle_frame(data):
    start_time = time.time()
    emit('frame', data, broadcast=True, binary=True)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    #print(f"[Latency Log] Frame processed and broadcast in {latency_ms:.2f} ms")


# @socketio.on('frame')
# def handle_frame(data):
#     emit('frame', data, broadcast=True, binary=True)

@socketio.on('click',namespace='/')
def handle_click(data):
    print(f"Click received: {data}")
    emit('click_response', data, broadcast=True)


@socketio.on('keydown',namespace='/')
def handle_keydown(data):
    print(f"Key down received: {data}")
    emit('keydown_response', data, broadcast=True)


@socketio.on('study_complete', namespace='/')
def handle_study_complete(data=None):
    client_ip = request.remote_addr
    completed_participants.add(client_ip)
    save_completed_participants()  # Save immediately
    print(f"Participant {client_ip} marked as completed")
    disconnect()


@atexit.register
def on_shutdown():
    save_completed_participants()


if __name__ == '__main__':
    load_completed_participants()
    socketio.run(app, host='0.0.0.0', port=5001)