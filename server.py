from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/pako.min.js')
def serve_pako():
    return send_from_directory('.', 'pako.min.js')

@socketio.on('frame')
def handle_frame(data):
    emit('frame', data, broadcast=True, binary=True)

@socketio.on('click')
def handle_click(data):
    print(f"Click received: {data}")
    emit('click_response', data, broadcast=True)

@socketio.on('keydown')
def handle_keydown(data):
    print(f"Key down received: {data}")
    emit('keydown_response', data, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001)
