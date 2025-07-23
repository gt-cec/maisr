from flask import Flask, render_template_string, send_from_directory
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Low-Latency Stream</title>
        <style>
            body { background: #111; text-align: center; color: white; }
            canvas { border: 1px solid #555; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Live Game Board</h1>
        <canvas id="board" width="300" height="300"></canvas>
        <script src="https://cdn.socket.io/4.3.2/socket.io.min.js"></script>
        <script src="/pako.min.js"></script>
        <script>
            const socket = io();
            const canvas = document.getElementById('board');
            const ctx = canvas.getContext('2d');

            socket.on('frame', function(data) {
                const dv = new DataView(data);
                const width = dv.getUint32(0);
                const height = dv.getUint32(4);
                const compressed = new Uint8Array(data.slice(8));
                const raw = pako.inflate(compressed);

                const imgData = ctx.createImageData(width, height);
                for (let i = 0, j = 0; i < raw.length; i += 3, j += 4) {
                    imgData.data[j] = raw[i];
                    imgData.data[j+1] = raw[i+1];
                    imgData.data[j+2] = raw[i+2];
                    imgData.data[j+3] = 255;
                }
                ctx.putImageData(imgData, 0, 0);
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/pako.min.js')
def serve_pako():
    return send_from_directory('.', 'pako.min.js')

@socketio.on('frame')
def handle_frame(data):
    emit('frame', data, broadcast=True, binary=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001)
