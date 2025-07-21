# Notes: install Flask and Flask-SocketIO before running the server:
#   pip install Flask Flask-SocketIO
# The server will be accessible at http://localhost:5000/

# set up a flask server with socketio
from flask import Flask, render_template, request
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)
current_frame = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/screen_update', methods=['POST'])
def screen_update():
    global current_frame
    data = request.get_json()
    if 'image' in data:
        print('Received screen update:', data['screen_index'])
        # send as base64 encoded image
        if isinstance(data['image'], str):
            # If the image is already a base64 string, we can directly emit it
            socketio.emit('image', {'image': data['image']})
        else:
            # If the image is a binary string, we need to convert it to base64
            import base64
            encoded_image = base64.b64encode(data['image']).decode('utf-8')
            socketio.emit('image', {'image': encoded_image})
        current_frame = data['image']
        return {'status': 'success'}, 200
    else:
        return {'status': 'error', 'message': 'No image provided'}, 400

@socketio.on('image')
def image(data):
    global current_frame
    current_frame = data['image']
    print('Received image frame')
    socketio.emit('response', {'data': 'Image received'})

@socketio.on('message')
def handle_message(data):
    print('Received message: ' + data)
    socketio.send('Message received: ' + data)
    socketio.emit('response', {'data': 'Message received: ' + data})
    socketio.send('Message received: ' + data)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.send('Welcome to the server!')
    socketio.emit('response', {'data': 'Welcome to the server!'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    socketio.send('Goodbye!')
    socketio.emit('response', {'data': 'Goodbye!'}) 

@socketio.on('click')
def handle_click(data):
    print('Click event received: ' + str(data))
    socketio.send('Click event received: ' + str(data))

if __name__ == '__main__':
    socketio.run(app, port=5001, debug=True)  # run the run_experiment function in a separate thread
