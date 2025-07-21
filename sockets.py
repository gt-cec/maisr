import socketio

sio = socketio.Client()

@sio.event
def connect():
    print('Connection established')
    sio.emit('click', {'data': 'Hello from Python client!'}) # Example: emitting an event

@sio.event
def disconnect():
    print('Disconnected from server')

@sio.event
def my_response(data): # Example: handling a custom event from the server
    print('Server response:', data)

try:
    sio.connect('http://localhost:5001/') # Replace with your server URL
    sio.wait()
except Exception as e:
    print(f"Error connecting or during runtime: {e}")
finally:
    sio.disconnect() # Ensure disconnection on exit