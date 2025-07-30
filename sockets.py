import pygame
from PIL import Image
from io import BytesIO
import socketio
import zlib
import struct

# SocketIO client
#sio.connect("http://localhost:5001")
#sio.connect("http://192.168.1.183:5001")

sio = socketio.Client()



@sio.event
def connect_error(data):
    print("Connection failed:", data)

@sio.event
def disconnect():
    print("Disconnected from server")

def connect():
    print("Connecting to server...")
    #sio.connect('http://99.45.36.114:5001', wait_timeout=10, namespaces=['/'])
    sio.connect('http://192.168.1.183:5001', wait_timeout=10, namespaces=['/'])
    print("Connected!")

human_controller = None
instruction_controller = None
pyg = pygame

def key_string_to_pygame(key_str):
    # Map common keys from JS to pygame
    keymap = {
        'ArrowLeft': pygame.K_LEFT,
        'ArrowRight': pygame.K_RIGHT,
        'ArrowUp': pygame.K_UP,
        'ArrowDown': pygame.K_DOWN,
        'Enter': pygame.K_RETURN,
        'Escape': pygame.K_ESCAPE,
        ' ': pygame.K_SPACE,
        'Shift': pygame.K_LSHIFT,
        'Control': pygame.K_LCTRL,
        'Alt': pygame.K_LALT,
    }
    if len(key_str) == 1:  # alphabet, numbers
        return ord(key_str.lower())
    return keymap.get(key_str, None)

def send_frame(window):
    raw = pygame.image.tostring(window, 'RGB')
    compressed = zlib.compress(raw)
    header = struct.pack('>II', window.get_width(), window.get_height())  # width, height
    try:
        sio.emit('frame', header + compressed)
    except socketio.exceptions.BadNamespaceError as e:
        print(f"Error sending frame, reconnecting")
        # Handle reconnection logic if needed
        sio.connect("http://localhost:5001")


# def send_frame(window):
#     # Option A: Use PNG compression (better quality, potentially smaller)
#     w, h = window.get_size()
#     raw = pygame.image.tostring(window, 'RGB')
#
#     # Convert to PIL Image for better compression
#     pil_image = Image.frombuffer('RGB', (w, h), raw)
#
#     # Use PNG with optimization
#     buffer = BytesIO()
#     pil_image.save(buffer, format='PNG', optimize=True, compress_level=6)
#     compressed = buffer.getvalue()
#
#     header = struct.pack('>II', w, h)
#     try:
#         # Remove the binary=True parameter - not needed for client emit
#         sio.emit('frame', header + compressed)
#     except Exception as e:
#         print(f"Error sending frame: {e}")


# def send_frame_jpeg(window, quality=75):
#     w, h = window.get_size()
#     raw = pygame.image.tostring(window, 'RGB')
#     pil_image = Image.frombuffer('RGB', (w, h), raw)
#
#     buffer = BytesIO()
#     pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
#     compressed = buffer.getvalue()
#
#     header = struct.pack('>II', w, h)
#     sio.emit('frame', header + compressed, binary=True)

# catch the click_response event
@sio.on('click_response')
def click_response(data):
    try:
        from pygame.event import Event
        print(f"Click response received: {data}")
        event = Event(pygame.MOUSEBUTTONDOWN, {
            "pos": (data["x"], data["y"]),
            "button": 1
        })
        pygame.event.post(event)
    except Exception as e:
        print("Error posting click event:", e)

@sio.on('keydown_response')
def keydown_response(data):
    try:
        from pygame.event import Event  # <--- force import capital E

        print(f"Key down response received: {data}", human_controller, instruction_controller)
        keycode = key_string_to_pygame(data["key"])
        if keycode is not None:
            event = Event(pygame.KEYDOWN, {"key": keycode})  # <--- use Event directly
            pygame.event.post(event)
        else:
            print(f"Warning: Unknown key {data['key']}")
    except Exception as e:
        print("Error posting keydown event:", e)