import pygame
from PIL import Image
from io import BytesIO
import socketio
import zlib
import struct

# SocketIO client
sio = socketio.Client()
sio.connect("http://localhost:5001")

human_controller = None
instruction_controller = None
pyg = None

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
    sio.emit('frame', header + compressed)

# catch the click_response event
@sio.on('click_response')
def click_response(data):
    print(f"Click response received: {data}")
    pyg.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": (data["x"], data["y"]), "button": 1}))

@sio.on('keydown_response')
def keydown_response(data):
    print(f"Key down response received: {data}", human_controller, instruction_controller)
    keycode = key_string_to_pygame(data["key"])
    pyg.post(pygame.event.Event(pygame.KEYDOWN, {"key": keycode}))
    # instruction_controller.handle_event(event = pygame.event.Event(pygame.KEYDOWN, {"key": keycode}))