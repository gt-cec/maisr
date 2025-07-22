import pygame
from PIL import Image
from io import BytesIO
import socketio
import zlib
import struct

# SocketIO client
sio = socketio.Client()
sio.connect("http://localhost:5001")

def send_frame(window):
    raw = pygame.image.tostring(window, 'RGB')
    compressed = zlib.compress(raw)
    header = struct.pack('>II', window.get_width(), window.get_height())  # width, height
    sio.emit('frame', header + compressed)
