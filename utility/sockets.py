import numpy as np
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


class DeltaFrameManager:
    def __init__(self, change_threshold=30, full_frame_interval=30):
        """
        Args:
            change_threshold: Pixel difference threshold (0-255)
            full_frame_interval: Send full frame every N frames to prevent drift
        """
        self.last_frame = None
        self.change_threshold = change_threshold
        self.full_frame_interval = full_frame_interval
        self.frame_count = 0
        self.last_full_frame = 0

    def should_send_frame(self, current_frame_array):
        """
        Determine if we should send a frame and what type
        Returns: (should_send, frame_type, processed_frame)
        frame_type: 'full', 'delta', or 'skip'
        """
        self.frame_count += 1

        # Force full frame periodically to prevent drift
        if (self.frame_count - self.last_full_frame) >= self.full_frame_interval:
            self.last_frame = current_frame_array.copy()
            self.last_full_frame = self.frame_count
            return True, 'full', current_frame_array

        # First frame is always full
        if self.last_frame is None:
            self.last_frame = current_frame_array.copy()
            self.last_full_frame = self.frame_count
            return True, 'full', current_frame_array

        # Calculate differences
        diff = np.abs(current_frame_array.astype(np.int16) - self.last_frame.astype(np.int16))

        # Create mask of significantly changed pixels
        change_mask = np.max(diff, axis=2) > self.change_threshold

        # Calculate percentage of changed pixels
        change_ratio = np.sum(change_mask) / change_mask.size

        # If too few changes, skip frame
        #if change_ratio < 0.01:  # Less than 1% changed
            #return False, 'skip', None

        # If major changes, send full frame
        if change_ratio > 0.25:  # More than 25% changed
            self.last_frame = current_frame_array.copy()
            self.last_full_frame = self.frame_count
            return True, 'full', current_frame_array

        # Send delta frame
        delta_data = {
            'mask': change_mask,
            'changed_pixels': current_frame_array[change_mask],
            'positions': np.where(change_mask)
        }

        self.last_frame = current_frame_array.copy()
        return True, 'delta', delta_data


# Initialize the delta manager globally
delta_manager = DeltaFrameManager(change_threshold=2, full_frame_interval=30)


def pygame_surface_to_array(surface):
    """Convert pygame surface to numpy array"""
    w, h = surface.get_size()
    raw = pygame.image.tostring(surface, 'RGB')
    array = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
    return array


def compress_full_frame(frame_array, quality=70):
    """Compress full frame using JPEG"""
    h, w, c = frame_array.shape
    pil_image = Image.fromarray(frame_array, 'RGB')

    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
    compressed = buffer.getvalue()

    return compressed


def compress_delta_frame(delta_data):
    """Compress delta frame data in a format JavaScript can handle"""
    mask = delta_data['mask']
    changed_pixels = delta_data['changed_pixels']
    positions = delta_data['positions']

    #print(f"Delta compression: {len(changed_pixels)} changed pixels")

    # Create a simple binary format that JavaScript can parse
    # Format: [num_pixels (4 bytes)] + [y1,x1,r1,g1,b1, y2,x2,r2,g2,b2, ...]
    num_pixels = len(changed_pixels)

    # Pack data: 4 bytes for count, then 5 bytes per pixel (y,x,r,g,b)
    data = bytearray()
    data.extend(struct.pack('>I', num_pixels))  # Big endian uint32

    for i in range(num_pixels):
        y = positions[0][i]
        x = positions[1][i]
        r, g, b = changed_pixels[i]

        # Pack as: y(2 bytes), x(2 bytes), r(1 byte), g(1 byte), b(1 byte)
        data.extend(struct.pack('>HHBBB', y, x, r, g, b))

    # Compress the binary data
    compressed = zlib.compress(data, level=6)
    #print(f"Delta compressed: {len(data)} -> {len(compressed)} bytes")

    return compressed


def send_frame_with_delta(window, quality=70):
    """
    Enhanced send_frame function with delta compression
    """
    try:
        # Convert pygame surface to numpy array
        frame_array = pygame_surface_to_array(window)

        # Check if we should send this frame
        should_send, frame_type, processed_frame = delta_manager.should_send_frame(frame_array)

        if not should_send:
            print("Skipping frame - no significant changes")
            return  # Skip this frame

        w, h = window.get_size()

        if frame_type == 'full':
            #print(f"Sending full frame ({w}x{h})")
            # Send full frame
            compressed = compress_full_frame(processed_frame, quality)
            header = struct.pack('>III', w, h, 0)  # 0 = full frame

        elif frame_type == 'delta':
            #print(f"Sending delta frame ({w}x{h})")
            # Send delta frame
            compressed = compress_delta_frame(processed_frame)
            header = struct.pack('>III', w, h, 1)  # 1 = delta frame

        # Send the frame
        total_size = len(header) + len(compressed)
        #print(f"Sending {frame_type} frame: {total_size} bytes")
        sio.emit('frame', header + compressed)

        # Optional: print stats
        if frame_type == 'delta':
            change_count = len(processed_frame['changed_pixels'])
            total_pixels = w * h
            #print(f"Delta frame: {change_count}/{total_pixels} pixels changed ({100 * change_count / total_pixels:.1f}%)")

    except Exception as e:
        print(f"Error sending frame: {e}")
        import traceback
        traceback.print_exc()


##################################################

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