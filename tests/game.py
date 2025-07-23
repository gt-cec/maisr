import pygame
import socketio
import zlib
import struct
import random

# Setup
WIDTH, HEIGHT = 300, 300
SERVER_URL = 'http://localhost:5001'

# Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Socket.IO client
sio = socketio.Client()
sio.connect(SERVER_URL)

def draw_board():
    screen.fill((30, 30, 30))
    for i in range(0, WIDTH, 100):
        for j in range(0, HEIGHT, 100):
            pygame.draw.rect(screen, (random.randint(0, 200), random.randint(0, 200), 50), (i + 10, j + 10, 80, 80))

def send_frame():
    raw = pygame.image.tostring(screen, 'RGB')
    compressed = zlib.compress(raw)
    header = struct.pack('>II', WIDTH, HEIGHT)  # width, height
    sio.emit('frame', header + compressed)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_board()
    pygame.display.flip()
    send_frame()
    clock.tick(1)  # 30 FPS target

sio.disconnect()
pygame.quit()
