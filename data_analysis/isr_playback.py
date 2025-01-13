import pygame
import json
import math
import argparse
from pathlib import Path


class PlaybackViewer:
    def __init__(self, log_file):
        # Initialize pygame
        pygame.init()

        # Constants from original simulation
        self.WINDOW_SIZE = (1450, 1080)
        self.GAMEBOARD_SIZE = 1000
        self.AIRCRAFT_COLORS = [(0, 160, 160), (0, 0, 255)]  # AI, Human
        self.AGENT_COLOR_UNOBSERVED = (255, 215, 0)  # gold
        self.AGENT_COLOR_OBSERVED = (128, 0, 128)  # purple
        self.AGENT_COLOR_THREAT = (255, 0, 0)  # red
        self.AGENT_THREAT_RADIUS = [0, 1.4, 2.5, 4]

        # Aircraft rendering constants
        self.AIRCRAFT_NOSE_LENGTH = 10
        self.AIRCRAFT_TAIL_LENGTH = 25
        self.AIRCRAFT_TAIL_WIDTH = 7
        self.AIRCRAFT_WING_LENGTH = 18
        self.AIRCRAFT_LINE_WIDTH = 5
        self.AIRCRAFT_ENGAGEMENT_RADIUS = 40
        self.AIRCRAFT_ISR_RADIUS = 85

        # Setup display
        self.window = pygame.display.set_mode(self.WINDOW_SIZE)
        pygame.display.set_caption("ISR Experiment Playback")
        self.clock = pygame.time.Clock()

        # Load game states
        self.states = self.load_states(log_file)
        self.current_state_idx = 0

        # Auto-play settings
        self.is_playing = False
        self.last_auto_step = 0
        self.auto_step_delay = 500  # 0.5 seconds in milliseconds

        # UI elements
        self.button_height = 40
        self.button_width = 80
        button_y = self.WINDOW_SIZE[1] - self.button_height - 10

        self.prev_button = pygame.Rect(10, button_y, self.button_width, self.button_height)
        self.next_button = pygame.Rect(100, button_y, self.button_width, self.button_height)
        self.play_button = pygame.Rect(190, button_y, self.button_width, self.button_height)
        self.font = pygame.font.SysFont(None, 36)

    def load_states(self, log_file):
        states = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    print(json.loads(line))
                    #print(type(json.loads(line)))
                    #print('\n')
                    data = json.loads(line)

                    if not type(data) == str:
                        print('data is a dict')
                        if data.get('type') == 'state' and 'game_state' in data:
                            states.append(data['game_state'])

                except json.JSONDecodeError:
                    continue  # Skip invalid lines

        if not states: raise ValueError("No valid game states found in log file")
        return states

    def draw_aircraft(self, x, y, direction, color, waypoint):
        # Draw aircraft body
        AIRCRAFT_ISR_RADIUS = 85
        AIRCRAFT_DETECTION_RADIUS = 40
        nose_point = (x + math.cos(direction) * self.AIRCRAFT_NOSE_LENGTH,
                      y + math.sin(direction) * self.AIRCRAFT_NOSE_LENGTH)
        tail_point = (x - math.cos(direction) * self.AIRCRAFT_TAIL_LENGTH,
                      y - math.sin(direction) * self.AIRCRAFT_TAIL_LENGTH)

        # Draw wings
        left_wingtip = (x - math.cos(direction - math.pi / 2) * self.AIRCRAFT_WING_LENGTH,
                        y - math.sin(direction - math.pi / 2) * self.AIRCRAFT_WING_LENGTH)
        right_wingtip = (x + math.cos(direction - math.pi / 2) * self.AIRCRAFT_WING_LENGTH,
                         y + math.sin(direction - math.pi / 2) * self.AIRCRAFT_WING_LENGTH)

        # Draw tail
        left_tail = (tail_point[0] - math.cos(direction - math.pi / 2) * self.AIRCRAFT_TAIL_WIDTH,
                     tail_point[1] - math.sin(direction - math.pi / 2) * self.AIRCRAFT_TAIL_WIDTH)
        right_tail = (tail_point[0] + math.cos(direction - math.pi / 2) * self.AIRCRAFT_TAIL_WIDTH,
                      tail_point[1] + math.sin(direction - math.pi / 2) * self.AIRCRAFT_TAIL_WIDTH)

        # Draw all components
        pygame.draw.line(self.window, color, tail_point, nose_point, self.AIRCRAFT_LINE_WIDTH)
        pygame.draw.line(self.window, color, left_wingtip, right_wingtip, self.AIRCRAFT_LINE_WIDTH)
        pygame.draw.line(self.window, color, left_tail, right_tail, self.AIRCRAFT_LINE_WIDTH)

        # Draw circles at joints
        for point in [nose_point, left_wingtip, right_wingtip, left_tail, right_tail]:
            pygame.draw.circle(self.window, color, point, self.AIRCRAFT_LINE_WIDTH / 2)

        # Draw engagement radius
        pygame.draw.circle(self.window, color, (x, y),40, 2)

        # draw the ISR radius
        target_rect = pygame.Rect((x, y), (0, 0)).inflate((AIRCRAFT_ISR_RADIUS * 2, AIRCRAFT_ISR_RADIUS * 2))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        semicircle_points = [(AIRCRAFT_ISR_RADIUS + math.cos(direction + math.pi * i / 180) * AIRCRAFT_ISR_RADIUS,
                              AIRCRAFT_ISR_RADIUS + math.sin(direction + math.pi * i / 180) * AIRCRAFT_ISR_RADIUS) for i in range(-90, 90 + 10, 10)]
        pygame.draw.polygon(shape_surf, color + (30,), semicircle_points)

        self.window.blit(shape_surf, target_rect)

        if not waypoint == [0,0]:
            pygame.draw.line(self.window, (0, 0, 0), (x, y), (waypoint[0], waypoint[1]),2)  # Draw line from aircraft to waypoint
            pygame.draw.rect(self.window, color, pygame.Rect(waypoint[0] - 5, waypoint[1] - 5, 10, 10))  # Draw box at waypoint location

    def draw_ship(self, ship):
        x, y = ship["position"]
        width = 10

        # Determine ship color based on observation status
        if not ship["observed"]:
            color = self.AGENT_COLOR_UNOBSERVED
        else:
            if ship["threat"] > 0:
                color = self.AGENT_COLOR_THREAT
            else:
                color = self.AGENT_COLOR_OBSERVED

        # Draw ship
        pygame.draw.circle(self.window, color, (x, y), width)

        # Draw threat radius if applicable
        threat_radius = width * self.AGENT_THREAT_RADIUS[ship["threat"]]
        possible_threat_radius = width * self.AGENT_THREAT_RADIUS[3]

        # Draw orange circle for unidentified targets
        if not ship["observed_threat"] or (ship["observed_threat"] and ship["threat"] == 0):
            pygame.draw.circle(self.window, self.AGENT_COLOR_UNOBSERVED,
                               (x, y), possible_threat_radius, 2)

        # Draw white circle for confirmed neutral
        if ship["observed_threat"] and ship["threat"] == 0:
            pygame.draw.circle(self.window, (255, 255, 255),
                               (x, y), possible_threat_radius, 2)

        # Draw red circle for confirmed hostile
        elif ship["observed_threat"] and ship["threat"] > 0:
            pygame.draw.circle(self.window, self.AGENT_COLOR_THREAT,
                               (x, y), threat_radius, 2)

    def draw_buttons(self):
        # Draw navigation buttons
        pygame.draw.rect(self.window, (200, 200, 200), self.prev_button)
        pygame.draw.rect(self.window, (200, 200, 200), self.next_button)
        pygame.draw.rect(self.window, (200, 200, 200), self.play_button)

        prev_text = self.font.render("←", True, (0, 0, 0))
        next_text = self.font.render("→", True, (0, 0, 0))
        play_text = self.font.render("⏸" if self.is_playing else "▶", True, (0, 0, 0))

        self.window.blit(prev_text, (self.prev_button.centerx - 10, self.prev_button.centery - 10))
        self.window.blit(next_text, (self.next_button.centerx - 10, self.next_button.centery - 10))
        self.window.blit(play_text, (self.play_button.centerx - 10, self.play_button.centery - 10))

        # Draw state counter
        counter_text = self.font.render(f"State: {self.current_state_idx + 1}/{len(self.states)}",
                                        True, (0, 0, 0))
        self.window.blit(counter_text, (300, self.WINDOW_SIZE[1] - self.button_height - 5))

        # Draw time
        time_text = self.font.render(f"Time: {self.states[self.current_state_idx]['time']}",True, (0, 0, 0))
        self.window.blit(time_text, (500, self.WINDOW_SIZE[1] - self.button_height - 5))

    def draw_state(self):
        state = self.states[self.current_state_idx]

        # Clear screen
        self.window.fill((255, 255, 255))

        # Draw grid lines
        pygame.draw.line(self.window, (0, 0, 0),
                         (self.GAMEBOARD_SIZE // 2, 0),
                         (self.GAMEBOARD_SIZE // 2, self.GAMEBOARD_SIZE), 2)
        pygame.draw.line(self.window, (0, 0, 0),
                         (0, self.GAMEBOARD_SIZE // 2),
                         (self.GAMEBOARD_SIZE, self.GAMEBOARD_SIZE // 2), 2)

        # Draw border
        pygame.draw.rect(self.window, (0, 0, 0),
                         (0, 0, self.GAMEBOARD_SIZE, self.GAMEBOARD_SIZE), 3)

        # Draw ships
        for ship in state["ships"]:
            self.draw_ship(ship)

        # Draw agent
        self.draw_aircraft(state["aircraft"][0]["position"][0],state["aircraft"][0]["position"][1], state["aircraft"][0]["direction"], (0, 160, 160), state["aircraft"][0]["waypoint"])

        # Draw human
        self.draw_aircraft(state["aircraft"][1]["position"][0], state["aircraft"][1]["position"][1], state["aircraft"][1]["direction"], (0, 0, 255),state["aircraft"][1]["waypoint"])

        # Draw UI elements
        self.draw_buttons()

        pygame.display.flip()

    def handle_auto_play(self):
        if self.is_playing:
            current_time = pygame.time.get_ticks()
            if current_time - self.last_auto_step >= self.auto_step_delay:
                if self.current_state_idx < len(self.states) - 1:
                    self.current_state_idx += 1
                    self.last_auto_step = current_time
                else:
                    # Stop playing when we reach the end
                    self.is_playing = False

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()

                    if self.prev_button.collidepoint(mouse_pos):
                        self.current_state_idx = max(0, self.current_state_idx - 1)
                    elif self.next_button.collidepoint(mouse_pos):
                        self.current_state_idx = min(len(self.states) - 1,
                                                     self.current_state_idx + 1)
                    elif self.play_button.collidepoint(mouse_pos):
                        self.is_playing = not self.is_playing
                        self.last_auto_step = pygame.time.get_ticks()

            self.handle_auto_play()
            self.draw_state()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    log_file = "maisr_subject914_round0_2025_01_13_14_31_52.jsonl"
    viewer = PlaybackViewer(log_file)
    viewer.run()