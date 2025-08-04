import math

import numpy as np
import pygame
import sys
from typing import Dict, List, Optional, Callable
from enum import Enum
import cv2
import requests
#import sockets

FPS = 6

class ScreenType(Enum):
    """Types of instructional screens"""
    WELCOME = "welcome"
    INSTRUCTIONS = "instructions"
    BETWEEN_EPISODES = "between_episodes"
    FINAL_SUMMARY = "final_summary"

class InstructionalScreen:
    """Base class for instructional screens"""

    def __init__(self, window_width: int = 1000, window_height: int = 1100, sio = None, admin=False):

        font_path = './user_study/AcPlus_IBM_VGA_8x16.ttf'
        #font_size = 16
        #font = pygame.font.Font(font_path, font_size)

        self.window_width = window_width
        self.window_height = window_height
        self.background_color = (0,0,0) #(30, 30, 30)  # Dark gray
        self.text_color = (255, 255, 255)  # White
        self.highlight_color = (100, 150, 255)  # Light blue
        self.font_large = pygame.font.Font(font_path, 36) #pygame.font.SysFont('Arial', 36, bold=True)
        self.font_medium = pygame.font.Font(font_path, 24)# pygame.font.SysFont('Arial', 24)
        self.font_small = pygame.font.Font(font_path, 18) #pygame.font.SysFont('Arial', 18)

        # Input state
        self.text_input = ""
        self.cursor_visible = True
        self.cursor_timer = 0
        self.cursor_blink_rate = 500  # milliseconds
        self.admin = admin

        self.sio = sio

    def render(self, window: pygame.Surface) -> None:
        """Render the screen content"""
        window.fill(self.background_color)
        self.draw_content(window)
        self.draw_navigation_hints(window)

    def draw_content(self, window: pygame.Surface) -> None:
        """Override this method to draw screen-specific content"""
        pass

    def draw_navigation_hints(self, window: pygame.Surface) -> None:
        """Draw navigation hints at the bottom of the screen"""
        hint_text = "Press ENTER to continue, ESC to exit"
        hint_surface = self.font_small.render(hint_text, True, (150, 150, 150))
        hint_rect = hint_surface.get_rect(center=(self.window_width // 2, self.window_height - 30))
        window.blit(hint_surface, hint_rect)

    def handle_event(self, event: pygame.event.Event) -> Dict:
        """Handle pygame events. Returns dict with action information"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                return {"action": "continue", "text_input": self.text_input}
            elif event.key == pygame.K_ESCAPE and self.admin:
                return {"action": "exit"}
            elif event.key == pygame.K_LEFT:
                return {"action": "previous"}
            elif event.key == pygame.K_RIGHT:
                return {"action": "next"}
            elif event.key == pygame.K_BACKSPACE:
                if self.text_input:
                    self.text_input = self.text_input[:-1]
            else:
                # Handle text input
                if event.unicode.isprintable():
                    self.text_input += event.unicode

        return {"action": "none"}

    def update(self, dt: int) -> None:
        """Update screen state (called each frame)"""
        # Handle cursor blinking
        self.cursor_timer += dt
        if self.cursor_timer >= self.cursor_blink_rate:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0

    def draw_text_centered(self, window: pygame.Surface, text: str, y_pos: int,
                           font: pygame.font.Font, color: tuple = None) -> None:
        """Helper method to draw centered text"""
        if color is None:
            color = self.text_color
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.window_width // 2, y_pos))
        window.blit(text_surface, text_rect)

    def draw_text_block(self, window: pygame.Surface, text_lines: List[str],
                        start_y: int, font: pygame.font.Font,
                        line_spacing: int = 30, color: tuple = None) -> None:
        """Helper method to draw a block of text lines"""
        if color is None:
            color = self.text_color
        current_y = start_y
        for line in text_lines:
            text_surface = font.render(line, True, color)
            text_rect = text_surface.get_rect(center=(self.window_width // 2, current_y))
            window.blit(text_surface, text_rect)
            current_y += line_spacing

    def draw_text_input_box(self, window: pygame.Surface, x: int, y: int,
                            width: int, height: int, prompt: str = "") -> None:
        """Draw a text input box with cursor"""
        # Draw prompt
        if prompt:
            prompt_surface = self.font_medium.render(prompt, True, self.text_color)
            window.blit(prompt_surface, (x, y - 35))

        # Draw input box
        input_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(window, (50, 50, 50), input_rect)
        pygame.draw.rect(window, self.highlight_color, input_rect, 2)

        # Draw text
        if self.text_input:
            text_surface = self.font_medium.render(self.text_input, True, self.text_color)
            window.blit(text_surface, (x + 10, y + 10))

        # Draw cursor
        if self.cursor_visible:
            cursor_x = x + 10 + self.font_medium.size(self.text_input)[0]
            cursor_y = y + 5
            pygame.draw.line(window, self.text_color,
                             (cursor_x, cursor_y), (cursor_x, cursor_y + height - 10), 2)


class WelcomeScreen(InstructionalScreen):
    """Welcome screen shown at the start of the experiment"""

    def __init__(self, subject_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subject_id = subject_id

    def draw_content(self, window: pygame.Surface) -> None:
        # Title
        self.draw_text_centered(window, "Welcome to the MAISR User Study", 200,
                                self.font_large, self.highlight_color)

        # Subject ID
        self.draw_text_centered(window, f"Subject ID: {self.subject_id}", 260,
                                self.font_medium)

        # Instructions
        instructions = [
            "You will be participating in a study involving human-AI cooperation",
            "in a multi-agent search and rescue scenario.",
            "",
            "During the experiment, you will:",
            "• Control an agent using keyboard inputs and mouse clicks",
            "• Work with an AI teammate to complete search missions",
            "• Provide feedback after each episode",
            "",
            "The experiment consists of multiple episodes with different",
            "AI teammates and scenarios.",
            "",
            "Please read the instructions carefully and ask the experimenter",
            "if you have any questions."
        ]

        self.draw_text_block(window, instructions, 350, self.font_medium, 35)


class InstructionsScreen(InstructionalScreen):
    """Detailed instructions screen"""

    def draw_content(self, window: pygame.Surface) -> None:
        # Title
        self.draw_text_centered(window, "How to Play", 150,
                                self.font_large, self.highlight_color)

        # Instructions
        instructions = [
            "CONTROLS:",
            "• Keys 1-7: Select different movement strategies",
            "• Mouse Click: Direct your agent to a specific location",
            "• SPACE: Pause/Resume the game",
            "• ESC: Exit the current episode",
            "",
            "OBJECTIVES:",
            "• Identify targets (shown as green circles when found)",
            "• Avoid threat areas (shown as gold circles)",
            "• Work with your AI teammate to maximize mission success",
            "",
            "VISUAL INDICATORS:",
            "• Green targets = Successfully identified",
            "• Orange targets = Unknown/unidentified",
            "• Gold circles = Threat areas to avoid",
            "• Blue/Red agents = You and your teammate",
            "",
            "Remember: Communication with your AI teammate happens",
            "through your actions and movement choices."
        ]

        self.draw_text_block(window, instructions, 220, self.font_medium, 32)


class BetweenEpisodesScreen(InstructionalScreen):
    """Screen shown between episodes"""

    def __init__(self, episode_num: int, total_episodes: int,
                 last_episode_reward: float = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_num = episode_num
        self.total_episodes = total_episodes
        self.last_episode_reward = last_episode_reward

    def draw_content(self, window: pygame.Surface) -> None:
        # Title
        self.draw_text_centered(window, f"Episode {self.episode_num} of {self.total_episodes}",
                                200, self.font_large, self.highlight_color)

        # Last episode performance (if available)
        if self.last_episode_reward is not None:
            self.draw_text_centered(window, f"Previous Episode Score: {self.last_episode_reward:.2f}",
                                    260, self.font_medium, (100, 255, 100))

        # Instructions
        instructions = [
            "You have completed the previous episode.",
            "Please take a moment to rest if needed.",
            "",
            "In the next episode, you may be working with a different",
            "AI teammate or facing a different scenario.",
            "",
            "Remember to:",
            "• Use the control scheme that works best for you",
            "• Pay attention to how your AI teammate behaves",
            "• Focus on the mission objectives",
            "",
            "When you're ready to continue, press ENTER."
        ]

        self.draw_text_block(window, instructions, 350, self.font_medium, 35)


class WorkloadSurveyScreen(InstructionalScreen):
    """NASA-TLX style workload survey screen"""

    def __init__(self, episode_config: str = "", admin=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_config = episode_config

        # Survey questions and their labels
        self.questions = [
            "How MENTALLY demanding was the task?",
            #"How PHYSICALLY demanding was the task?",
            "How much TIME PRESSURE did you feel?",
            "How much EFFORT did the task take?",
            "How would you rate your PERFORMANCE?",
            "How much FRUSTRATION did you feel?"
        ]

        # User responses (1-7 scale, None = not answered)
        self.responses = {question: None for question in self.questions}

        # Visual properties for rating bars
        self.bar_width = 600
        self.bar_height = 50
        self.segment_width = self.bar_width // 7
        self.bar_start_x = 200#(self.window_width - self.bar_width) // 2
        self.bar_spacing = 135
        self.first_bar_y = 245

        # Colors
        self.unselected_color = (100, 100, 100)
        self.selected_color = (100, 150, 255)
        self.hover_color = (150, 150, 150)
        self.border_color = (200, 200, 200)
        self.admin = admin

        # Mouse interaction
        self.hover_segment = None  # (question_index, segment_index)

    def draw_content(self, window: pygame.Surface) -> None:
        # Title
        title = "Please answer these questions about the last round,"
        #if self.episode_config:
            #title = f"Please rate your workload in the last round:"
        self.draw_text_centered(window, title, 120, self.font_large)

        self.draw_text_centered(window, "with 1 being the lowest and 7 being the highest.", 155, self.font_large)

        # Draw rating bars
        for i, question in enumerate(self.questions):
            self.draw_rating_bar(window, i, question)

        # Draw continue button/instructions
        self.draw_continue_section(window)

    def draw_rating_bar(self, window: pygame.Surface, question_index: int, question_text: str) -> None:
        """Draw a single rating bar with 7 segments"""
        y_pos = self.first_bar_y + (question_index * self.bar_spacing)

        # Draw question label
        label_surface = self.font_large.render(question_text, True, self.text_color)
        window.blit(label_surface, (self.bar_start_x, y_pos - 40))

        # Draw each segment of the rating bar
        for segment in range(7):
            segment_x = self.bar_start_x + (segment * self.segment_width)
            segment_rect = pygame.Rect(segment_x, y_pos, self.segment_width, self.bar_height)

            # Determine segment color
            if self.responses[question_text] == segment + 1:
                # This segment is selected
                color = self.selected_color
            elif (self.hover_segment and
                  self.hover_segment[0] == question_index and
                  self.hover_segment[1] == segment):
                # This segment is being hovered
                color = self.hover_color
            else:
                # Unselected segment
                color = self.unselected_color

            # Fill segment
            pygame.draw.rect(window, color, segment_rect)

            # Draw border
            pygame.draw.rect(window, self.border_color, segment_rect, 2)

            # Draw segment number
            number_text = str(segment + 1)
            number_surface = self.font_large.render(number_text, True, self.text_color)
            number_rect = number_surface.get_rect(center=segment_rect.center)
            window.blit(number_surface, number_rect)

    def draw_continue_section(self, window: pygame.Surface) -> None:
        """Draw the continue button area"""
        continue_y = self.first_bar_y + (len(self.questions) * self.bar_spacing) + 0

        # Check if all questions are answered
        all_answered = all(response is not None for response in self.responses.values())

        if all_answered:
            # Draw enabled continue button
            arrow_text = "→"
            button_color = self.selected_color
            text_color = self.text_color
            instruction_text = "Click the arrows or press ENTER to continue"
        else:
            # Draw disabled continue button
            arrow_text = "→"
            button_color = self.unselected_color
            text_color = (100, 100, 100)
            unanswered_count = sum(1 for response in self.responses.values() if response is None)
            instruction_text = f"Please answer all questions ({unanswered_count} remaining)"

        # Draw arrow button
        button_size = 60
        button_x = self.window_width - 200
        button_y = continue_y - 50
        button_rect = pygame.Rect(button_x, button_y, button_size, button_size)

        pygame.draw.rect(window, button_color, button_rect, border_radius=10)
        pygame.draw.rect(window, self.border_color, button_rect, 3, border_radius=10)

        # Draw arrow
        arrow_surface = self.font_large.render(arrow_text, True, text_color)
        arrow_rect = arrow_surface.get_rect(center=button_rect.center)
        window.blit(arrow_surface, arrow_rect)

        # Draw instruction text
        instruction_surface = self.font_medium.render(instruction_text, True, text_color)
        instruction_rect = instruction_surface.get_rect(center=(self.window_width // 2, continue_y - 0))
        window.blit(instruction_surface, instruction_rect)

        # Store button rect for click detection
        self.continue_button_rect = button_rect if all_answered else None

    def handle_event(self, event: pygame.event.Event) -> Dict:
        """Handle mouse clicks and keyboard input"""
        if event.type == pygame.MOUSEMOTION:
            self.handle_mouse_hover(event.pos)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                clicked_result = self.handle_mouse_click(event.pos)
                if clicked_result:
                    return clicked_result

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                if self.all_questions_answered():
                    return {"action": "continue", "survey_data": self.get_survey_data()}
            elif event.key == pygame.K_ESCAPE and self.admin:
                return {"action": "exit"}

        return {"action": "none"}

    def handle_mouse_hover(self, mouse_pos: tuple) -> None:
        """Update hover state based on mouse position"""
        self.hover_segment = None

        for i, question in enumerate(self.questions):
            y_pos = self.first_bar_y + (i * self.bar_spacing)

            # Check if mouse is over this rating bar
            if (self.bar_start_x <= mouse_pos[0] <= self.bar_start_x + self.bar_width and
                    y_pos <= mouse_pos[1] <= y_pos + self.bar_height):

                # Determine which segment
                segment_index = (mouse_pos[0] - self.bar_start_x) // self.segment_width
                if 0 <= segment_index < 7:
                    self.hover_segment = (i, segment_index)
                break

    def handle_mouse_click(self, mouse_pos: tuple) -> Optional[Dict]:
        """Handle mouse clicks on rating bars and continue button"""
        # Check continue button click
        if (hasattr(self, 'continue_button_rect') and
                self.continue_button_rect and
                self.continue_button_rect.collidepoint(mouse_pos)):
            if self.all_questions_answered():
                return {"action": "continue", "survey_data": self.get_survey_data()}

        # Check rating bar clicks
        for i, question in enumerate(self.questions):
            y_pos = self.first_bar_y + (i * self.bar_spacing)

            # Check if click is within this rating bar
            if (self.bar_start_x <= mouse_pos[0] <= self.bar_start_x + self.bar_width and
                    y_pos <= mouse_pos[1] <= y_pos + self.bar_height):

                # Determine which segment was clicked
                segment_index = (mouse_pos[0] - self.bar_start_x) // self.segment_width
                if 0 <= segment_index < 7:
                    # Update response
                    self.responses[question] = segment_index + 1
                    print(f"Selected {question}: {segment_index + 1}")
                break

        return None

    def all_questions_answered(self) -> bool:
        """Check if all questions have been answered"""
        return all(response is not None for response in self.responses.values())

    def get_survey_data(self) -> Dict:
        """Get the survey responses in a structured format"""
        return {
            "episode_config": self.episode_config,
            "responses": self.responses.copy(),
            "timestamp": pygame.time.get_ticks()
        }

    def draw_navigation_hints(self, window: pygame.Surface) -> None:
        """Override to show custom navigation hints"""
        # Navigation hints are drawn in draw_continue_section instead
        pass


class TeammatePreferenceSurveyScreen(InstructionalScreen):
    """Survey screen for teammate preference questions with clickable icons"""

    def __init__(self, *args, agent_appearance=None, last_agent_appearance=None, admin=False, **kwargs):
        super().__init__(*args, **kwargs)

        # Survey questions
        self.questions = [
            "Which teammate was better at searching efficiently?",
            "Which teammate was better at coordinating with you?",
            "Which teammate did you prefer?"
        ]

        # User responses (None = not answered, 'green' or 'purple' for selection)
        self.responses = {i: None for i in range(len(self.questions))}

        # Visual properties
        self.icon_size = 80
        self.icon_spacing = 200
        self.question_spacing = 175 + 15
        self.first_question_y = 225 + 150 + 100 - 40

        # Colors
        self.unselected_color = (120, 120, 120)
        self.green_color = (76, 175, 80)
        self.purple_color = (156, 39, 176)
        self.selected_bg_color = (255, 255, 255)
        self.icon_bg_color = (120, 120, 120)#(180, 180, 180)
        self.admin = admin

        # Continue button
        self.continue_button_size = 80
        self.continue_button_pos = (self.window_width - 120, self.window_height - 120)

        # Mouse interaction
        self.hover_icon = None  # (question_index, icon_type)

        self.agent_appearance = agent_appearance
        self.last_agent_appearance = last_agent_appearance if last_agent_appearance is not None else 'green'

    def draw_content(self, window: pygame.Surface) -> None:
        # Title
        title = 'What did you think of the last two teammates you flew with?'
        self.draw_text_centered(window, title, 150, self.font_large)

        #self.draw_teammate_icon(window, 300, 300, self.agent_appearance, -1, True, self.icon_size * 2.2,scale=2.5)
        self.draw_teammate_icon(window, 300, 225, self.agent_appearance, -1, True, 0, scale=2.5)
        #self.draw_teammate_icon(window, 700, 300, self.last_agent_appearance, -1, True,self.icon_size * 2.2, scale=2.5)
        self.draw_teammate_icon(window, 700, 225, self.last_agent_appearance, -1, True, 0, scale=2.5)

        middle_text = "Answer each question below by clicking the boxes."

        self.draw_text_centered(window, middle_text, 340, self.font_large)

        # Draw each question with teammate icons
        for i, question in enumerate(self.questions):
            self.draw_question_with_icons(window, i, question)

        # Draw continue button
        self.draw_continue_button(window)

    def draw_question_with_icons(self, window: pygame.Surface, question_index: int, question_text: str) -> None:
        """Draw a question with two teammate icon options"""

        y_pos = self.first_question_y + (question_index * self.question_spacing)

        # Draw question text (handle multi-line)
        lines = question_text.split('\n')
        if len(lines) > 1:
            # Multi-line question
            for line_idx, line in enumerate(lines):
                line_y = y_pos - 30 + (line_idx * 30)
                self.draw_text_centered(window, line, line_y, self.font_large)
        else:
            # Single line question
            self.draw_text_centered(window, question_text, y_pos, self.font_large)

        # Calculate icon positions
        center_x = self.window_width // 2
        green_icon_x = center_x - self.icon_spacing // 2
        purple_icon_x = center_x + self.icon_spacing // 2
        icon_y = y_pos + 80

        # Draw green teammate icon
        #self.draw_teammate_icon(window, green_icon_x, icon_y, 'green', question_index, self.responses[question_index] == 'green')
        self.draw_teammate_icon(window, green_icon_x, icon_y, self.agent_appearance, question_index, self.responses[question_index] == self.agent_appearance, self.icon_size*1.5)

        # Draw purple teammate icon
        #self.draw_teammate_icon(window, purple_icon_x, icon_y, 'purple', question_index, self.responses[question_index] == 'purple')
        self.draw_teammate_icon(window, purple_icon_x, icon_y, self.last_agent_appearance, question_index,self.responses[question_index] == self.last_agent_appearance, self.icon_size*1.5)

    def draw_teammate_icon(self, window: pygame.Surface, x: int, y: int, icon_type: str,
                           question_index: int, is_selected: bool, icon_size, scale = 1.8) -> None:
        """Draw a teammate icon as a rotated and scaled-up mini aircraft rendering"""

        # Dummy env-like constants (scaled up 25%)
        #scale = 1.2
        NOSE = 10 * scale
        TAIL = 25 * scale
        WING = 18 * scale
        TAIL_WIDTH = 7 * scale
        LINE_WIDTH = int(math.floor(5 * scale))

        # 90 degrees counterclockwise = π/2
        direction = -math.pi / 2

        # Colors
        if icon_type == 'green': color = (76, 175, 80)
        elif icon_type == 'purple': color = (156, 39, 176)
        elif icon_type == 'red': color = (255, 25, 25)
        elif icon_type == 'brown': color = (255, 150, 0)
        elif icon_type == 'black': color = (30, 30, 30)
        else: color = (180, 180, 180)


        # Draw background box
        #icon_rect = pygame.Rect(x - self.icon_size // 2, y - self.icon_size // 2, self.icon_size, self.icon_size)
        icon_rect = pygame.Rect(x - icon_size // 2, y - icon_size // 2, icon_size, icon_size)
        bg_color = self.selected_bg_color if is_selected else self.icon_bg_color
        pygame.draw.rect(window, bg_color, icon_rect, border_radius=10)
        pygame.draw.rect(window, (100, 100, 100), icon_rect, 3, border_radius=10)

        # Aircraft points
        nose = (x + math.cos(direction) * NOSE, y + math.sin(direction) * NOSE)
        tail = (x - math.cos(direction) * TAIL, y - math.sin(direction) * TAIL)
        left_wing = (x - math.cos(direction - math.pi / 2) * WING,
                     y - math.sin(direction - math.pi / 2) * WING)
        right_wing = (x + math.cos(direction - math.pi / 2) * WING,
                      y + math.sin(direction - math.pi / 2) * WING)
        left_tail = (tail[0] - math.cos(direction - math.pi / 2) * TAIL_WIDTH,
                     tail[1] - math.sin(direction - math.pi / 2) * TAIL_WIDTH)
        right_tail = (tail[0] + math.cos(direction - math.pi / 2) * TAIL_WIDTH,
                      tail[1] + math.sin(direction - math.pi / 2) * TAIL_WIDTH)

        # Draw aircraft shape
        pygame.draw.line(window, color, tail, nose, LINE_WIDTH)
        pygame.draw.line(window, color, left_tail, right_tail, LINE_WIDTH)
        pygame.draw.line(window, color, left_wing, right_wing, LINE_WIDTH)

        # Appearance-specific markings
        if icon_type == 'purple':
            size = WING/1.5
            rect = pygame.Rect(0, 0, size, size)
            rect.center = (x, y)
            pygame.draw.rect(window, color, rect)
        elif icon_type == 'green':
            for pt in [left_wing, right_wing]:
                end = (pt[0] + math.cos(direction) * 15,
                       pt[1] + math.sin(direction) * 15)
                pygame.draw.line(window, color, pt, end, LINE_WIDTH)

        elif icon_type == 'red':
            for pt in [left_wing, right_wing]:
                end = (pt[0] - math.cos(direction) * 18,
                       pt[1] - math.sin(direction) * 18)
                pygame.draw.line(window, color, pt, end, LINE_WIDTH)

            perp_angle = direction + math.pi / 2
            start = (
                nose[0] - math.cos(perp_angle) * 4.5,
                nose[1] - math.sin(perp_angle) * 4.5
            )
            end = (
                nose[0] + math.cos(perp_angle) * 4.5,
                nose[1] + math.sin(perp_angle) * 4.5
            )
            pygame.draw.line(window, color, start, end, LINE_WIDTH)

        # Store for click detection
        if not hasattr(self, 'icon_rects'):
            self.icon_rects = {}
        self.icon_rects[(question_index, icon_type)] = icon_rect


    def draw_continue_button(self, window: pygame.Surface) -> None:
        """Draw the continue arrow button"""
        all_answered = all(response is not None for response in self.responses.values())

        # Button rectangle
        button_rect = pygame.Rect(self.continue_button_pos[0] - self.continue_button_size // 2,
                                  self.continue_button_pos[1] - self.continue_button_size // 2,
                                  self.continue_button_size, self.continue_button_size)

        # Colors based on state
        if all_answered:
            bg_color = (100, 150, 255)
            arrow_color = (255, 255, 255)
        else:
            bg_color = (100, 100, 100)
            arrow_color = (150, 150, 150)

        # Draw button background
        pygame.draw.rect(window, bg_color, button_rect, border_radius=10)
        pygame.draw.rect(window, (200, 200, 200), button_rect, 3, border_radius=10)

        # Draw arrow symbol
        arrow_font = pygame.font.SysFont('Arial', 36, bold=True)
        arrow_surface = arrow_font.render('→', True, arrow_color)
        arrow_rect = arrow_surface.get_rect(center=button_rect.center)
        window.blit(arrow_surface, arrow_rect)

        # Store button rect for click detection
        self.continue_button_rect = button_rect if all_answered else None

    def handle_event(self, event: pygame.event.Event) -> Dict:
        """Handle mouse clicks and keyboard input"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                clicked_result = self.handle_mouse_click(event.pos)
                if clicked_result:
                    return clicked_result

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                if self.all_questions_answered():
                    return {"action": "continue", "survey_data": self.get_survey_data()}
            elif event.key == pygame.K_ESCAPE and self.admin:
                return {"action": "exit"}

        return {"action": "none"}

    def handle_mouse_click(self, mouse_pos: tuple) -> Optional[Dict]:
        """Handle mouse clicks on icons and continue button"""
        # Check continue button click
        if (hasattr(self, 'continue_button_rect') and
                self.continue_button_rect and
                self.continue_button_rect.collidepoint(mouse_pos)):
            if self.all_questions_answered():
                return {"action": "continue", "survey_data": self.get_survey_data()}

        # Check icon clicks
        if hasattr(self, 'icon_rects'):
            for (question_index, icon_type), icon_rect in self.icon_rects.items():
                if icon_rect.collidepoint(mouse_pos):
                    # Update response
                    self.responses[question_index] = icon_type
                    print(f"Selected {icon_type} teammate for question {question_index + 1}")
                    break

        return None

    def all_questions_answered(self) -> bool:
        """Check if all questions have been answered"""
        return all(response is not None for response in self.responses.values())

    def get_survey_data(self) -> Dict:
        """Get the survey responses in a structured format"""
        return {
            "survey_type": "teammate_preference",
            "responses": {
                "performed_better": self.responses[0],
                "adapted_better": self.responses[1],
                "preferred_overall": self.responses[2]
            },
            "timestamp": pygame.time.get_ticks()
        }

    def draw_navigation_hints(self, window: pygame.Surface) -> None:
        """Override to show custom navigation hints"""
        # Navigation hints are drawn in draw_continue_button instead
        pass

# class FinalSummaryScreen(InstructionalScreen):
#     """Final screen shown after all episodes"""
#
#     def __init__(self, experiment_results: List[Dict], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.experiment_results = experiment_results
#
#     def draw_content(self, window: pygame.Surface) -> None:
#         # Title
#         self.draw_text_centered(window, "Experiment Complete!", 150,
#                                 self.font_large, self.highlight_color)
#
#         # Summary stats
#         if self.experiment_results:
#             total_reward = sum(r['reward'] for r in self.experiment_results)
#             avg_reward = total_reward / len(self.experiment_results)
#
#             self.draw_text_centered(window, f"Episodes Completed: {len(self.experiment_results)}",
#                                     220, self.font_medium)
#             self.draw_text_centered(window, f"Total Score: {total_reward:.2f}",
#                                     250, self.font_medium)
#             self.draw_text_centered(window, f"Average Score: {avg_reward:.2f}",
#                                     280, self.font_medium, (100, 255, 100))
#
#         # Thank you message
#         thank_you = [
#             "",
#             "Thank you for participating in this study!",
#             "",
#             "Your data will help us understand how humans and AI",
#             "can work together more effectively in complex tasks.",
#             "",
#             "If you have any questions about the study, please",
#             "ask the experimenter.",
#             "",
#             "Press ENTER to exit."
#         ]
#
#         self.draw_text_block(window, thank_you, 350, self.font_medium, 35)


class GameInstructionScreen(InstructionalScreen):
    """Base class for game instruction screens with navigation"""

    #def __init__(self, screen_number: int, total_screens: int, admin=False, *args, **kwargs):
    def __init__(self, screen_number: int, total_screens: int, *args, admin=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.screen_number = screen_number
        self.total_screens = total_screens
        self.admin = admin

        # Arrow button properties
        self.arrow_button_size = 60
        self.arrow_button_margin = 30
        self.arrow_color = (100, 150, 255)
        self.arrow_hover_color = (120, 170, 255)
        self.arrow_disabled_color = (100, 100, 100)

        # Mouse hover state
        self.hover_next = False
        self.hover_prev = False

    def draw_navigation_hints(self, window: pygame.Surface) -> None:
        """Draw navigation arrows and hints"""
        # Draw arrow buttons
        self.draw_arrow_buttons(window)

        # Draw text hint
        if self.screen_number < self.total_screens:
            hint_text = "Click the arrows or use arrow keys to continue"
        else:
            hint_text = "Click the arrows or use arrow keys to continue"

        hint_surface = self.font_medium.render(hint_text, True, (150, 150, 150))
        hint_rect = hint_surface.get_rect(center=(self.window_width // 2, self.window_height - 90))
        window.blit(hint_surface, hint_rect)

    def draw_arrow_buttons(self, window: pygame.Surface) -> None:
        """Draw clickable arrow buttons"""
        button_y = self.window_height - 120

        # Previous arrow (left)
        if self.screen_number > 1:
            prev_button_x = self.arrow_button_margin
            prev_color = self.arrow_hover_color if self.hover_prev else self.arrow_color
            self.prev_button_rect = pygame.Rect(prev_button_x, button_y, self.arrow_button_size, self.arrow_button_size)

            pygame.draw.rect(window, prev_color, self.prev_button_rect, border_radius=10)
            pygame.draw.rect(window, (200, 200, 200), self.prev_button_rect, 3, border_radius=10)

            # Draw left arrow
            arrow_font = pygame.font.SysFont('Arial', 36, bold=True)
            arrow_surface = arrow_font.render('←', True, (255, 255, 255))
            arrow_rect = arrow_surface.get_rect(center=self.prev_button_rect.center)
            window.blit(arrow_surface, arrow_rect)
        else:
            self.prev_button_rect = None

        # Next arrow (right)
        if self.screen_number < self.total_screens:
            next_button_x = self.window_width - self.arrow_button_margin - self.arrow_button_size
            next_color = self.arrow_hover_color if self.hover_next else self.arrow_color
            self.next_button_rect = pygame.Rect(next_button_x, button_y, self.arrow_button_size, self.arrow_button_size)

            pygame.draw.rect(window, next_color, self.next_button_rect, border_radius=10)
            pygame.draw.rect(window, (200, 200, 200), self.next_button_rect, 3, border_radius=10)

            # Draw right arrow
            arrow_font = pygame.font.SysFont('Arial', 36, bold=True)
            arrow_surface = arrow_font.render('→', True, (255, 255, 255))
            arrow_rect = arrow_surface.get_rect(center=self.next_button_rect.center)
            window.blit(arrow_surface, arrow_rect)
        elif self.screen_number == self.total_screens:
            # Continue button on final screen
            next_button_x = self.window_width - self.arrow_button_margin - self.arrow_button_size
            next_color = self.arrow_hover_color if self.hover_next else self.arrow_color
            self.next_button_rect = pygame.Rect(next_button_x, button_y, self.arrow_button_size, self.arrow_button_size)

            pygame.draw.rect(window, next_color, self.next_button_rect, border_radius=10)
            pygame.draw.rect(window, (200, 200, 200), self.next_button_rect, 3, border_radius=10)

            # Draw checkmark or continue symbol
            arrow_font = pygame.font.SysFont('Arial', 36, bold=True)
            arrow_surface = arrow_font.render('→', True, (255, 255, 255))
            arrow_rect = arrow_surface.get_rect(center=self.next_button_rect.center)
            window.blit(arrow_surface, arrow_rect)
        else:
            self.next_button_rect = None

    def handle_event(self, event: pygame.event.Event) -> Dict:
        """Handle navigation between screens"""
        if event.type == pygame.MOUSEMOTION:
            self.handle_mouse_hover(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                clicked_result = self.handle_mouse_click(event.pos)
                if clicked_result:
                    return clicked_result
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and self.screen_number < self.total_screens + 1:
                return {"action": "next"}
            elif event.key == pygame.K_LEFT and self.screen_number > 1:
                return {"action": "previous"}
            elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                return {"action": "continue"}
            elif event.key == pygame.K_ESCAPE and self.admin:
                return {"action": "exit"}
        return {"action": "none"}

    def handle_mouse_hover(self, mouse_pos: tuple) -> None:
        """Update hover state based on mouse position"""
        self.hover_prev = False
        self.hover_next = False

        if hasattr(self, 'prev_button_rect') and self.prev_button_rect and self.prev_button_rect.collidepoint(
                mouse_pos):
            self.hover_prev = True
        elif hasattr(self, 'next_button_rect') and self.next_button_rect and self.next_button_rect.collidepoint(
                mouse_pos):
            self.hover_next = True

    def handle_mouse_click(self, mouse_pos: tuple) -> Optional[Dict]:
        """Handle mouse clicks on arrow buttons"""
        if hasattr(self, 'prev_button_rect') and self.prev_button_rect and self.prev_button_rect.collidepoint(
                mouse_pos):
            return {"action": "previous"}
        elif hasattr(self, 'next_button_rect') and self.next_button_rect and self.next_button_rect.collidepoint(
                mouse_pos):
            if self.screen_number < self.total_screens:
                return {"action": "next"}
            else:
                return {"action": "continue"}
        return None


class Instruct1Screen(GameInstructionScreen):
    """Welcome screen"""

    def draw_content(self, window: pygame.Surface) -> None:
        body_text = ["Welcome to our study!",
                     "",
                     "",
                     "Today you will play a 2-dimensional video game.",
                     "",
                     "",
                     "You will work alongside AI teammates that are",
                     "trained using reinforcement learning.",
                     "",
                     "",
                     "This study will take about 30 minutes."
                     ]

        self.draw_text_block(window, body_text, 250, self.font_large, 40)

# class Instruct2Screen(GameInstructionScreen):
#     """Map introduction screen with looping video"""
#
#     def __init__(self, video_path: str, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.video_path = video_path
#         self.cap = cv2.VideoCapture(video_path)
#         if not self.cap.isOpened():
#             print(f"Error opening video file: {video_path}")
#         self.last_frame_time = pygame.time.get_ticks()
#         self.frame_interval = int(1000 / self.cap.get(cv2.CAP_PROP_FPS))
#
#         self.current_frame = None  # Store last loaded frame surface
#
#     def get_next_video_frame(self):
#         if not self.cap.isOpened():
#             return None
#
#         current_time = pygame.time.get_ticks()
#         if current_time - self.last_frame_time >= self.frame_interval:
#             ret, frame = self.cap.read()
#             if not ret:
#                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
#                 ret, frame = self.cap.read()
#             if ret:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = cv2.resize(frame, (600, 600))
#                 frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
#                 self.current_frame = frame_surface
#                 self.last_frame_time = current_time
#
#         return self.current_frame
#
#     def draw_content(self, window: pygame.Surface) -> None:
#         # Draw text
#         lines = [
#             "In this game, you will control a 2D aircraft",
#             "to fly around a map like the one below:"
#         ]
#         y_pos = 150
#         for line in lines:
#             self.draw_text_centered(window, line, y_pos, self.font_large)
#             y_pos += 40
#
#         image_rect = self.image.get_rect(center=(self.window_width // 2, 50 + self.window_width // 2))
#         window.blit(self.image, image_rect)
#
#         # Draw video frame
#         frame_surface = self.get_next_video_frame()
#         if frame_surface:
#             rect = frame_surface.get_rect(center=(self.window_width // 2, 600))
#             window.blit(frame_surface, rect)
#         else:
#             # Fallback placeholder
#             pygame.draw.rect(window, (100, 100, 100), (200, 250, 600, 600))

class Instruct2Screen(GameInstructionScreen):
    """Map introduction screen"""

    def __init__(self, image_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_path = image_path
        self.image = None
        try:
            self.image = pygame.image.load(image_path)
            self.image = pygame.transform.scale(self.image, (650, 650))
        except pygame.error:
            print(f"Could not load image: {image_path}")

    def draw_content(self, window: pygame.Surface) -> None:
        # Top text at position (100, 50)
        # Draw text
        lines = [
            "In this game, you will control a 2D",
            "aircraft to fly around the map below:"
        ]
        y_pos = 150
        for line in lines:
            self.draw_text_centered(window, line, y_pos, self.font_large)
            y_pos += 40

        # Draw image centered at 500x500 if available
        if self.image:
            image_rect = self.image.get_rect(center=(self.window_width // 2, 100 + self.window_width // 2))
            window.blit(self.image, image_rect)
        else:
            # Draw placeholder rectangle
            placeholder_rect = pygame.Rect(250, 150, 500, 500)
            pygame.draw.rect(window, (100, 100, 100), placeholder_rect)
            pygame.draw.rect(window, self.text_color, placeholder_rect, 2)
            placeholder_text = "Map Image (500x500)"
            text_surface = self.font_medium.render(placeholder_text, True, self.text_color)
            text_rect = text_surface.get_rect(center=placeholder_rect.center)
            window.blit(text_surface, text_rect)


class Instruct3Screen(GameInstructionScreen):
    """Player aircraft control introduction"""

    def __init__(self, image_path: str, video_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_path = image_path
        self.video_path = video_path
        self.image = None

        try:
            self.image = pygame.image.load(image_path)
            self.image = pygame.transform.scale(self.image, (200, 200))
        except pygame.error:
            print(f"Could not load image: {image_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error opening video file: {video_path}")
        self.last_frame_time = pygame.time.get_ticks()
        self.frame_interval = int(1000 / self.cap.get(cv2.CAP_PROP_FPS))
        self.current_frame = None

    def get_next_video_frame(self):
        if not self.cap.isOpened():
            return None
        current_time = pygame.time.get_ticks()
        if current_time - self.last_frame_time >= self.frame_interval:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (563, 300))
                self.current_frame = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                self.last_frame_time = current_time
        return self.current_frame

    def draw_content(self, window: pygame.Surface) -> None:
        self.draw_text_centered(window, "You control the BLUE aircraft.", 150, self.font_large)

        if self.image:
            image_rect = self.image.get_rect(center=(self.window_width // 2, 300))
            window.blit(self.image, image_rect)

        middle_text = ["You control your aircraft by clicking",
                       "on the map where you want to fly."]
        y_pos = 460
        for line in middle_text:
            self.draw_text_centered(window, line, y_pos, self.font_large)
            y_pos += 35

        # Draw video just below middle text
        frame_surface = self.get_next_video_frame()
        if frame_surface:
            rect = frame_surface.get_rect(center=(self.window_width // 2, 700))
            window.blit(frame_surface, rect)
        else:
            pygame.draw.rect(window, (100, 100, 100), (200, 600, 600, 400))

        bottom_text = ["The aircraft will automatically",
                       "fly to the point you clicked."]
        y_pos = 890
        for line in bottom_text:
            self.draw_text_centered(window, line, y_pos, self.font_large)
            y_pos += 35


# class Instruct3Screen(GameInstructionScreen):
#     """Player aircraft control introduction"""
#
#     def __init__(self, image_path: str, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.image_path = image_path
#         self.image = None
#         try:
#             self.image = pygame.image.load(image_path)
#             self.image = pygame.transform.scale(self.image, (200, 200))
#         except pygame.error:
#             print(f"Could not load image: {image_path}")
#
#     def draw_content(self, window: pygame.Surface) -> None:
#
#         # Top text at (100, 100)
#         #text_surface = self.font_large.render("You control the BLUE aircraft.", True, self.text_color)
#         #window.blit(text_surface, (100, 150))
#         self.draw_text_centered(window, "You control the BLUE aircraft.", 150, self.font_large)
#
#         # Blue circle at (500, 200) with radius 50px
#         image_rect = self.image.get_rect(center=(self.window_width // 2, 300))
#         window.blit(self.image, image_rect)
#         #pygame.draw.circle(window, (0, 100, 255), (500, 300), 75)
#
#         # Bottom text at (50, 500)
#         middle_text = ["You control your aircraft by clicking",
#                        "on the map where you want to fly."]
#
#         bottom_text = ["The aircraft will automatically",
#                        "fly to the point you clicked."]
#
#         y_pos = 500
#         for line in middle_text:
#             self.draw_text_centered(window, line, y_pos, self.font_large)
#             y_pos += 35
#
#         y_pos = 850
#         for line in bottom_text:
#             self.draw_text_centered(window, line, y_pos, self.font_large)
#             y_pos += 35


class Instruct4Screen(GameInstructionScreen):
    """AI teammate introduction"""

    def __init__(self, image_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_path = image_path
        self.image = None
        try:
            self.image = pygame.image.load(image_path)
            self.image = pygame.transform.scale(self.image, (855*.8*.9, 200*1.2*.9))
        except pygame.error:
            print(f"Could not load image: {image_path}")

    def draw_content(self, window: pygame.Surface) -> None:
        # Top text at (100, 100)

        top_text = ["In each level, you will be assisted by",
                    "an AI teammate selected from a pool."]

        y_pos = 150
        for line in top_text:
            self.draw_text_centered(window, line, y_pos, self.font_large)
            y_pos += 35

        # Red circle at (500, 200) with radius 50px (AI teammate)
        image_rect = self.image.get_rect(center=(self.window_width // 2, 350))
        window.blit(self.image, image_rect)

        # Bottom text at (50, 500)
        bottom_text = ["",
                       "Your teammate will independently fly",
                       "around the map to identify targets.",
                       "",
                       "",
                       "All teammates are trained with the same goals as",
                       "you (15 regular + 2 high-value targets), but they",
                       "might do better or worse depending on their training.",
                       "",
                       "",
                       "You will need to coordinate with your teammate to succeed.",
                       "Try to search the map efficiently and balance your risk.",
                       "",
                       "Note: you can fly as close to your teammate as you want.",
                       "There is no risk of collision."
                       ]

        y_pos = 450
        for line in bottom_text:
            self.draw_text_centered(window, line, y_pos, self.font_large)
            y_pos += 35


class Instruct5Screen(GameInstructionScreen):
    """Regular targets explanation"""

    def __init__(self, sensor_image_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_image_path = sensor_image_path
        self.sensor_image = None
        try:
            self.sensor_image = pygame.image.load(sensor_image_path)
            # Scale to width 600px while maintaining aspect ratio
            original_size = self.sensor_image.get_size()
            scale_factor = 600 / original_size[0]
            new_height = int(original_size[1] * scale_factor)
            self.sensor_image = pygame.transform.scale(self.sensor_image, (600, new_height))
        except pygame.error:
            print(f"Could not load image: {sensor_image_path}")

    def draw_content(self, window: pygame.Surface) -> None:

        top_text = ["You earn points by identifying",
                    "targets on the map."]

        y_pos = 150
        for line in top_text:
            self.draw_text_centered(window, line, y_pos, self.font_large)
            y_pos += 35

        # Gold circle at (100, 200) with radius 10px
        #pygame.draw.circle(window, (255, 215, 0), (200, 330), 15) # TODO render dark gold outline around circle
        pygame.draw.circle(window, (139, 117, 0), (200, 330), 17)  # dark gold outline
        pygame.draw.circle(window, (255, 215, 0), (200, 330), 14)  # fill


        # Text at (150, 200)
        text_surface = self.font_large.render("There are 15 REGULAR targets on the map.", True, self.text_color)
        window.blit(text_surface, (300, 320))

        # Gold circle
        #pygame.draw.circle(window, (255, 215, 0), (200, 450), 50) # TODO change this to be a white circle with a gold outline, and an upside down gold triangle inside it
        # White circle with gold outline
        pygame.draw.circle(window, (255, 215, 0), (200, 450), 52)
        pygame.draw.circle(window, (0, 0, 0), (200, 450), 48)


        #triangle_points = [(195, 465), (185, 440), (220, 435)]
        cx, cy = 200, 450
        side = 40  # side length of the triangle
        h = (3 ** 0.5 / 2) * side  # height of the equilateral triangle

        triangle_points = [
            (cx - side / 2, cy - h / 3),  # bottom-left
            (cx + side / 2, cy - h / 3),  # bottom-right
            (cx, cy + 2 * h / 3)  # top (pointing down)
        ]
        pygame.draw.polygon(window, (255, 215, 0), triangle_points) # Upside down gold triangle inside


        text_surface = self.font_large.render("There are 4 HIGH-VALUE targets on the map.", True, self.text_color)
        window.blit(text_surface, (300, 430))

        text_surface = self.font_large.render("To be identified, the high value target's", True, self.text_color)
        window.blit(text_surface, (300, 465))

        text_surface = self.font_large.render("TRIANGLE must enter your sensor range", True,self.text_color)
        window.blit(text_surface, (300, 465+35))
        #
        # bottom_text = ["Each level has 15 regular",
        #                "targets and 4 high-value targets."]
        # y_pos = 650
        # for line in bottom_text:
        #     self.draw_text_centered(window, line, y_pos, self.font_large)
        #     y_pos += 35


class Instruct6Screen(GameInstructionScreen):
    def __init__(self, hvt_video_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap = cv2.VideoCapture(hvt_video_path)
        if not self.cap.isOpened():
            print(f"Error opening video file: {hvt_video_path}")
        self.last_frame_time = pygame.time.get_ticks()
        self.frame_interval = int(1000 / self.cap.get(cv2.CAP_PROP_FPS))
        self.current_frame = None

    def get_next_video_frame(self):
        if not self.cap.isOpened():
            return None
        current_time = pygame.time.get_ticks()
        if current_time - self.last_frame_time >= self.frame_interval:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (450, 450))
                self.current_frame = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                self.last_frame_time = current_time
        return self.current_frame

    def draw_content(self, window: pygame.Surface) -> None:
        instruction_text = [
            "To identify a target, you must",
            "fly close enough that your",
            "sensor overlaps the target."
        ]
        y_pos = 150
        for line in instruction_text:
            self.draw_text_centered(window, line, y_pos, self.font_large)
            y_pos += 35

        frame_surface = self.get_next_video_frame()
        if frame_surface:
            rect = frame_surface.get_rect(center=(self.window_width // 2, 600))
            window.blit(frame_surface, rect)
        else:
            pygame.draw.rect(window, (100, 100, 100), (200, 300, 600, 400))


# class Instruct6Screen(GameInstructionScreen):
#     """High-value targets explanation"""
#
#     def __init__(self, hvt_image_path: str, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.hvt_image_path = hvt_image_path
#         self.hvt_image = None
#         try:
#             self.hvt_image = pygame.image.load(hvt_image_path)
#             # Scale to width 600px while maintaining aspect ratio
#             original_size = self.hvt_image.get_size()
#             scale_factor = 600 / original_size[0]
#             new_height = int(original_size[1] * scale_factor)
#             self.hvt_image = pygame.transform.scale(self.hvt_image, (600, new_height))
#         except pygame.error:
#             print(f"Could not load image: {hvt_image_path}")
#
#     def draw_content(self, window: pygame.Surface) -> None:
#         # # Gold circle at (100, 200) with radius 50px
#         # pygame.draw.circle(window, (255, 215, 0), (200, 200), 50)
#         #
#         # text_surface = self.font_large.render("High-value targets are worth 9 points.", True, self.text_color)
#         # window.blit(text_surface, (300, 200))
#
#         # Text at (100, 300)
#         instruction_text = ["To identify a target, you must",
#                             "fly close enough that your",
#                             "sensor overlaps the target."]
#         y_pos = 150
#         for line in instruction_text:
#             self.draw_text_centered(window, line, y_pos, self.font_large)
#             y_pos += 35
#


class Instruct7Screen(GameInstructionScreen):
    def __init__(self, detection_video_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap = cv2.VideoCapture(detection_video_path)
        if not self.cap.isOpened():
            print(f"Error opening video file: {detection_video_path}")
        self.last_frame_time = pygame.time.get_ticks()
        self.frame_interval = int(1000 / self.cap.get(cv2.CAP_PROP_FPS))
        self.current_frame = None

    def get_next_video_frame(self):
        if not self.cap.isOpened():
            return None
        current_time = pygame.time.get_ticks()
        if current_time - self.last_frame_time >= self.frame_interval:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (550, 550))
                self.current_frame = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                self.last_frame_time = current_time
        return self.current_frame

    def draw_content(self, window: pygame.Surface) -> None:
        # warning_text = ["High-value targets might detect you if you fly",
        #                 "within identification range. Each detection",
        #                 "reduces your score by 15 points."]

        top_text = ["Your goal is to identify all 15 targets",
                    "and exactly 2 high-value targets.",
                    "",
                    "",
                    "You have about 75 seconds per round.",
                    "",
                    "",
                    "If you identify more or less than 2 high-",
                    "value targets, your score will be reduced.",
                    "",
                    "",
                    "If you complete your goals before the round timer",
                    "ends, you will receive a bonus for finishing early.",
                    "",
                    "",
                    "Your final score is calculated as:",
                    "",
                    "5 × (regular targets) - 10 × abs(2 - high value targets)",
                    "+ 1.25 × (# of seconds early)",
                    "",
                    "",
                    "Good luck!"
                    ]

        y_pos = 150
        for line in top_text:
            self.draw_text_centered(window, line, y_pos, self.font_large)
            y_pos += 35

        # frame_surface = self.get_next_video_frame()
        # if frame_surface:
        #     rect = frame_surface.get_rect(center=(self.window_width // 2, 550))
        #     window.blit(frame_surface, rect)
        #
        # bottom_text = ["To be successful, you will need to weigh the risks",
        #                "and rewards of identifying high-value targets."]
        # y_pos = 875
        # for line in bottom_text:
        #     self.draw_text_centered(window, line, y_pos, self.font_large)
        #     y_pos += 35

class SecondPracticeIntroScreen(GameInstructionScreen):
    """Screen shown between the two practice rounds with navigation arrows"""

    def __init__(self, window_width: int = 1000, window_height: int = 1100, sio=None):
        # Set screen_number=1 and total_screens=1 since it's a single page
        super().__init__(screen_number=1, total_screens=1,
                         window_width=window_width, window_height=window_height, sio=sio)

    def draw_content(self, window: pygame.Surface) -> None:
        top_text = [
            "Nice job! Let's do one more practice",
            "round just to get more familiar."
        ]

        y_pos = 350
        for line in top_text:
            self.draw_text_centered(window, line, y_pos, self.font_large)
            y_pos += 35

        # if self.sio is not None:
        #         self.sio.send_frame(window)

class InterScreen(GameInstructionScreen):
    """Screen shown between the two practice rounds with navigation arrows"""

    def __init__(self, window_width: int = 1000, window_height: int = 1100, sio=None):
        # Set screen_number=1 and total_screens=1 since it's a single page
        super().__init__(screen_number=1, total_screens=1,
                         window_width=window_width, window_height=window_height, sio=sio)

    def draw_content(self, window: pygame.Surface) -> None:
        top_text = [
            "Nice job! Click the right arrow to start the next round.",
        ]

        y_pos = 300
        for line in top_text:
            self.draw_text_centered(window, line, y_pos, self.font_large)
            y_pos += 35


# class Instruct7Screen(GameInstructionScreen):
#     """Final instructions screen"""
#
#
#     def draw_content(self, window: pygame.Surface) -> None:
#
#         # Text at (250, 350) - wrap text
#         warning_text = ["High-value targets might detect you if you fly",
#                         "within identification range. Each detection",
#                         "reduces your score by 15 points."]
#         y_pos = 150
#         for line in warning_text:
#             self.draw_text_centered(window, line, y_pos, self.font_large)
#             y_pos += 35
#
#         # TODO add video loaded from /img/detection_video.mp4
#
#         # Bottom text at (100, 700)
#         bottom_text = ["To be successful, you will need to weigh the risks",
#                        "and rewards of identifying high-value targets."]
#
#         y_pos = 750
#         for line in bottom_text:
#             self.draw_text_centered(window, line, y_pos, self.font_large)
#             y_pos += 35


class Instruct8Screen(GameInstructionScreen):
    """Final instructions screen"""

    def draw_content(self, window: pygame.Surface) -> None:
        instruction_text = [
            "You will complete 14 rounds in the game.",
            "",
            "Each round is approximately 1 minute long.",
            "",
            "",
            "After each round, you will answer",
            "a few questions about the teammate",
            "you just worked with.",
            "",
            "",
            "Please try to remember how each",
            "teammate looked, and how they performed!",
            "",
            "",
            "We are developing AI teammates that can adapt",
            "to humans. Your responses help us identify",
            "which techniques are most effective!"
        ]

        self.draw_text_block(window, instruction_text, 200, self.font_large, 40)


class InstructTLXScreen(GameInstructionScreen):
    """Instruction screen explaining the NASA-TLX workload survey questions"""

    def draw_content(self, window: pygame.Surface) -> None:
        title_text = "Understanding the Workload Survey (NASA-TLX)"
        self.draw_text_centered(window, title_text, 120, self.font_large, self.highlight_color)

        explanation_lines = [
            "After some rounds, you will answer 5 questions:",
            "",
            "1. MENTAL DEMAND – How mentally challenging was the round?",
            "",
            "2. TIME PRESSURE – Did you feel rushed to finish the round?",
            "",
            "3. EFFORT – How much effort did it take to perform well?",
            "",
            "4. PERFORMANCE – How well do you think you performed?",
            "",
            "5. FRUSTRATION – How stressed or annoyed did you feel?",
            "",
            "",
            "You will rate each from 1 (very low) to 7 (very high).",
            "Please rate each round relative to the others."
        ]

        self.draw_text_block(window, explanation_lines, 220, self.font_large, 38)



class PlaceholderScreen(GameInstructionScreen):
    """Placeholder screen for later editing"""
    def draw_content(self, window):
        self.draw_text_centered(window, "[ The Scoring System ]", 250, self.font_large)
        placeholder_lines = [
            "Your performance is tracked using a score system:",
            "",
            ""
        ]
        self.draw_text_block(window, placeholder_lines, 400, self.font_medium, 40)

class AfterPracticeScreen(GameInstructionScreen):
    """Placeholder screen for later editing"""
    def __init__(self, window_width: int, window_height: int, sio=None):
        super().__init__(screen_number=1, total_screens=1,
                         window_width=window_width, window_height=window_height)

    def draw_content(self, window):
        self.draw_text_centered(window, "", 250, self.font_large)
        placeholder_lines = [
            "Nice work! Now it's time to play for real.",
            "",
            "When you're ready, click the arrow to start the first round."
        ]
        self.draw_text_block(window, placeholder_lines, 400, self.font_large, 40)

        if self.sio is not None:
            #import sockets
            self.sio.send_frame(window)

class BeforeSoloScreen(GameInstructionScreen):
    """Placeholder screen for later editing"""
    def __init__(self, window_width: int, window_height: int, sio=None):
        super().__init__(screen_number=1, total_screens=1,
                         window_width=window_width, window_height=window_height)
        self.sio = sio
        self.second = False

    def draw_content(self, window):
        self.draw_text_centered(window, "", 250, self.font_large)
        if self.second is False:
            placeholder_lines = [
                "Now you'll complete two solo rounds, without a teammate.",
                "",
                "After these rounds, you'll continue playing with teammates."
            ]
        else:
            placeholder_lines = [
                "The next round will be solo as well.",
                "",
                "After this round, you'll continue playing with teammates."
            ]
        self.draw_text_block(window, placeholder_lines, 400, self.font_large, 40)

        # if self.sio is not None:
        #     #import sockets
        #     self.sio.send_frame(window)


class PracticeIntroScreen(GameInstructionScreen):
    """Instructional screen before the practice round"""
    def draw_content(self, window):
        self.draw_text_centered(window, "Practice Round", 150, self.font_large, self.highlight_color)
        intro_lines = [
            "You will now play a practice round.",
            "",
            "",
            "This is just for you to get used to the controls and",
            "gameplay. There is no survey after this round.",
            "",
            "",
            "You will also play this round solo, without a teammate.",
            #"In the practice round, you will play alongside an agent who",
            #"is still learning. It may not do as well as the other agents!"
            "",
            "",
            "When you're ready, click the arrow to begin.",
            "Please only click once, and the level will begin loading."

        ]
        self.draw_text_block(window, intro_lines, 250, self.font_large, 40)

class FinalSummaryScreen(InstructionalScreen):
    """Final screen shown after completing the experiment"""

    def __init__(self, experiment_results, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_results = experiment_results

    def draw_content(self, window):
        self.draw_text_centered(window, "Thank You!", 150, self.font_large, self.highlight_color)

        if self.experiment_results:
            #total_reward = sum(r['reward'] for r in self.experiment_results)
            #avg_reward = total_reward / len(self.experiment_results)

            self.draw_text_centered(window, f"Rounds Completed: {len(self.experiment_results)}", 230, self.font_large)
            #self.draw_text_centered(window, f"Total Score: {total_reward:.2f}", 270, self.font_large)
            #self.draw_text_centered(window, f"Average Score: {avg_reward:.2f}", 310, self.font_large, (100, 255, 100))

        message_lines = [
            "",
            "Thank you for participating in this user study!",
            "",
            "",
            "Your participation helps us improve",
            "human-AI teaming in critical tasks.",
            "",
            "",
            "Please tell the researcher that you are done.",
            "",
            "If you have questions or feedback, please ask the researcher.",
            "",
            "",
            "",
        ]
        self.draw_text_block(window, message_lines, 380, self.font_large, 35)


class InstructionSeriesManager:
    """Manages the series of instruction screens"""

    def __init__(self, window: pygame.Surface, clock: pygame.time.Clock,
                 map_image_path: str = None, sensor_image_path: str = None,
                 hvt_video_path: str = None, detection_video_path: str = None,
                 click_video_path: str = None,
                 human_image_path: str = None, teammate_image_path: str = None, sio = None, admin=False):
        self.window = window
        self.clock = clock
        self.screen_manager = ScreenManager(window, clock)
        self.sio = sio
        self.admin = admin

        total_screens = 10

        self.screens = [
            Instruct1Screen(1, total_screens, window.get_width(), window.get_height()),
            Instruct2Screen(map_image_path or "map_image.jpg", 2, total_screens, window.get_width(),
                            window.get_height()),
            Instruct3Screen(human_image_path, click_video_path or "img/click_control.mp4", 3, total_screens,
                            window.get_width(), window.get_height()),
            Instruct5Screen(sensor_image_path or "sensor_image.jpg", 4, total_screens, window.get_width(),
                            window.get_height()),
            Instruct6Screen(hvt_video_path or "img/target_id_video.mp4", 5, total_screens, window.get_width(),
                            window.get_height()),
            Instruct7Screen(detection_video_path or "img/detection_video.mp4", 6, total_screens, window.get_width(),
                            window.get_height()),
            Instruct4Screen(teammate_image_path, 7, total_screens, window.get_width(), window.get_height()),
            Instruct8Screen(8, total_screens, window.get_width(), window.get_height()),
            InstructTLXScreen(9, total_screens, window.get_width(), window.get_height()),  # <-- New screen
            PracticeIntroScreen(10, total_screens, window.get_width(), window.get_height()),
        ]

        self.current_screen_index = 0

    def run_instruction_series(self) -> Dict:
        """Run through all instruction screens with proper event handling"""
        while 0 <= self.current_screen_index < len(self.screens):
            current_screen = self.screens[self.current_screen_index]
            if self.sio is not None:
                #import sockets
                self.sio.instruction_controller = current_screen
                self.sio.pyg = pygame.event

            # Run the screen manually to ensure proper event handling
            running = True
            result = {"action": "none"}

            while running:
                dt = self.screen_manager.clock.tick(FPS)

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT and self.admin:
                        result = {"action": "exit"}
                        running = False
                    else:
                        event_result = current_screen.handle_event(event)
                        if event_result["action"] != "none":
                            result = event_result
                            running = False

                # Update screen (for hover effects)
                current_screen.update(dt)

                # Render
                current_screen.render(self.screen_manager.window)
                pygame.display.flip()
                if self.sio is not None:
                    self.sio.send_frame(self.window)

            # Process the result
            if result["action"] == "next":
                self.current_screen_index += 1
            elif result["action"] == "previous":
                self.current_screen_index -= 1
            elif result["action"] == "continue":
                if self.current_screen_index == len(self.screens) - 1:
                    # Last screen, finish instructions
                    return {"action": "complete"}
                else:
                    # Continue to next screen
                    self.current_screen_index += 1
            elif result["action"] == "exit":
                return {"action": "exit"}

        return {"action": "complete"}


class ScreenManager:
    """Manages the display and flow of instructional screens"""

    def __init__(self, window: pygame.Surface, clock: pygame.time.Clock):
        self.window = window
        self.clock = clock
        self.current_screen = None

    def show_screen(self, screen: InstructionalScreen,
                    on_continue: Callable = None,
                    on_exit: Callable = None) -> Dict:
        """
        Display a screen and handle user interaction.

        Args:
            screen: The InstructionalScreen to display
            on_continue: Callback function when user continues
            on_exit: Callback function when user exits

        Returns:
            Dict with action result and any collected data
        """
        self.current_screen = screen
        running = True
        result = {"action": "none", "text_input": ""}

        while running:
            dt = self.clock.tick(FPS)


            import sockets
            sockets.send_frame(self.window)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    result = {"action": "exit"}
                    running = False
                else:
                    event_result = screen.handle_event(event)
                    if event_result["action"] != "none":
                        result = event_result
                        if event_result["action"] in ["continue", "exit"]:
                            running = False

            # Update screen
            screen.update(dt)

            # Render
            screen.render(self.window)
            pygame.display.flip()

        # Execute callbacks
        if result["action"] == "continue" and on_continue:
            on_continue(result)
        # elif result["action"] == "exit" and on_exit:
        #     on_exit(result)

        return result
