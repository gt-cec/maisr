import pygame
import sys
from typing import Dict, List, Optional, Callable
from enum import Enum


class ScreenType(Enum):
    """Types of instructional screens"""
    WELCOME = "welcome"
    INSTRUCTIONS = "instructions"
    BETWEEN_EPISODES = "between_episodes"
    FINAL_SUMMARY = "final_summary"


class InstructionalScreen:
    """Base class for instructional screens"""

    def __init__(self, window_width: int = 1500, window_height: int = 1200):
        self.window_width = window_width
        self.window_height = window_height
        self.background_color = (30, 30, 30)  # Dark gray
        self.text_color = (255, 255, 255)  # White
        self.highlight_color = (100, 150, 255)  # Light blue
        self.font_large = pygame.font.SysFont('Arial', 36, bold=True)
        self.font_medium = pygame.font.SysFont('Arial', 24)
        self.font_small = pygame.font.SysFont('Arial', 18)

        # Input state
        self.text_input = ""
        self.cursor_visible = True
        self.cursor_timer = 0
        self.cursor_blink_rate = 500  # milliseconds

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
            elif event.key == pygame.K_ESCAPE:
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

    def __init__(self, episode_config: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_config = episode_config

        # Survey questions and their labels
        self.questions = [
            "Mental demand",
            "Physical demand",
            "Temporal demand",
            "Effort",
            "Performance",
            "Frustration"
        ]

        # User responses (1-7 scale, None = not answered)
        self.responses = {question: None for question in self.questions}

        # Visual properties for rating bars
        self.bar_width = 600
        self.bar_height = 50
        self.segment_width = self.bar_width // 7
        self.bar_start_x = (self.window_width - self.bar_width) // 2
        self.bar_spacing = 80
        self.first_bar_y = 200

        # Colors
        self.unselected_color = (100, 100, 100)
        self.selected_color = (100, 150, 255)
        self.hover_color = (150, 150, 150)
        self.border_color = (200, 200, 200)

        # Mouse interaction
        self.hover_segment = None  # (question_index, segment_index)

    def draw_content(self, window: pygame.Surface) -> None:
        # Title
        title = "Please rate your workload in the last round:"
        if self.episode_config:
            title = f"Please rate your workload in the last round: ({self.episode_config})"
        self.draw_text_centered(window, title, 120, self.font_large)

        # Draw rating bars
        for i, question in enumerate(self.questions):
            self.draw_rating_bar(window, i, question)

        # Draw continue button/instructions
        self.draw_continue_section(window)

    def draw_rating_bar(self, window: pygame.Surface, question_index: int, question_text: str) -> None:
        """Draw a single rating bar with 7 segments"""
        y_pos = self.first_bar_y + (question_index * self.bar_spacing)

        # Draw question label
        label_surface = self.font_medium.render(question_text, True, self.text_color)
        window.blit(label_surface, (self.bar_start_x, y_pos - 35))

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
            number_surface = self.font_small.render(number_text, True, self.text_color)
            number_rect = number_surface.get_rect(center=segment_rect.center)
            window.blit(number_surface, number_rect)

    def draw_continue_section(self, window: pygame.Surface) -> None:
        """Draw the continue button area"""
        continue_y = self.first_bar_y + (len(self.questions) * self.bar_spacing) + 50

        # Check if all questions are answered
        all_answered = all(response is not None for response in self.responses.values())

        if all_answered:
            # Draw enabled continue button
            arrow_text = "→"
            button_color = self.selected_color
            text_color = self.text_color
            instruction_text = "Click arrow or press ENTER to continue"
        else:
            # Draw disabled continue button
            arrow_text = "→"
            button_color = self.unselected_color
            text_color = (100, 100, 100)
            unanswered_count = sum(1 for response in self.responses.values() if response is None)
            instruction_text = f"Please answer all questions ({unanswered_count} remaining)"

        # Draw arrow button
        button_size = 60
        button_x = self.window_width - 150
        button_y = continue_y
        button_rect = pygame.Rect(button_x, button_y, button_size, button_size)

        pygame.draw.rect(window, button_color, button_rect, border_radius=10)
        pygame.draw.rect(window, self.border_color, button_rect, 3, border_radius=10)

        # Draw arrow
        arrow_surface = self.font_large.render(arrow_text, True, text_color)
        arrow_rect = arrow_surface.get_rect(center=button_rect.center)
        window.blit(arrow_surface, arrow_rect)

        # Draw instruction text
        instruction_surface = self.font_small.render(instruction_text, True, text_color)
        instruction_rect = instruction_surface.get_rect(center=(self.window_width // 2, continue_y + 100))
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
            elif event.key == pygame.K_ESCAPE:
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



class FinalSummaryScreen(InstructionalScreen):
    """Final screen shown after all episodes"""

    def __init__(self, experiment_results: List[Dict], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_results = experiment_results

    def draw_content(self, window: pygame.Surface) -> None:
        # Title
        self.draw_text_centered(window, "Experiment Complete!", 150,
                                self.font_large, self.highlight_color)

        # Summary stats
        if self.experiment_results:
            total_reward = sum(r['reward'] for r in self.experiment_results)
            avg_reward = total_reward / len(self.experiment_results)

            self.draw_text_centered(window, f"Episodes Completed: {len(self.experiment_results)}",
                                    220, self.font_medium)
            self.draw_text_centered(window, f"Total Score: {total_reward:.2f}",
                                    250, self.font_medium)
            self.draw_text_centered(window, f"Average Score: {avg_reward:.2f}",
                                    280, self.font_medium, (100, 255, 100))

        # Thank you message
        thank_you = [
            "",
            "Thank you for participating in this study!",
            "",
            "Your data will help us understand how humans and AI",
            "can work together more effectively in complex tasks.",
            "",
            "If you have any questions about the study, please",
            "ask the experimenter.",
            "",
            "Press ENTER to exit."
        ]

        self.draw_text_block(window, thank_you, 350, self.font_medium, 35)


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
            dt = self.clock.tick(60)

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
        elif result["action"] == "exit" and on_exit:
            on_exit(result)

        return result