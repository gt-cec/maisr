# Currently unused. Might move GUI element creation here later - Ryan
import pygame
import sys

class Button:
    def __init__(self, text, x, y, width, height):
        self.text = text
        self.x, self.y = x,y
        self.width, self.height = width, height
        self.rect = pygame.Rect(x,y,width,height)
        self.color = (255, 120, 80)
        self.default_color = (255, 120, 80)
        self.border_color = (0,0,0)
        self.border_width = 3
        self.font = pygame.font.SysFont(None, 36)
        self.is_latched = False

    def draw(self,win):
        if self.is_latched:
            self.color = (self.color[0]*.7, self.color[1]*.7, self.color[2]*.7)
            self.border_width = 4

            pygame.draw.line(win, (0,0,0), (self.x - 3, self.y - 3), (self.x + self.width + 3, self.y - 3),self.border_width)  # Top border
            pygame.draw.line(win, self.border_color, (self.x - 3, self.y + self.height + 1),(self.x + self.width + 3, self.y + self.height + 1), self.border_width)  # Bottom border
            pygame.draw.line(win, self.border_color, (self.x - 3, self.y - 3), (self.x - 3, self.y + self.height + 1),self.border_width)  # Left border
            pygame.draw.line(win, self.border_color, (self.x + self.width + 1, self.y - 3),(self.x + self.width + 1, self.y + self.height + 1), self.border_width)  # Right border

        pygame.draw.rect(win, self.color, self.rect)
        text_surface = self.font.render(self.text, True, (0,0,0))
        text_rect = text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + self.rect.height // 2))
        win.blit(text_surface, text_rect)

    def is_clicked(self,pos):
        return self.rect.collidepoint(pos)

class ScoreWindow:
    def __init__(self, score, x, y):
        self.rect = pygame.Rect(x,y,150,70)
        self.color = (200,200,200)
        self.font = pygame.font.SysFont(None, 36)
        self.score = 0

    def draw(self,win):
        black = (0,0,0)
        pygame.draw.rect(win, self.color, self.rect)
        text_surface = self.font.render('SCORE', True, black)
        text_rect = text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 0.5*self.rect.height // 2))
        win.blit(text_surface, text_rect)

        score_text_surface = self.font.render(str(self.score), True, black)
        score_text_rect = score_text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 1.4*self.rect.height // 2))
        win.blit(score_text_surface, score_text_rect)

    def update(self,score):
        self.score = score # Note: The truth source for score is env.score. This gets updated from that.

class HealthWindow:
    def __init__(self, agent_id, x, y, text, title_color):
        self.rect = pygame.Rect(x,y,150,70)
        self.color = (200,200,200)
        self.font = pygame.font.SysFont(None, 36)
        self.agent_id = agent_id
        self.damage = 0
        self.text = text
        self.damage_text_color = (0,0,0)
        self.title_color = title_color

    def draw(self,win):
        pygame.draw.rect(win, self.color, self.rect)
        #health_text = 'AGENT ' + str(self.agent_id)
        text_surface = self.font.render(self.text, True, self.title_color)
        text_rect = text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 0.5*self.rect.height // 2))
        win.blit(text_surface, text_rect)

        if self.damage >= 70: self.damage_text_color = (255,0,0)
        elif self.damage >= 40: self.damage_text_color = (210,160,0)


        health_num_text_surface = self.font.render(str(round(self.damage,1)), True, self.damage_text_color) # TODO: Update with agent health
        health_num_text_rect = health_num_text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 1.4*self.rect.height // 2))
        win.blit(health_num_text_surface, health_num_text_rect)

    def update(self,damage):
        self.damage = damage # Note: The truth source for score is env.score. This gets updated from that.

class TimeWindow:
    def __init__(self, x, y,current_time=0): # Takes current game time in raw form (milliseconds).
        self.rect = pygame.Rect(x,y,150,70)
        self.color = (200,200,200)
        self.font = pygame.font.SysFont(None, 36)
        self.current_time = current_time

    def draw(self,win):
        black = (0,0,0)
        pygame.draw.rect(win, self.color, self.rect)

        timer_title_surface = self.font.render('TIME LEFT', True, black)
        win.blit(timer_title_surface, timer_title_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 0.5 * self.rect.height // 2)))

        time_text_surface = self.font.render(str(round(120 - self.current_time/1000,0)), True, black)
        time_text_rect = time_text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 1.4*self.rect.height // 2))
        win.blit(time_text_surface, time_text_rect)

    def update(self,time):
        self.current_time = time


class AgentInfoDisplay:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = 340  # Fixed height to accommodate two threat lines
        self.title_font = pygame.font.SysFont(None, 36)
        self.header_font = pygame.font.SysFont(None, 28)
        self.text_font = pygame.font.SysFont(None, 24)

        # Colors matching existing UI
        self.title_bg = (200, 200, 200)
        self.content_bg = (230, 230, 230)
        self.text_color = (0, 0, 0)
        self.warning_color = (255, 0, 0)
        self.highlight_color = (0, 0, 255)

        # Risk level colors and text colors
        self.risk_colors = {
            'LOW': {'bg': (0, 190, 0), 'text': (0, 0, 0)},
            'MEDIUM': {'bg': (255, 165, 0), 'text': (0, 0, 0)},
            'HIGH': {'bg': (255, 0, 0), 'text': (0, 0, 0)},
            'EXTREME': {'bg': (0, 0, 0), 'text': (255, 255, 255)}
        }

    def draw(self, window, agent_info):
        # Draw main container
        pygame.draw.rect(window, self.content_bg,
                         pygame.Rect(self.x, self.y, self.width, self.height))

        # Draw title bar
        title_height = 40
        pygame.draw.rect(window, self.title_bg,
                         pygame.Rect(self.x, self.y, self.width, title_height))

        # Draw title
        title_surface = self.title_font.render('AGENT STATUS', True, self.text_color)
        window.blit(title_surface, title_surface.get_rect(
            center=(self.x + self.width // 2, self.y + title_height // 2)))

        current_y = self.y + title_height + 15

        # Current Search Mode
        if agent_info.get('show_current_action', False):
            strategy = agent_info.get('decision', {}).get('strategy')
            if strategy:
                mode_text = "Search Mode: "
                if strategy == 'defensive_search':
                    mode_text += "Defensive (Target ID Only)"
                elif strategy == 'aggressive_search':
                    mode_text += "Aggressive (Target + WEZ ID)"
                mode_surface = self.text_font.render(mode_text, True, self.highlight_color)
                window.blit(mode_surface, (self.x + 15, current_y))
                current_y += 30

            # Current Action
            action_details = agent_info.get('action', {}).get('details', 'N/A')
            text_surface = self.text_font.render(action_details, True, self.text_color)
            window.blit(text_surface, (self.x + 15, current_y))
            current_y += 35

        if agent_info.get('show_risk_info', False):
            # Risk Level Label
            label_text = "Risk Level: "
            label_surface = self.header_font.render(label_text, True, self.text_color)
            window.blit(label_surface, (self.x + 15, current_y))

            # Get risk level and colors
            risk_level = agent_info.get('risk', {}).get('level', 'LOW')
            risk_colors = self.risk_colors.get(risk_level, self.risk_colors['LOW'])

            # Calculate positions and sizes for risk level display
            label_width = label_surface.get_width()
            level_text = risk_level
            level_surface = self.header_font.render(level_text, True, risk_colors['text'])
            level_width = level_surface.get_width() + 20
            level_height = level_surface.get_height() + 6

            # Draw colored background rectangle
            pygame.draw.rect(window, risk_colors['bg'],
                             pygame.Rect(self.x + 15 + label_width,
                                         current_y - 3,
                                         level_width,
                                         level_height))

            # Draw risk level text
            window.blit(level_surface, (self.x + 15 + label_width + 10,
                                        current_y))

            current_y += 35

        # Decision Rationale section
        if agent_info.get('show_decision_rationale', False):
            current_y += 10
            constraints = agent_info.get('decision', {}).get('constraints', [])
            for constraint in constraints:
                if "density" in constraint.lower():
                    text_surface = self.text_font.render(constraint, True, self.highlight_color)
                else:
                    text_surface = self.text_font.render(constraint, True, self.text_color)
                window.blit(text_surface, (self.x + 15, current_y))
                current_y += 25

        # Add separator line above threat callouts
        current_y = self.y + self.height - 85  # Fixed position from bottom
        pygame.draw.line(window, (180, 180, 180),
                         (self.x + 10, current_y),
                         (self.x + self.width - 10, current_y), 2)
        current_y += 15

        # Threat Callouts section (limited to two threats)
        if agent_info.get('show_risk_info', False):
            threats = agent_info.get('risk', {}).get('threats', [])
            if threats:
                # Add "Detected Threats:" header
                header_surface = self.header_font.render("Detected Threats:", True, self.text_color)
                window.blit(header_surface, (self.x + 15, current_y))
                current_y += 30

                # List up to two threats
                for threat in threats[:2]:  # Limit to first two threats
                    threat_text = f"Threat at {threat['bearing']:.0f}°, {threat['distance']:.0f} units"
                    text_surface = self.text_font.render(threat_text, True, self.warning_color)
                    window.blit(text_surface, (self.x + 15, current_y))
                    current_y += 25

        # Draw border
        pygame.draw.rect(window, (0, 0, 0),
                         pygame.Rect(self.x, self.y, self.width, self.height), 2)