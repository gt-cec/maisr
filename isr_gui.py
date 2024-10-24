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


        #health_num_text_surface = self.font.render(str(round(self.damage,1)), True, self.damage_text_color) # TODO: Update with agent health
        health_num_text_surface = self.font.render(str(round(self.damage,1)) + '/4', True,self.damage_text_color)  # TODO: Update with agent health
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
    def __init__(self, x, y, width=445, height=250):
        self.rect = pygame.Rect(x, y, width, height)
        self.title_rect = pygame.Rect(x, y, width, 40)
        self.content_rect = pygame.Rect(x, y + 40, width, height - 40)
        self.color = (230, 230, 230)
        self.title_color = (200, 200, 200)
        self.title_font = pygame.font.SysFont(None, 36)
        self.content_font = pygame.font.SysFont(None, 24)
        self.text = []
        self.line_height = 30
        self.text_margin = 10

        # Risk level colors
        self.risk_colors = {
            'LOW': ((50, 255, 50), (0, 0, 0)),  # (Green bg, black text)
            'MEDIUM': ((255, 165, 0), (0, 0, 0)),  # (Orange bg, black text)
            'HIGH': ((255, 50, 50), (0, 0, 0)),  # (Red bg, black text)
            'EXTREME': ((0, 0, 0), (255, 255, 255))  # (Black bg, white text)
        }

    def draw(self, window):
        # Draw background rectangles
        pygame.draw.rect(window, self.title_color, self.title_rect)
        pygame.draw.rect(window, self.color, self.content_rect)

        # Draw title
        title_surface = self.title_font.render('AGENT STATUS', True, (0, 0, 0))
        title_rect = title_surface.get_rect(center=(self.title_rect.centerx, self.title_rect.centery))
        window.blit(title_surface, title_rect)

        # Draw content
        y_offset = self.content_rect.y + self.text_margin
        for line in self.text:
            if "RISK LEVEL:" in line:
                # Split the line into label and value
                label, risk_level = line.split(": ")
                risk_level = risk_level.strip()

                # Render the label part
                label_surface = self.content_font.render(label + ": ", True, (0, 0, 0))
                window.blit(label_surface, (self.content_rect.x + self.text_margin, y_offset))

                if risk_level in self.risk_colors:
                    bg_color, text_color = self.risk_colors[risk_level]

                    # Render the risk level to get its width
                    risk_surface = self.content_font.render(risk_level, True, text_color)
                    risk_width = risk_surface.get_width()

                    # Draw background rectangle just for the risk level text
                    risk_rect = pygame.Rect(
                        self.content_rect.x + self.text_margin + label_surface.get_width(),
                        y_offset-3,
                        risk_width+6,
                        self.line_height - 10
                    )
                    pygame.draw.rect(window, bg_color, risk_rect)

                    # Draw the risk level text
                    window.blit(risk_surface, (risk_rect.x, y_offset))
                else:
                    # Fallback if risk level not recognized
                    risk_surface = self.content_font.render(risk_level, True, (0, 0, 0))
                    window.blit(risk_surface,
                                (self.content_rect.x + self.text_margin + label_surface.get_width(), y_offset))
            else:
                # Draw regular lines
                text_surface = self.content_font.render(line, True, (0, 0, 0))
                window.blit(text_surface, (self.content_rect.x + self.text_margin, y_offset))

            y_offset += self.line_height

        # Draw borders
        border_width = 4
        pygame.draw.line(window, (0, 0, 0), (self.rect.x, self.rect.y),
                         (self.rect.x + self.rect.width, self.rect.y), border_width)
        pygame.draw.line(window, (0, 0, 0), (self.rect.x, self.rect.y),
                         (self.rect.x, self.rect.y + self.rect.height), border_width)
        pygame.draw.line(window, (0, 0, 0), (self.rect.x + self.rect.width, self.rect.y),
                         (self.rect.x + self.rect.width, self.rect.y + self.rect.height), border_width)
        pygame.draw.line(window, (0, 0, 0), (self.rect.x, self.rect.y + self.rect.height),
                         (self.rect.x + self.rect.width, self.rect.y + self.rect.height), border_width)

    def update_text(self, new_text):
        self.text = new_text