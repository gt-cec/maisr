# Currently unused. Might move GUI element creation here later - Ryan
import pygame
import sys

class Button:
    def __init__(self, text, x, y, width, height, color):
        self.text = text
        self.rect = pygame.Rect(x,y,width,height)
        self.color = color
        self.font = pygame.font.SysFont(None, 36)

    def draw(self,win):
        black = (0,0,0)
        pygame.draw.rect(win, self.color, self.rect)
        text_surface = self.font.render(self.text, True, black)
        text_rect = text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + self.rect.height // 2))
        win.blit(text_surface, text_rect)

    def is_clicked(self,pos):
        #self.color = (self.color[0]*1.1, self.color[1]*1.1, self.color[2]*1.1)
        return self.rect.collidepoint(pos)

# game_width*0.5 - 150/2, game_width + 10

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
    def __init__(self, agent_id, x, y):
        self.rect = pygame.Rect(x,y,150,70)
        self.color = (200,200,200)
        self.font = pygame.font.SysFont(None, 36)
        self.agent_id = agent_id
        self.damage = 0

    def draw(self,win):
        black = (0,0,0)
        pygame.draw.rect(win, self.color, self.rect)
        health_text = 'AGENT ' + str(self.agent_id)
        text_surface = self.font.render(health_text, True, black)
        text_rect = text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 0.5*self.rect.height // 2))
        win.blit(text_surface, text_rect)

        health_num_text_surface = self.font.render(str(round(self.damage,1)), True, black) # TODO: Update with agent health
        health_num_text_rect = health_num_text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 1.4*self.rect.height // 2))
        win.blit(health_num_text_surface, health_num_text_rect)

    def update(self,damage):
        self.damage = damage # Note: The truth source for score is env.score. This gets updated from that.