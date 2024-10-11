# Currently unused. Might move GUI element creation here later - Ryan
import pygame
import sys

class Button:
    def __init__(self, text, x, y, width, height):
        self.text = text
        self.rect = pygame.Rect(x,y,width,height)
        self.color = (255, 120, 80)
        self.font = pygame.font.SysFont(None, 36)

    def draw(self,win):
        black = (0,0,0)
        pygame.draw.rect(win, self.color, self.rect)
        text_surface = self.font.render(self.text, True, black)
        text_rect = text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + self.rect.height // 2))
        win.blit(text_surface, text_rect)

    def is_clicked(self,pos):
        self.color = (230,90,50)
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
    def __init__(self, x, y,current_time=0):
        self.rect = pygame.Rect(x,y,150,70)
        self.color = (200,200,200)
        self.font = pygame.font.SysFont(None, 36)
        self.current_time = current_time

    def draw(self,win):
        black = (0,0,0)
        pygame.draw.rect(win, self.color, self.rect)

        timer_title_surface = self.font.render('TIME LEFT', True, black)
        win.blit(timer_title_surface, timer_title_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 0.5 * self.rect.height // 2)))

        time_text_surface = self.font.render(str(round(self.current_time,1)), True, black)
        time_text_rect = time_text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 1.4*self.rect.height // 2))
        win.blit(time_text_surface, time_text_rect)

    def update(self,time):
        self.current_time = time # Note: The truth source for score is env.score. This gets updated from that.