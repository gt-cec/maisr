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