import pygame
import sys
import os
SCREEN_WIDTH = 540
SCREEN_HEIGHT = 960

RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
PURPLE = (255,0,255)
BLACK = (0,0,0)
WHITE = (255,255,255)
LIGHTGRAY = (180,180,180)
DARKGRAY = (127,127,127)

P1COLOR = (255,0,0)
P2COLOR = (255,255,0)


pygame.mixer.init()

# from https://pixabay.com/ko/music/search/genre/%EB%B9%84%EB%94%94%EC%98%A4%20%EA%B2%8C%EC%9E%84/
background_sound = pygame.mixer.Sound('files/sound/bgm.mp3')
background_sound.set_volume(0.3)

# from https://pixabay.com/ko/sound-effects/search/arcade/
game_sound = pygame.mixer.Sound('files/sound/game_sound.mp3')

# from https://freesound.org/people/MATRIXXX_/sounds/349873/ 
drop_sound = pygame.mixer.Sound('files/sound/drop_sound.wav')

# from https://pixabay.com/ko/sound-effects/search/game%20success/
recommend_sound = pygame.mixer.Sound('files/sound/recommend_sound.wav')
recommend_sound.set_volume(0.7)

# from https://pixabay.com/sound-effects/search/level/
win_sound = pygame.mixer.Sound('files/sound/win_sound.mp3')
win_sound.set_volume(0.5)

# from https://pixabay.com, but I'm not sure what is the exact address.
draw_sound = pygame.mixer.Sound('files/sound/draw_sound.wav')
draw_sound.set_volume(0.5)

# from https://pixabay.com/ko/sound-effects/search/booing/
fail_sound = pygame.mixer.Sound('files/sound/lose_sound.mp3')
fail_sound.set_volume(0.7)

# from https://pixabay.com/ko/sound-effects/search/arcade/
button_sound = pygame.mixer.Sound('files/sound/button_sound.mp3')
button_sound.set_volume(0.3)


# main_font from https://campaign.naver.com/nanumsquare_neo/#download

class Button:
    def __init__(self, name, cx=0,cy=0,width=100,height=100,color=WHITE, font_size=30):
        self.name = name
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.color = color
        self.clicked = False
        self.clicked_color = LIGHTGRAY
        self.font = pygame.font.Font('files/font/main_font.ttf', font_size)

    def draw_and_get_event(self, surface, event):
        x = self.cx-self.width/2
        y = self.cy-self.height/2
        # print(self.x,self.y, self.width, self.height)
        mouse_pos = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        # print(click)
        border = pygame.draw.rect(surface,self.color, (x,y,self.width,self.height))
        if x+self.width > mouse_pos[0] > x and y+self.height > mouse_pos[1] > y:
            

            if event is not None:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.clicked_color = DARKGRAY
                    self.clicked = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if self.clicked: 
                        self.clicked = False
                        self.clicked_color = LIGHTGRAY
                        button_sound.play()
                        return True
                    else:
                        self.clicked = False

            pygame.draw.rect(surface, self.clicked_color,(x,y,self.width,self.height))
            # if click[0] == 1:
            #     print(click)     
            #     return True
        elif self.clicked:
            if event.type == pygame.MOUSEBUTTONUP:
                self.clicked = False
                self.clicked_color = LIGHTGRAY
            pygame.draw.rect(surface, self.clicked_color,(x,y,self.width,self.height))
        else:
            border = pygame.draw.rect(surface,self.color, (x,y,self.width,self.height))
        text = self.font.render(self.name, True, BLACK)
        text_rect = text.get_rect(center=(surface.get_width()/2, surface.get_height()/2))
        text_rect.center = border.center
        surface.blit(text, text_rect)

        return False