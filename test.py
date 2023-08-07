import pygame
import pygame.gfxdraw
import numpy as np



SCREEN_WIDTH = 540
SCREEN_HEIGHT = 960
WHITE = (255,255,255)
RED = (255,0,0)
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.init()

def fill_arc(color, center, radius, theta0, theta1, ndiv=50):
    x0, y0 = center

    dtheta = (theta1 - theta0) / ndiv
    angles = [theta0 + i*dtheta for i in range(ndiv + 1)] 

    points = [(x0, y0)] + [(x0 + radius * np.cos(theta), y0 - radius * np.sin(theta)) for theta in angles]

    pygame.gfxdraw.filled_polygon(SCREEN, points, color)

run = True
while run:
    SCREEN.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            SCREEN.fill(WHITE)
            run = False

    start = 200
    end = 380

    fill_arc(RED, (400,400), 40, start/180*np.pi, end/180*np.pi, ndiv=50)
    pygame.display.flip()
    