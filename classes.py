import pygame
# from game import EFFECT_SOUND

SCREEN_WIDTH = 540
SCREEN_HEIGHT = 960

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
PURPLE = (255,0,255)
BLACK = (0,0,0)
WHITE = (255,255,255)
LIGHTGRAY = (180,180,180)
DARKGRAY = (127,127,127)




pygame.mixer.init()
'''
sound는 3개의 원소를 가진 tuple로 쓰기로 한다
a_sound = (pygame.mixer.Sound(), default sound volumn, sound type)

default sound volumn: (0 ~ 1)
sound type: 0 -> music, 1 -> effect
'''
# from https://pixabay.com/ko/music/search/genre/%EB%B9%84%EB%94%94%EC%98%A4%20%EA%B2%8C%EC%9E%84/
background_sound = (pygame.mixer.Sound('files/sound/bgm.mp3'), 0.3, 0)

# from https://pixabay.com/ko/sound-effects/search/arcade/
game_sound = (pygame.mixer.Sound('files/sound/game_sound.mp3'),1, 0)

# from https://freesound.org/people/MATRIXXX_/sounds/349873/ 
drop_sound = (pygame.mixer.Sound('files/sound/drop_sound.wav'),1, 1)


# from https://pixabay.com/ko/sound-effects/search/game%20success/
recommend_sound = (pygame.mixer.Sound('files/sound/recommend_sound.wav'), 0.7, 1)


# from https://pixabay.com/sound-effects/search/level/
win_sound = (pygame.mixer.Sound('files/sound/win_sound.mp3'), 0.5, 1)

# from https://pixabay.com, but I'm not sure what is the exact address.
draw_sound = (pygame.mixer.Sound('files/sound/draw_sound.wav'), 0.5, 1)



# from https://pixabay.com/ko/sound-effects/search/booing/
fail_sound = (pygame.mixer.Sound('files/sound/lose_sound.mp3'), 0.7, 1)


# from https://pixabay.com/ko/sound-effects/search/arcade/
button_sound = (pygame.mixer.Sound('files/sound/button_sound.mp3'), 0.3, 1)

# from https://pixabay.com/ko/sound-effects/search/pop/
undo_sound = (pygame.mixer.Sound('files/sound/undo_sound.mp3'), 0.5, 1)

# from https://pixabay.com/ko/sound-effects/search/boop/

select_color_sound = (pygame.mixer.Sound('files/sound/select_color_sound.mp3'), 0.7, 1)

# from https://pixabay.com/ko/sound-effects/search/xylophone/
do = (pygame.mixer.Sound('files/sound/xylophone_do.mp3'), 1, 1)
mi = (pygame.mixer.Sound('files/sound/xylophone_mi.mp3'), 1, 1)
sol = (pygame.mixer.Sound('files/sound/xylophone_sol.mp3'), 1, 1)
high_do = (pygame.mixer.Sound('files/sound/xylophone_do2.mp3'), 1, 1)
connect4_sound = [do, mi, sol, high_do]

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
        self.change_text_color()
        

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
            pygame.draw.rect(surface,self.color, (x,y,self.width,self.height))
        text = self.font.render(self.name, True, self.text_color)
        text_rect = text.get_rect(center=(surface.get_width()/2, surface.get_height()/2))
        text_rect.center = border.center
        surface.blit(text, text_rect)

        return False
    

    def change_color(self, color):
        self.color = color
        self.change_text_color()

    def change_text_color(self):
        r,g,b = self.color
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b # per ITU-R BT.709
        # print("luma:",luma)
        if luma < 127.5: self.text_color=WHITE
        else: self.text_color=BLACK

color_picker = (
    (255, 0, 0),  # - 빨간색
    (0, 255, 0),  # - 초록색
    (0, 0, 255),  # - 파란색
    (255, 255, 0),  # - 노란색
    (255, 0, 255),  # - 마젠타
    (0, 255, 255),  # - 시안
    (10, 10, 10),  # - 검은색 
    (0, 128, 0),  # - 어두운 초록색
    (0, 0, 128),  # - 어두운 파란색
    (128, 128, 0),  # - 어두운 노란색
    (128, 0, 128),  # - 어두운 보라색
    (0, 128, 128),  # - 어두운 청녹색
    (192, 0, 0),  # - 선명한 빨간색
    (0, 192, 0),  # - 선명한 초록색
    (0, 0, 192),  # - 선명한 파란색
    (255, 128, 0),  # - 주황색
    (255, 0, 128),  # - 로즈
    (128, 255, 0),  # - 라임
    (0, 255, 128),  # - 민트
    (128, 0, 255),  # - 보라
    (0, 128, 255),  # - 하늘색
    (255, 128, 128),  # - 연한 빨간색
    (128, 255, 128),  # - 연한 초록색
    (128, 128, 255),  # - 연한 파란색
    (255, 255, 153),  # - 레몬
    (153, 255, 255),  # - 아쿠아
    (255, 102, 255),  # - 핑크
    (153, 153, 255),  # - 라벤더
    (57, 255, 20),  # - 밝은 라임
    (255, 140, 105),  # - 살구색
    (51, 153, 255),  # - 코너플라워 블루
    (153, 255, 51),  # - 샤르트루즈 그린
    (255, 80, 147),  # - 핫 핑크
    (102, 51, 153),  # - 임페리얼 퍼플
    (244, 164, 96),  # - 살구석
    (128, 222, 234),  # - 파우더 블루
    (227, 11, 92),  # - 자몽
    (153, 102, 51),  # - 밤색
    (246, 173, 198),  # - 연분홍
    (85, 107, 47),  # - 올리브 드라브
    (216, 191, 216),  # - 시스
    (65, 105, 225),  # - 로열 블루
)