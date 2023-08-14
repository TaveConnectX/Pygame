import pygame
import pygame.gfxdraw
import sys
import numpy as np
import os
import copy
import pickle
from classes import *
from test_model import test_main
from functions import *



pygame.init()




ver = "1.0.2"  # version
# Clock 객체 생성
clock = pygame.time.Clock()
frame = 60

setting = load_setting()

FIRST_PLAYER = setting['first_player']
P1COLOR = setting['p1_color']
P2COLOR = setting['p2_color']
MUSIC_SOUND = setting['music']
EFFECT_SOUND = setting['effect']

# print(EFFECT_SOUND)
# pygame.Rect(x,y,width, height)
# myRect = pygame.Rect(150, 200, 200, 100)

# 중력을 계산하기 위한 class
class FallingInfo:
    def __init__(self):
        self.pos = (None, None)  # 현재 돌의 좌표
        self.base_pos = (None,None)  # 바닥 돌의 좌표
        self.target_pos = (None, None)  # 도달해야 하는 좌표 
        self.v = 0  # 속도
        self.max_v = 0  # 소리의 크기를 계산하기 위한 최대 속력 
        self.g = 0.8  # 중력가속도 
        # 돌이 무한으로 튀어올라서 멈추지 않는 문제를 방지하기 위해 bounce한 수 세기 
        self.bounce = 0  # 돌이 튀어오른 수
        
    # target 좌표를 이용해서 변수들을 초기화 
    def set_pos(self, target_coord):
        self.bounce = 0
        self.v = 0
        target_row, target_col = target_coord
        self.target_pos = coord2pos(SCREEN,(target_row,target_col))
        # 바닥은 (5,col) 보다 낮은 (6,col)
        # 떨어지기 시작하는 위치는 (0,col) 보다 높은 (-1, col) 로 설정 
        self.base_pos = coord2pos(SCREEN,(6,target_col))  
        self.pos = coord2pos(SCREEN,(-1,target_col))  

    def calculate_info(self):
        self.v += self.g
        self.pos[1] += self.v

        if self.target_pos[1] < self.pos[1] and self.v > 0:
            
            
            self.bounce += 1
            if self.bounce==1 and self.max_v==0: self.max_v = self.v
            play_sound(drop_sound, repeat=False, custom_volume=self.v/self.max_v)
            
            self.v *= -1/2
            self.pos[1] = self.target_pos[1]

            
        if self.bounce >= 5: 
            self.v = 0
            self.pos = self.target_pos
        # print(self.v)
        
    # 떨어지던 돌이 멈췄는지 확인 
    def stopped(self):
        if self.pos==(None,None) or self.bounce==5: return True
        else: return False


def play_sound(sound_info, repeat=False, custom_volume=1):
    sound, default_volume, sound_type = sound_info
    system_volume = EFFECT_SOUND if sound_type else MUSIC_SOUND
    sound.set_volume(default_volume * system_volume * custom_volume)
    if repeat: sound.play(-1)
    else: sound.play()



            

def intro():
    w, h = SCREEN.get_size()
    intro_button_width = w//3
    intro_button_height = h//16
    print(w,h,intro_button_width,intro_button_height)
    
    play_sound(background_sound, repeat=True)
    music_on = True
    setting_button = Button("",w-55,h-55, 70,70)
    intro_buttons = [
        Button('new game',cx=w/2,cy=h/2+intro_button_height*1,width=intro_button_width, height=intro_button_height),
        Button('continue',cx=w/2,cy=h/2+intro_button_height*2,width=intro_button_width, height=intro_button_height),
        Button('how to',cx=w/2,cy=h/2+intro_button_height*3,width=intro_button_width, height=intro_button_height),
        Button('review',cx=w/2,cy=h/2+intro_button_height*4,width=intro_button_width, height=intro_button_height),
        Button('info',cx=w/2,cy=h/2+intro_button_height*5,width=intro_button_width, height=intro_button_height),
        Button('quit',cx=w/2,cy=h/2+intro_button_height*6,width=intro_button_width, height=intro_button_height)
    ]
    logo_image = pygame.image.load('files/image/logo.png')
    logo_image = pygame.transform.scale(logo_image, (w/1.1, w/1.1/4))
    logo_rect = logo_image.get_rect()
    logo_rect.x = w/2- logo_image.get_width()/2
    logo_rect.y = h/4- logo_image.get_height()/2

    # from https://www.pngwing.com/en/free-png-ddmrj
    setting_image = pygame.image.load('files/image/setting_icon.png')
    setting_image = pygame.transform.scale(setting_image, (70, 70))
    setting_rect = setting_image.get_rect()
    setting_rect.x = w-90
    setting_rect.y = h-90
    
    ee_cnt = 0

    
    run = True
    event = None
    SCREEN.fill((255,255,255))
    
    while run:
        if not music_on: 
            background_sound[0].set_volume(background_sound[1] * MUSIC_SOUND)
            music_on = True
        SCREEN.blit(logo_image, logo_rect)
        setting_action = setting_button.draw_and_get_event(SCREEN,event)
        SCREEN.blit(setting_image, setting_rect)
        for button in intro_buttons:
            action = button.draw_and_get_event(SCREEN,event)
            if action:
                print(button.name)
                play_sound(button_sound, repeat=False, custom_volume=1)
                music_on = False
                background_sound[0].set_volume(background_sound[1] * MUSIC_SOUND / 3)
                if button.name == 'new game': select_difficulty()
                elif button.name == 'continue': play(difficulty=None, cont_game=True)
                elif button.name == 'how to': how_to()
                elif button.name == 'review': review()
                elif button.name == 'info': info()
                elif button.name == 'quit': run=False
                else:
                    print('error')
        
        if setting_action:
            music_on = False
            play_sound(button_sound, repeat=False, custom_volume=1)
            background_sound[0].set_volume(background_sound[1] * MUSIC_SOUND / 3)
            setting()
        if ee_cnt==11:
            ee()
            ee_cnt=0
        for event in pygame.event.get():
            # 마우스를 클릭해서 
            if event.type == pygame.MOUSEBUTTONDOWN:
                clicked_x, clicked_y = pygame.mouse.get_pos()
                if logo_rect.x <=clicked_x<=logo_rect.x+logo_image.get_width():
                    if logo_rect.y <= clicked_y <= logo_rect.y+logo_image.get_height():
                        ee_cnt += 1
                    else: ee_cnt = 0
                else: ee_cnt = 0
            if event.type == pygame.QUIT:
                run = False
        clock.tick(frame)
        pygame.display.flip()


    pygame.quit()
    sys.exit()






def select_difficulty():
    w, h = SCREEN.get_size()
    # 버튼 3개 만들고
    easy_button = Button('easy',cx=w/2,cy=h/2/2,width=w/3)
    normal_button = Button('normal',cx=w/2,cy=h/2,width=w/3)
    hard_button = Button('hard',cx=w/2,cy=h/4*3,width=w/3)
    back_button = Button('<',25,25,50,50)
    go_back = False
    # 선택하면 play()    
    run = True
    event = None
    SCREEN.fill((255,255,255))
    easy_action = easy_button.draw_and_get_event(SCREEN,event)
    normal_action = normal_button.draw_and_get_event(SCREEN,event)
    hard_action = hard_button.draw_and_get_event(SCREEN,event)
    while run:
        for event in pygame.event.get():
            
            

            easy_action = easy_button.draw_and_get_event(SCREEN,event)
            normal_action = normal_button.draw_and_get_event(SCREEN,event)
            hard_action = hard_button.draw_and_get_event(SCREEN,event)
            if event.type == pygame.QUIT:
                run = False
                sys.exit()
        if easy_action:
            play_sound(button_sound, repeat=False, custom_volume=1)
            play(difficulty='easy')
            return
        elif normal_action:
            play_sound(button_sound, repeat=False, custom_volume=1)
            play(difficulty='normal')
            return
        elif hard_action:
            play_sound(button_sound, repeat=False, custom_volume=1)
            play(difficulty='hard')
            return
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            return
        pygame.display.flip()

def no_board_to_continue():
    w,h = SCREEN.get_size()

    back_button = Button('back',cx=w/2,cy=h*3/4,width=w/3,height=100)

    border = pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
    font = pygame.font.Font('files/font/main_font.ttf', 30)
    text = font.render("이어할 게임이 없습니다", True, BLACK)
    text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    text_rect.center = border.center

    run = True
    event = None
    go_back = False
    
    while run:
        SCREEN.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
                sys.exit()
        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            return
        go_back = back_button.draw_and_get_event(SCREEN, event)
        SCREEN.blit(text, text_rect)
        pygame.display.flip()


def draw_cursor(x, player):
    if player == 1: color = P1COLOR
    else: color = P2COLOR
    pygame.draw.polygon(SCREEN, color, ((x,90),(x-10,80),(x+10,80)), width=0)

def draw_circle_with_pos(pos,player):
    w,h = SCREEN.get_size()
    r = (w-100)/7/2/1.05
    if abs(player) == 1: color = P1COLOR
    elif abs(player) == 2: color = P2COLOR
    # recommend 
    else:
        R1, G1, B1 = P1COLOR
        R2, G2, B2 = P2COLOR
        # color = (255-(R1+R2)//2, 255-(G1+G2)//2, 255-(B1+B2)//2)
        color = ((255-R1)/3*2+R1,(255-G1)/3*2+G1,(255-B1)/3*2+B1)
    x,y = pos
    x += 0.5
    y += 0.5
    pos = (x,y)
    pygame.draw.circle(SCREEN,color,pos,r)
    if player < 0:
        pygame.draw.circle(SCREEN,WHITE,pos,r*0.8)


def fill_arc(color, center, radius, theta0, theta1, break_time, ndiv=50):
    x0, y0 = center

    dtheta = (theta1 - theta0) / ndiv
    angles = [theta0 + i*dtheta for i in range(ndiv + 1)] 

    points = [(x0, y0)] + [(x0 + radius * np.cos(theta), y0 - radius * np.sin(theta)) for theta in angles]

    r,g,b = color
    color = (r+(255-r)/30*break_time,g+(255-g)/30*break_time,b+(255-b)/30*break_time)
    pygame.gfxdraw.filled_polygon(SCREEN, points, color)


def get_broken_circle_info_with_coord(coord, player):
    w,h = SCREEN.get_size()
    r = (w-100)/7/2/1.05
    if player==1: color=P1COLOR
    elif player==2: color=P2COLOR
    else:
        color = None

    rand_angle = np.random.uniform(0, 2*np.pi)
    x,y = coord2pos(SCREEN,coord)
    broken_x, broken_y = x+r*np.cos(rand_angle), y+r*np.sin(rand_angle)
    # print("check broken")
    # print("r:",r)
    # print("x:",x,", broken x:", broken_x)
    # print("y:",y, ", broken y:",broken_y)

    rotate_rad_1 = np.random.uniform(15, 35) * (np.pi / 180.0)
    broken_center_x_1 = round(np.cos(rotate_rad_1)*(x-broken_x) - np.sin(rotate_rad_1)*(y-broken_y)) + broken_x
    broken_center_y_1 = round(np.sin(rotate_rad_1)*(x-broken_x) + np.cos(rotate_rad_1)*(y-broken_y)) + broken_y
    # print("broken center1:", broken_center_x_1, broken_center_y_1)
    # print("start angle:", (-rand_angle-rotate_rad_1) * 180/np.pi, ", end angle:", (np.pi-rand_angle-rotate_rad_1) * 180/np.pi)
    
    
    rotate_rad_2 = -1 * rotate_rad_1
    broken_center_x_2 = round(np.cos(rotate_rad_2)*(x-broken_x) - np.sin(rotate_rad_2)*(y-broken_y)) + broken_x
    broken_center_y_2 = round(np.sin(rotate_rad_2)*(x-broken_x) + np.cos(rotate_rad_2)*(y-broken_y)) + broken_y
    # print("broken center2:", broken_center_x_2, broken_center_y_2)
    # print("start angle:", (np.pi-rand_angle-rotate_rad_2) * 180/np.pi, ", end angle:",(2*np.pi-rand_angle-rotate_rad_2) * 180/np.pi)




    return (
        color, int(r), \
        (broken_center_x_1, broken_center_y_1), \
            -rand_angle-rotate_rad_1,np.pi-rand_angle-rotate_rad_1,   \
        (broken_center_x_2, broken_center_y_2), \
            np.pi-rand_angle-rotate_rad_2, 2*np.pi-rand_angle-rotate_rad_2, 
    )
    




def play(difficulty,cont_game=False):
    print('play')
    if FIRST_PLAYER == 3:
        player = np.random.choice([1,2])
    else: player = FIRST_PLAYER
    back_button = Button('<',25,25,50,50)
    
    go_back = False
    if cont_game:
        load_infos = load_continue()
        if load_infos: 
            boards, player, difficulty, remained_undo = load_infos
            board = boards[-1]
            # 이어하기를 위한 보드 초기화
            continue_boards = copy.deepcopy(boards)
        else:
            no_board_to_continue()
            return
    else:
        board = np.zeros((6,7))
        # board = [[np.random.choice([1, 2]) for _ in range(7)] for _ in range(6)]
        # 이어하기를 위한 보드 초기화
        continue_boards = [copy.deepcopy(board)]
        remained_undo = 3 
    undo_button = Button('undo {}'.format(str(remained_undo).rjust(2," ")),\
                         SCREEN.get_width()-97,500,100,50,\
                         font = pygame.font.Font('files/font/monospace_font.ttf', 20)
                         )
    recommend_button = Button('만약 AI라면...',cx=SCREEN.get_width()/2,cy=SCREEN.get_height()*3/4,width=SCREEN.get_width()/2,height=100)

    border = pygame.draw.rect(SCREEN, WHITE, (60,475,70,50))
    font = pygame.font.Font('files/font/monospace_font.ttf', 20)
    text = font.render(difficulty.ljust(8, " "), True, BLACK)
    text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    text_rect.center = border.center

    practice_border = pygame.draw.rect(SCREEN, WHITE, (60,500,70,50))
    # font = pygame.font.Font('files/font/main_font.ttf', 20)
    practice_text = font.render('practice', True, BLACK)
    practice_text_rect = practice_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    practice_text_rect.center = practice_border.center


    block_event = False
    show_recommend = False
    run = True
    event = None
    break_event, break_time = False, 0
    x, y = 50,100
    clicked_x, clicked_y = 0,0
    falling_piece = FallingInfo()
    coord_recommend = (None, None)
    SCREEN.fill(WHITE)
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] != 0:
                pos = coord2pos(SCREEN, (i,j))
                draw_circle_with_pos(pos, player=board[i][j])
    go_back = back_button.draw_and_get_event(SCREEN, event)
    undo_action = undo_button.draw_and_get_event(SCREEN, event)
    SCREEN.blit(text, text_rect)
    if remained_undo >= 4: 
        recommend_button.draw_and_get_event(SCREEN, event)
        SCREEN.blit(practice_text, practice_text_rect)
    draw_table(SCREEN)
    draw_cursor(x,player)
    pygame.display.flip()
    
    background_sound[0].set_volume(0)
    play_sound(game_sound, repeat=True, custom_volume=1)
    next_board = copy.deepcopy(board)
    while run:
        SCREEN.fill(WHITE)
        if falling_piece.stopped(): 
            if block_event: block_event = False
            board = next_board
        win_info = is_win(board,2//player)
        if win_info[0] != 0:
            print("for review")
            game_sound[0].stop()
            if win_info[0] == 3:
                save_review(continue_boards,3, difficulty)
            else:
                save_review(continue_boards,2//player, difficulty)
            end(board, win_info[0], win_info[1:], difficulty,remained_undo)
            
            return
        
        if break_event:
            color, radius, center_coord1, theta1, end_theta1, center_coord2, theta2, end_theta2 = broken_circle_info_1
            fill_arc(color, center_coord1, radius, theta1, end_theta1, break_time)
            fill_arc(color, center_coord2, radius, theta2, end_theta2, break_time)
            color, radius, center_coord1, theta1, end_theta1, center_coord2, theta2, end_theta2 = broken_circle_info_2
            fill_arc(color, center_coord1, radius, theta1, end_theta1, break_time)
            fill_arc(color, center_coord2, radius, theta2, end_theta2, break_time)
            break_time += 1
            if break_time == 30:
                break_time = 0
                break_event = False
        
        
        if player == 2 and not block_event:
            block_event = True
            col = test_main(board, player,difficulty)
            next_board, player, (drop_row, drop_col), is_valid = get_next_state(board,col,player)
            if is_valid: 
                continue_boards.append(copy.deepcopy(next_board))
                falling_piece.set_pos((drop_row,drop_col))
                falling_piece.calculate_info()
            
        for event in pygame.event.get():
            x,y = pygame.mouse.get_pos()
            if block_event: break
            if event.type == pygame.MOUSEBUTTONDOWN:
                clicked_x, clicked_y = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONUP:
                print(clicked_x)
                x,y = pygame.mouse.get_pos()
                if not is_valid_x(SCREEN, clicked_x, clicked_y, undo_button): continue
                if not is_valid_x(SCREEN, x, y, undo_button): continue
                if remained_undo>=4:
                    if not is_valid_x(SCREEN, clicked_x, clicked_y, recommend_button): continue
                    if not is_valid_x(SCREEN, x, y, recommend_button): continue

                
                coord_recommend = (None, None)
                col = x2col(SCREEN, x)
                next_board, player, (drop_row, drop_col), is_valid = get_next_state(board,col,player)
                block_event = True
                if is_valid: 
                    continue_boards.append(copy.deepcopy(next_board))
                    falling_piece.set_pos((drop_row,drop_col))
                    falling_piece.calculate_info()
                # print(board)

            
            
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                save_continue(continue_boards, player,difficulty, remained_undo)
                game_sound[0].stop()
                run = False
                sys.exit()
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        undo_action = undo_button.draw_and_get_event(SCREEN, event)
        if remained_undo >= 4: 
            SCREEN.blit(practice_text, practice_text_rect)
            show_recommend = recommend_button.draw_and_get_event(SCREEN, event)
        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            save_continue(continue_boards, player,difficulty, remained_undo)
            SCREEN.fill(WHITE)
            game_sound[0].stop()
            return
        
        if undo_action and not block_event and not break_event:
            if remained_undo==0 or len(continue_boards)<=2: pass
            else:
                coord_recommend = (None, None)
                remained_undo -= 1
                break_board = continue_boards[-1] - continue_boards[-3]
                break_pieces = np.transpose(np.nonzero(break_board))
                # print(break_pieces)
                continue_boards = continue_boards[:-2]
                board = continue_boards[-1]
                next_board = continue_boards[-1]
                undo_button.name = 'undo  {}'.format(remained_undo)
                break_event = True

                i,j=break_pieces[0]
                broken_circle_info_1 = get_broken_circle_info_with_coord((i,j), break_board[i,j])
                i,j= break_pieces[1]
                broken_circle_info_2 = get_broken_circle_info_with_coord((i,j), break_board[i,j])
                play_sound(undo_sound)

        if show_recommend and not block_event and not break_event:
            # play_sound(button_sound, repeat=False, custom_volume=1)
            if coord_recommend == (None, None):
                row = 0
                
                # ai 추천은 사람의 입장에서 진행 -> player=1
                col = test_main(board, 1, 'hard')
                for r in range(5,-1,-1):
                    if board[r][col] == 0:
                        row = r
                        break
                pos = coord2pos(SCREEN, (row,col))
                coord_recommend = pos
                play_sound(recommend_sound, repeat=False, custom_volume=1)
        
        if not falling_piece.stopped():
            falling_piece.calculate_info()
            draw_circle_with_pos(falling_piece.pos, player=2//player)
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    pos = coord2pos(SCREEN, (i,j))
                    draw_circle_with_pos(pos, player=board[i][j])
        if coord_recommend != (None, None):
            draw_circle_with_pos(coord_recommend,player=3)

        draw_table(SCREEN)
        if player==1: draw_cursor(x,player)
        elif player==2 and block_event: draw_cursor(x, 2//player)
        SCREEN.blit(text, text_rect)
        
        clock.tick(frame)
        pygame.display.flip()

def show_connect4(board, player, coords):
    run = True
    term = frame//2
    t = 0
    n = 0
    while run:
        t += 1
        SCREEN.fill(WHITE)
            
        for event in pygame.event.get():
                # print(board)

            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
                sys.exit()
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    pos = coord2pos(SCREEN, (i,j))
                    draw_circle_with_pos(pos, player=board[i][j])
        
        # coords[:n]까지 그리기
        for i in range(n):
            pos = coord2pos(SCREEN, coords[i])
            draw_circle_with_pos(pos, player=-player)
        if not t%term:
            if n<=3: play_sound(connect4_sound[n], repeat=False, custom_volume=1)
            n += 1
            
            if n == 5: return
            


        draw_table(SCREEN)
        clock.tick(frame)
        pygame.display.flip()




def end(board, player, coords, difficulty, remained_undo):
    save_continue([],None, None, None)
    is_draw = False if player in [1,2] else True
    if not is_draw: show_connect4(board, player, coords)
    w,h = SCREEN.get_size()
    
    record = load_record()

    if player == 1: 
        text_content = "이겼습니다! 축하드립니다!"
        record[difficulty][0] += 1
        play_sound(win_sound, repeat=False, custom_volume=1)
    elif player == 2: 
        text_content = "아쉽게도 졌네요 ㅠㅠ"
        record[difficulty][2] += 1
        play_sound(fail_sound, repeat=False, custom_volume=1)
    else: 
        text_content = "비겼습니다! 한번 더하면 이길지도...?"
        record[difficulty][1] += 1
        play_sound(draw_sound, repeat=False, custom_volume=1)

    if remained_undo<=3:
        save_record(record)
    print("record:",record)
    back_button = Button('back',cx=w/2,cy=h*3/4,width=w/3,height=100)

    border = pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
    font = pygame.font.Font('files/font/main_font.ttf', 30)
    text = font.render(text_content, True, BLACK)
    text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    text_rect.center = border.center

    run = True
    event = None
    go_back = False
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
                sys.exit()
        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            return
        go_back = back_button.draw_and_get_event(SCREEN, event)
        pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
        SCREEN.blit(text, text_rect)
        pygame.display.flip()

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    pos = coord2pos(SCREEN, (i,j))
                    draw_circle_with_pos(pos, player=board[i][j])

        if not is_draw:
            for i in range(4):
                pos = coord2pos(SCREEN, coords[i])
                draw_circle_with_pos(pos, player=-player)
        draw_table(SCREEN)
        

def how_to():
    # review의 페이지 레이아웃을 그대로 따라간다.
    '''
    규칙
    1. 번갈아 가면서 돌을 놓는다
    2. 돌은 아래로 떨어진다
    3. 4목을 만들면 이긴다
    '''

    '''
    애니메이션으로 만들어야 할 것들
    1. 돌을 번갈아 가면서 두는 동작
    2. 돌이 아래로 떨어지는 동작
    3. 4목을 만들 때 4목을 가리키는 동작
    '''
    print('how to')
    w,h = SCREEN.get_size()
    back_button = Button('<',25,25,50,50)
    previous_button = Button('<<',cx=w/4,cy=h*3/4,width=w/2,height=100)
    next_button = Button('>>',cx=w/4*3,cy=h*3/4,width=w/2,height=100)
    with open('files/how_to_page_1.pkl', 'rb') as file:
        boards_page_1 = pickle.load(file)
    idx, max_idx = 0, len(boards_page_1)-1
    cnt_frame = 0
    go_back, go_prev, go_next= False, False, False
    event = None
    block_event = False
    x, y = 50,100
    player = 1
    falling_piece = FallingInfo()
    draw_table(SCREEN)
    pygame.display.flip()
    page = 1  # how-to 에 사용될 페이지 
    run = True
    board = np.zeros((6,7))
    next_board = copy.deepcopy(board)
    term, t, n, dingdong = frame//2, 0, 0, False
    page_3_arr = [(5,1),(4,2),(3,3),(2,4)]
    border = pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
    font = pygame.font.Font('files/font/main_font.ttf', 30)
    text = font.render("서로 차례대로 돌을 놓습니다", True, BLACK)
    text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    text_rect.center = border.center

    while run:
        SCREEN.fill(WHITE)
        if falling_piece.stopped(): 
            if block_event: block_event = False
            board = next_board
            if board[5][2]==2 and board[5][3]==1:
                board = np.zeros((6,7))
                player = 1
            if board[4][2]==1:
                dingdong = True
                t += 1
                if not t%term:
                    if n<=3: play_sound(connect4_sound[n], repeat=False, custom_volume=1)
                    n += 1
                    
                if n==5:
                    dingdong = False
                    t, n = 0, 0
                    board[4][2]=0
                    player=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
                sys.exit()
        if page==1:
            text = font.render("서로 차례대로 돌을 놓습니다", True, BLACK)
            text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
            text_rect.center = border.center
            cnt_frame = cnt_frame+1 if cnt_frame<frame//2 else 0
            if cnt_frame == frame//2:
                idx = idx+1 if idx<max_idx else 0
            board = boards_page_1[idx]
        elif page==2:
            text = font.render("돌은 위에서 아래로 떨어집니다", True, BLACK)
            text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
            text_rect.center = border.center
            if player==1 and not block_event:
                block_event = True
                col = 3
                next_board, player, (drop_row, drop_col), is_valid = get_next_state(board,col,player)
                falling_piece.set_pos((drop_row,drop_col))
                falling_piece.calculate_info()
            elif player==2 and not block_event:
                block_event = True
                col = 2
                next_board, player, (drop_row, drop_col), is_valid = get_next_state(board,col,player)
                falling_piece.set_pos((drop_row,drop_col))
                falling_piece.calculate_info()
        elif page==3:
            text = font.render("4목을 완성하면 승리!", True, BLACK)
            text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
            text_rect.center = border.center

            if not block_event and not dingdong:
                block_event = True
                col = 2
                next_board, player, (drop_row, drop_col), is_valid = get_next_state(board,col,player)
                falling_piece.set_pos((drop_row,drop_col))
                falling_piece.calculate_info()
    
        else: break 
        go_back = back_button.draw_and_get_event(SCREEN, event)
        if page != 1:
            go_prev = previous_button.draw_and_get_event(SCREEN, event)
            
        if page != 3:
            go_next = next_button.draw_and_get_event(SCREEN, event)
            
        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            return
        if go_prev:
            play_sound(button_sound, repeat=False, custom_volume=1)
            page = page-1 if page>=1 else page
            go_prev = False
            idx = 0
            falling_piece = FallingInfo()
            board, next_board = np.zeros((6,7)), np.zeros((6,7))
            t,n, dingdong = 0,0,False
        if go_next:
            play_sound(button_sound, repeat=False, custom_volume=1)
            page = page+1 if page < 3 else page
            go_next = False
            idx = 0
            falling_piece = FallingInfo()
            board, next_board = np.zeros((6,7)), np.zeros((6,7))
            if page==3:
                player=1
                next_board = np.array([
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,2,1,0,0],
                    [0,0,0,1,1,0,0],
                    [0,2,0,1,2,0,0],
                    [0,1,2,2,2,1,0]
                ])

        if not falling_piece.stopped():
            falling_piece.calculate_info()
            draw_circle_with_pos(falling_piece.pos, player=2//player)

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    pos = coord2pos(SCREEN, (i,j))
                    draw_circle_with_pos(pos, player=board[i][j])
        
        for i in range(n):
            pos = coord2pos(SCREEN, page_3_arr[i])
            draw_circle_with_pos(pos, player=-2//player)
        draw_table(SCREEN)
        clock.tick(frame)
        SCREEN.blit(text, text_rect)
        pygame.display.flip()
            






def no_board_to_review():
    w,h = SCREEN.get_size()

    back_button = Button('back',cx=w/2,cy=h*3/4,width=w/3,height=100)

    border = pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
    font = pygame.font.Font('files/font/main_font.ttf', 30)
    text = font.render("복기할 게임이 없습니다", True, BLACK)
    text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    text_rect.center = border.center

    run = True
    event = None
    go_back = False
    
    while run:
        SCREEN.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
                sys.exit()
        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            return
        go_back = back_button.draw_and_get_event(SCREEN, event)
        SCREEN.blit(text, text_rect)
        pygame.display.flip()

def review():
    load_infos = load_review()
    w,h = SCREEN.get_size()
    idx = 0  # 배열 간 이동 
    if load_infos:
        review_boards, player, difficulty = load_infos
    else:
        print("no board to review")
        no_board_to_review()
        return
    
    print("player:",player)
    back_button = Button('<',25,25,50,50)
    previous_button = Button('<<',cx=w/4,cy=h*3/4,width=w/2,height=100)
    next_button = Button('>>',cx=w/4*3,cy=h*3/4,width=w/2,height=100)
    recommend_button = Button('만약 AI라면...',cx=w/2,cy=h*3/4+100,width=w/2,height=100)
    continue_button = Button('play from here (연습 모드)',w-177,500,260,50, font_size=20)
    
    font = pygame.font.Font('files/font/monospace_font.ttf', 30)
    border = pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
    text_content = "{} / {}".format(str(idx).rjust(2, " "), str(len(review_boards)-1).rjust(2," "))
    text = font.render(text_content, True, BLACK)
    text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    text_rect.center = border.center

    diff_border = pygame.draw.rect(SCREEN, WHITE, (50,475,70,50))
    diff_font = pygame.font.Font('files/font/main_font.ttf', 20)
    diff_text = diff_font.render(difficulty, True, BLACK)
    diff_text_rect = diff_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    diff_text_rect.center = diff_border.center


    

    if (len(review_boards)+player)%2: fp = 1
    else: fp = 0
    last_board = review_boards[-1]
    _, *last_coords = is_win(last_board, player)

    go_back, go_prev, go_next, show_recommend = False, False, False, False
    continue_action = False
    coord_recommend = (None, None)
    SCREEN.fill(WHITE)
    draw_table(SCREEN)
    run = True
    event = None
    while run:
        SCREEN.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
                sys.exit()
            
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        go_prev = previous_button.draw_and_get_event(SCREEN, event)
        go_next = next_button.draw_and_get_event(SCREEN, event)
        if (idx+fp)%2 and idx!=len(review_boards)-1:
            show_recommend = recommend_button.draw_and_get_event(SCREEN, event)
        if idx!=len(review_boards)-1:
            continue_action = continue_button.draw_and_get_event(SCREEN, event)


        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            return
        if go_prev:
            play_sound(button_sound, repeat=False, custom_volume=1)
            idx = idx-1 if idx>=1 else len(review_boards)-1
            go_prev = False
            coord_recommend = (None, None)
        if go_next:
            play_sound(button_sound, repeat=False, custom_volume=1)
            idx = idx+1 if idx<len(review_boards)-1 else 0
            go_next = False
            coord_recommend = (None, None)
        if show_recommend and (idx+fp)%2 and idx!=len(review_boards)-1:
            # play_sound(button_sound, repeat=False, custom_volume=1)
            if coord_recommend == (None, None):
                row = 0
                
                # ai 추천은 사람의 입장에서 진행 -> player=1
                col = test_main(review_boards[idx], 1, 'hard')
                for r in range(5,-1,-1):
                    if review_boards[idx][r][col] == 0:
                        row = r
                        break
                pos = coord2pos(SCREEN, (row,col))
                coord_recommend = pos
                play_sound(recommend_sound, repeat=False, custom_volume=1)
                
        
        if coord_recommend != (None, None):
            draw_circle_with_pos(coord_recommend,player=3)


        if continue_action:
            play_sound(button_sound, repeat=False, custom_volume=1)
            if coord_recommend != (None, None): 
                continue_boards = copy.deepcopy(review_boards[:idx+1])
                tmp_board = copy.deepcopy(continue_boards[-1])
                tmp_board[row][col] = 1
                continue_boards.append(tmp_board)
                print(continue_boards)
                player = 2
            
            else:
                continue_boards = copy.deepcopy(review_boards[:idx+1])
                player = 1 if (idx+fp)%2 else 2
            remained_undo = 99
            save_continue(continue_boards, player, difficulty, remained_undo)
            play(None, cont_game=True)
            return 
        

        for i in range(len(review_boards[idx])):
            for j in range(len(review_boards[idx][0])):
                if review_boards[idx][i][j] != 0:
                    pos = coord2pos(SCREEN, (i,j))
                    draw_circle_with_pos(pos, player=review_boards[idx][i][j])
            

        if player in [1,2] and idx == len(review_boards)-1:
            for coord in last_coords:
                pos = coord2pos(SCREEN, coord)
                draw_circle_with_pos(pos, player=-player)
        draw_table(SCREEN)
        text_content = "{} / {}".format(str(idx).rjust(2, " "), str(len(review_boards)-1).rjust(2," "))
        text = font.render(text_content, True, BLACK)
        SCREEN.blit(text, text_rect)
        SCREEN.blit(diff_text, diff_text_rect)
        pygame.display.flip()


def info():
    record = load_record()
    w, h = SCREEN.get_size()


    font = pygame.font.Font('files/font/monospace_font.ttf', 30)

    easy_text_content = "EASY    {} | {} | {}".format(
        str(record['easy'][0]).rjust(2, " "),
        str(record['easy'][1]).rjust(2, " "),
        str(record['easy'][2]).rjust(2, " ")
    )
    # print("easy len:",len(easy_text_content))
    easy_border = pygame.draw.rect(SCREEN, WHITE, (w/3,h/2/2,w/3,100))
    easy_text = font.render(easy_text_content, True, BLACK)
    easy_text_rect = easy_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    easy_text_rect.center = easy_border.center


    normal_text_content = "NORMAL  {} | {} | {}".format(
        str(record['normal'][0]).rjust(2, " "),
        str(record['normal'][1]).rjust(2, " "),
        str(record['normal'][2]).rjust(2, " ")
    )
    # print("norm len:",len(normal_text_content))
    normal_border = pygame.draw.rect(SCREEN, WHITE, (w/3,h/2,w/3,100))
    normal_text = font.render(normal_text_content, True, BLACK)
    normal_text_rect = normal_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    normal_text_rect.center = normal_border.center

    hard_text_content = "HARD    {} | {} | {}".format(
        str(record['hard'][0]).rjust(2, " "),
        str(record['hard'][1]).rjust(2, " "),
        str(record['hard'][2]).rjust(2, " ")
    )
    # print("hard len:",len(hard_text_content))
    hard_border = pygame.draw.rect(SCREEN, WHITE, (w/3,h/4*3,w/3,100))
    hard_text = font.render(hard_text_content, True, BLACK)
    hard_text_rect = hard_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    hard_text_rect.center = hard_border.center

    rate_font = pygame.font.Font('files/font/monospace_font.ttf', 20)

    easy_rate_text_content = str(calculate_win_rate(*record['easy'])).rjust(4, " ") + "%"
    # print("easy len:",len(easy_text_content))
    easy_rate_border = pygame.draw.rect(SCREEN, WHITE, (w/3*2+10,h/4+72,50,25))
    easy_rate_text = rate_font.render(easy_rate_text_content, True, BLACK)
    easy_rate_text_rect = easy_rate_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    easy_rate_text_rect.center = easy_rate_border.center

    normal_rate_text_content = str(calculate_win_rate(*record['normal'])).rjust(4, " ") + "%"
    # print("easy len:",len(easy_text_content))
    normal_rate_border = pygame.draw.rect(SCREEN, WHITE, (w/3*2+10,h/2+72,50,25))
    normal_rate_text = rate_font.render(normal_rate_text_content, True, BLACK)
    normal_rate_text_rect = normal_rate_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    normal_rate_text_rect.center = normal_rate_border.center

    hard_rate_text_content = str(calculate_win_rate(*record['hard'])).rjust(4, " ") + "%"
    # print("easy len:",len(easy_text_content))
    hard_rate_border = pygame.draw.rect(SCREEN, WHITE, (w/3*2+10,h/4*3+72,50,25))
    hard_rate_text = rate_font.render(hard_rate_text_content, True, BLACK)
    hard_rate_text_rect = hard_rate_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    hard_rate_text_rect.center = hard_rate_border.center


    font = pygame.font.Font('files/font/main_font.ttf', 20)
    ver_text_content = "ver {}".format(ver)
    ver_border = pygame.draw.rect(SCREEN, WHITE, (w-100,h-50,w,h))
    
    ver_text = font.render(ver_text_content, True, BLACK)
    ver_text_rect = ver_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    ver_text_rect.center = ver_border.center


    back_button = Button('<',25,25,50,50)
    go_back = False
    # 선택하면 play()    
    run = True
    event = None
    
    while run:
        SCREEN.fill((255,255,255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                sys.exit()
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            return
        SCREEN.blit(easy_text, easy_text_rect)
        SCREEN.blit(normal_text, normal_text_rect)
        SCREEN.blit(hard_text, hard_text_rect)
        SCREEN.blit(ver_text, ver_text_rect)
        SCREEN.blit(easy_rate_text, easy_rate_text_rect)
        SCREEN.blit(normal_rate_text, normal_rate_text_rect)
        SCREEN.blit(hard_rate_text, hard_rate_text_rect)
        # pygame.draw.rect(SCREEN, BLACK, (w/3*2+10,h/2/2+70,50,25))
        pygame.display.flip()

    

def setting():
    
    w, h = SCREEN.get_size()
    global FIRST_PLAYER
    global P1COLOR
    global P2COLOR
    global MUSIC_SOUND
    global EFFECT_SOUND
    
    
    # 버튼 3개 만들고
    player_dict = {
        1:'P1',
        2:'AI',
        3:'Random'
    }
    player_button = Button('FIRST PLAYER: {}'.format(player_dict[FIRST_PLAYER]),cx=w/2,cy=h/2/2,width=w-100)
    p1_color_button = Button('P1',cx=w/2-200,cy=h/2,width=100, height=100, color=P1COLOR)
    p2_color_button = Button('AI',cx=w/2+200,cy=h/2,width=100, height=100, color=P2COLOR)
    sound_button = Button('SOUND',cx=w/2,cy=h/4*3,width=w-100)
    reset_button = Button('reset setting',cx=w-70,cy=h-35,width=140,height=70, font_size=20)

    border = pygame.draw.rect(SCREEN, WHITE, (0,h/2-50,w,100))
    font = pygame.font.Font('files/font/main_font.ttf', 30)
    text = font.render('COLOR', True, BLACK)
    text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    text_rect.center = border.center



    back_button = Button('<',25,25,50,50)
    go_back = False
    # 선택하면 play()    
    run = True
    event = None
    SCREEN.fill((255,255,255))
    while run:
        SCREEN.fill((255,255,255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                sys.exit()
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        player_action = player_button.draw_and_get_event(SCREEN, event)
        p1_color_action = p1_color_button.draw_and_get_event(SCREEN, event)
        p2_color_action = p2_color_button.draw_and_get_event(SCREEN, event)
        sound_action = sound_button.draw_and_get_event(SCREEN, event)
        reset_action = reset_button.draw_and_get_event(SCREEN, event)

        SCREEN.blit(text, text_rect)
        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            save_setting({
                'first_player':FIRST_PLAYER,
                'p1_color':P1COLOR,
                'p2_color':P2COLOR,
                'music':MUSIC_SOUND,
                'effect':EFFECT_SOUND
            })
            SCREEN.fill(WHITE)
            # 설정을 저장하는 코드

            return
        if player_action: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            player_setting()
            player_button.name = 'FIRST PLAYER: {}'.format(player_dict[FIRST_PLAYER])
            
        if p1_color_action: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            p1_color_setting()
            p1_color_button.change_color(P1COLOR)
            
        if p2_color_action: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            p2_color_setting()
            p2_color_button.change_color(P2COLOR)
            
        if sound_action: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            sound_setting()

        if reset_action:
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)

            with open('files/default_setting.pkl', 'rb') as file:
                setting = pickle.load(file)
            save_setting(setting)
            
            FIRST_PLAYER = setting['first_player']
            P1COLOR = setting['p1_color']
            P2COLOR = setting['p2_color']
            MUSIC_SOUND = setting['music']
            EFFECT_SOUND = setting['effect']
            
            player_button.name = 'FIRST PLAYER: {}'.format(player_dict[FIRST_PLAYER])
            p1_color_button.change_color(P1COLOR)
            p2_color_button.change_color(P2COLOR)

            background_sound[0].set_volume(background_sound[1] * MUSIC_SOUND/3)

        pygame.display.flip()


def player_setting():
    w, h = SCREEN.get_size()
    p1_button = Button('Player 1',cx=w/2,cy=h/4,width=w/3)
    ai_button = Button('AI',cx=w/2,cy=h/2,width=w/3)
    random_button = Button('Random',cx=w/2,cy=h/4*3,width=w/3)

    global FIRST_PLAYER


    back_button = Button('<',25,25,50,50)
    go_back = False
    # 선택하면 play()    
    run = True
    event = None
    SCREEN.fill((255,255,255))
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                sys.exit()
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        p1_action = p1_button.draw_and_get_event(SCREEN, event)
        ai_action = ai_button.draw_and_get_event(SCREEN, event)
        random_action = random_button.draw_and_get_event(SCREEN, event)
        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            return
        if p1_action: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            FIRST_PLAYER = 1
            return
        if ai_action: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            FIRST_PLAYER = 2
            return
        if random_action: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            FIRST_PLAYER = 3
            return
        pygame.display.flip()

def p1_color_setting():
    w, h = SCREEN.get_size()

    global P1COLOR

    back_button = Button('<',25,25,50,50)
    random_button = Button('random',w-97,500,100,50, font_size=20)
    go_back, random_action = False, False
    # 선택하면 play()    
    run = True
    event = None
    mousepressed = False
    SCREEN.fill((255,255,255))
    pygame.display.flip()
    while run:
        for event in pygame.event.get():
            x, y = pygame.mouse.get_pos()
            if x<50 or x>w-50: continue
            color = SCREEN.get_at((x,y))
            if event.type == pygame.QUIT:
                run = False
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                r,g,b,_ = color
                if (r,g,b)!=(0,0,0) and (r,g,b)!=(255,255,255) and (r,g,b)!=(180,180,180):
                    mousepressed = True
            if event.type == pygame.MOUSEBUTTONUP and mousepressed:
                mousepressed = False
                r2,g2,b2,_ = color
                if (r,g,b) == (r2,g2,b2):
                    select_color_sound[0].stop()
                    P1COLOR = (r,g,b)
                    play_sound(select_color_sound)
            # print(color)
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        random_action = random_button.draw_and_get_event(SCREEN, event)
        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            return
        if random_action:
            select_color_sound[0].stop()
            r,g,b = np.random.randint(256,size=3)
            P1COLOR = (r,g,b)
            play_sound(select_color_sound)
        for idx in range(42):
            row, col = idx//7, idx%7
            x,y = coord2pos(SCREEN, (row,col))
            x += 0.5
            y += 0.5
            pygame.draw.circle(SCREEN,color_picker[idx],(x,y),(w-100)/7/2/1.05)

        pygame.draw.circle(SCREEN,P1COLOR,(w/2,h/4*3),w/4)
        draw_table(SCREEN)
        pygame.display.flip()


def p2_color_setting():
    w, h = SCREEN.get_size()

    global P2COLOR

    back_button = Button('<',25,25,50,50)
    random_button = Button('random',w-97,500,100,50, font_size=20)
    go_back, random_action = False, False
    # 선택하면 play()    
    run = True
    event = None
    mousepressed = False
    SCREEN.fill((255,255,255))
    pygame.display.flip()
    while run:
        for event in pygame.event.get():
            x, y = pygame.mouse.get_pos()
            if x<50 or x>w-50: continue
            color = SCREEN.get_at((x,y))
            if event.type == pygame.QUIT:
                run = False
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                r,g,b,_ = color
                if (r,g,b)!=(0,0,0) and (r,g,b)!=(255,255,255) and (r,g,b)!=(180,180,180):
                    mousepressed = True
            if event.type == pygame.MOUSEBUTTONUP and mousepressed:
                mousepressed = False
                r2,g2,b2,_ = color
                if (r,g,b) == (r2,g2,b2):
                    select_color_sound[0].stop()
                    P2COLOR = (r,g,b)
                    play_sound(select_color_sound)
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        random_action = random_button.draw_and_get_event(SCREEN, event)
        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            SCREEN.fill(WHITE)
            return
        if random_action:
            select_color_sound[0].stop()
            r,g,b = np.random.randint(256,size=3)
            P2COLOR = (r,g,b)
            play_sound(select_color_sound)
        for idx in range(42):
            row, col = idx//7, idx%7
            x,y = coord2pos(SCREEN, (row,col))
            x += 0.5
            y += 0.5
            pygame.draw.circle(SCREEN,color_picker[idx],(x,y),(w-100)/7/2/1.05)

        pygame.draw.circle(SCREEN,P2COLOR,(w/2,h/4*3),w/4)
        draw_table(SCREEN)
        pygame.display.flip()


def sound_setting():
    w,h = SCREEN.get_size()
    back_button = Button('<',25,25,50,50)
    music_down_button = Button('-',cx=w/2-200,cy=h/4*3-100,width=100, height=100)
    music_up_button = Button('+',cx=w/2+200,cy=h/4*3-100,width=100, height=100)
    effect_down_button = Button('-',cx=w/2-200,cy=h/4*3,width=100, height=100)
    effect_up_button = Button('+',cx=w/2+200,cy=h/4*3,width=100, height=100)
    
    global MUSIC_SOUND
    global EFFECT_SOUND

    falling_piece = FallingInfo()
    draw_table(SCREEN)
    run = True
    board = np.zeros((6,7))
    next_board = copy.deepcopy(board)
    block_event = False
    event = None
    player = 1

    background_sound[0].set_volume(0)
    play_sound(game_sound, repeat=True, custom_volume=1)


    
    font = pygame.font.Font('files/font/main_font.ttf', 30)

    music_border = pygame.draw.rect(SCREEN, BLACK, (w/2-150,h/4*3-150,300,100))
    music_text = font.render('MUSIC', True, BLACK)
    music_text_rect = music_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    music_text_rect.center = music_border.center

    effect_border = pygame.draw.rect(SCREEN, WHITE, (w/2-150,h/4*3-50,300,100))
    effect_text = font.render('EFFECT', True, BLACK)
    effect_text_rect = effect_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    effect_text_rect.center = effect_border.center

    while run:
        SCREEN.fill(WHITE)
        if falling_piece.stopped(): 
            if block_event: block_event = False
            board = next_board
            if board[5][2]==2 and board[5][3]==1:
                board = np.zeros((6,7))
                player = 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
                sys.exit()

        if player==1 and not block_event:
            block_event = True
            col = 3
            next_board, player, (drop_row, drop_col), is_valid = get_next_state(board,col,player)
            falling_piece.set_pos((drop_row,drop_col))
            falling_piece.calculate_info()
        elif player==2 and not block_event:
            block_event = True
            col = 2
            next_board, player, (drop_row, drop_col), is_valid = get_next_state(board,col,player)
            falling_piece.set_pos((drop_row,drop_col))
            falling_piece.calculate_info()


        SCREEN.blit(music_text, music_text_rect)
        SCREEN.blit(effect_text, effect_text_rect)
        go_back = back_button.draw_and_get_event(SCREEN, event)
        md_action = music_down_button.draw_and_get_event(SCREEN, event)
        mu_action = music_up_button.draw_and_get_event(SCREEN, event)
        ed_action = effect_down_button.draw_and_get_event(SCREEN, event)
        eu_action = effect_up_button.draw_and_get_event(SCREEN, event)
        if MUSIC_SOUND > 0:
            pygame.draw.line(SCREEN, P1COLOR, (w/2-150,h/4*3-54), (w/2-151 + 300*MUSIC_SOUND,h/4*3-54), 6)
        if EFFECT_SOUND > 0:
            pygame.draw.line(SCREEN, P2COLOR, (w/2-150,h/4*3+47), (w/2-151 + 300*EFFECT_SOUND,h/4*3+47), 5)
        
        if md_action: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            MUSIC_SOUND -= 0.1
            if MUSIC_SOUND<0.05: MUSIC_SOUND=0
            game_sound[0].set_volume(game_sound[1] * MUSIC_SOUND)
            # print(MUSIC_SOUND)
        if mu_action: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            MUSIC_SOUND += 0.1
            if MUSIC_SOUND>0.95: MUSIC_SOUND=1 
            game_sound[0].set_volume(game_sound[1] * MUSIC_SOUND)
            # print(MUSIC_SOUND)

        if ed_action: 
            EFFECT_SOUND -= 0.1
            if EFFECT_SOUND<0.05: EFFECT_SOUND=0
            play_sound(button_sound, repeat=False, custom_volume=1)
        if eu_action: 
            EFFECT_SOUND += 0.1
            if EFFECT_SOUND>0.95: EFFECT_SOUND=1
            play_sound(button_sound, repeat=False, custom_volume=1)
        if go_back: 
            play_sound(button_sound, repeat=False, custom_volume=1)
            game_sound[0].stop()
            background_sound[0].set_volume(background_sound[1]*MUSIC_SOUND/3)
            SCREEN.fill(WHITE)
            return
        if not falling_piece.stopped():
            falling_piece.calculate_info()
            draw_circle_with_pos(falling_piece.pos, player=2//player)

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    pos = coord2pos(SCREEN, (i,j))
                    draw_circle_with_pos(pos, player=board[i][j])

        draw_table(SCREEN)
        clock.tick(frame)
        pygame.display.flip()


def ee():
    back_button = Button('back',cx=SCREEN.get_width()/2,cy=SCREEN.get_height()*3/4,width=SCREEN.get_width()/2,height=100)
    run = True
    t = 0
    idx = 0
    event = None
    board = ee_boards[idx]
    bg_volume = background_sound[0].get_volume()
    background_sound[0].set_volume(0)
    play_sound(ee_sound, repeat=True)
    while run:
        SCREEN.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
                sys.exit()

        go_back = back_button.draw_and_get_event(SCREEN, event)
        if go_back:
            ee_sound[0].stop()
            play_sound(button_sound, repeat=False, custom_volume=1)
            background_sound[0].set_volume(bg_volume)
            SCREEN.fill(WHITE)
            return
        
        t += 1
        if t%60==0:
            idx = idx+1 if idx<len(ee_boards)-1 else 0
            board = ee_boards[idx]
            t = 0

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    pos = coord2pos(SCREEN, (i,j))
                    draw_circle_with_pos(pos, player=board[i][j])
        draw_table(SCREEN)
        clock.tick(frame)
        pygame.display.flip()



    

# 이미지 로드
# pygame.image.load(image_file)
# pygame.transform.scale(image object, (width, height)) 

# SCREEN.fill((R,G,B))
# SCREEN.fill( (255, 255, 255) )

# 이미지를 화면에 출력 
# SCREEN.blit(image object,  coord or Rect)

# 선 그리기
# pygame.draw.line(SCREEN, color, start_pos, end_pos, width=1) -> Rect
# 선 여러개 그리기
# pygame.draw.lines(SCREEN, color, closed, points, width=1) -> Rect
# 부드러운 선
# pygame.draw.aaline(SCREEN, color, start_pos, end_pos, blend=1) -> Rect
# 부드러운 선 여러개 그리기
# pygame.draw.aaline(SCREEN, color, closed, points, blend=1) -> Rect
# 네모 그리기
# pygame.draw.rect(SCREEN, color, rect, width=0) -> Rect
# 다각형 그리기
# pygame.draw.polygon(surface, color, points, width=0) -> Rect
# 원 그리기
# pygame.draw.circle(surface, color, center, radius, width=0) -> Rect
# 타원 그리기
# pygame.draw.ellipse(surface, color, rect, width=0) -> Rect

# 충돌 체크 방법
'''
def collision_check(A, B):
    if A.top < B.bottom and B.top < A.bottom and A.left < B.right and B.left < A.right:
        return True
    else:
        return False

A_rect = A_object.get_rect()
B_rect  = B_object.get_rect()

collision_check(A_rect, B_rect)
'''

# 사각형을 이미지로 이용하기 위해서
# rect_image = pygame.Surface((10,20))
# rect_image.fill(RED)
# 이미지의 Rect 정보를 저장

# player_Rect = rect_image.get_rect()


# print( pygame.font.get_fonts() )
# 폰트 설정
# my_font = pygame.font.SysFont('arial', 30, bold=False, italic=False)

# render(text,antialias, color, background)
# text_Title = my_font.render("Pygame Text Test", True, BLACK)
# 이미지가 가운데 올 수 있도록 좌표값 수정
# python 3.8 이상에서 integer가 필요한 곳에 float가 들어가면 DeprecationWarning이 나옴.
# 따라서 round() 처리를 해준다.
# player_Rect.centerx = round(SCREEN_WIDTH / 2)
# player_Rect.centery = round(SCREEN_HEIGHT / 2)

# 화면 업데이트
# pygame.display.update()
# pygame.display.filp()
pygame.display.set_caption('Connect 4 by Team Connect X of TAVE 11th')
def main():
    intro()
    

if __name__ == '__main__':
    main()
