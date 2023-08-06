import pygame
import sys
import numpy as np
import os
import copy
import pickle
from classes import *
from test_model import test_main




pygame.init()



SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
ver = 0.5  # version
# Clock 객체 생성
clock = pygame.time.Clock()
frame = 60
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
    def set_pos(self, target_cord):
        self.bounce = 0
        self.v = 0
        target_row, target_col = target_cord
        self.target_pos = cord2pos((target_row,target_col))
        # 바닥은 (5,col) 보다 낮은 (6,col)
        # 떨어지기 시작하는 위치는 (0,col) 보다 높은 (-1, col) 로 설정 
        self.base_pos = cord2pos((6,target_col))  
        self.pos = cord2pos((-1,target_col))  

    def calculate_info(self):
        self.v += self.g
        self.pos[1] += self.v

        if self.target_pos[1] < self.pos[1] and self.v > 0:
            
            
            self.bounce += 1
            if self.bounce==1 and self.max_v==0: self.max_v = self.v
            drop_sound.set_volume(self.v/self.max_v)
            drop_sound.play()
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


# continue.pkl을 불러온다. 없다면 False 리턴 
def load_continue():
    if not os.path.isfile('files/continue.pkl'):
        return []
    else:
        with open('files/continue.pkl', 'rb') as file:
        # pickle.load() 함수를 사용하여 객체를 로드합니다.
            return pickle.load(file)

# 뒤로 가기 등을 눌렀을 때 continue_states 배열을 저장
def save_continue(continue_states, player, difficulty):
    if continue_states:
        save_infos = [continue_states,player, difficulty]
    else: save_infos = []
    with open('files/continue.pkl', 'wb') as file:
        # pickle.dump() 함수를 사용하여 객체를 저장합니다.
        pickle.dump(save_infos, file)

def load_review():
    if not os.path.isfile('files/review.pkl'):
        return []
    else:
        with open('files/review.pkl', 'rb') as file:
        # pickle.load() 함수를 사용하여 객체를 로드합니다.
            return pickle.load(file)

def save_review(states, player, difficulty):
    save_infos = [states,player, difficulty]
    with open('files/review.pkl', 'wb') as file:
        # pickle.dump() 함수를 사용하여 객체를 저장합니다.
        pickle.dump(save_infos, file)

'''
게임 기록을 불러오고 저장하는 함수
record = {
    'easy':[win, draw, lose],
    'normal':[win, draw, lose],
    'hard':[win, draw, lose]
}
'''
def load_record():
    if not os.path.isfile('files/record.pkl'):
        init_record = {
            'easy':[0,0,0],
            'normal':[0,0,0],
            'hard':[0,0,0]
        }
        with open('files/record.pkl', 'wb') as file:
            pickle.dump(init_record, file)
        return init_record
    else:
        with open('files/record.pkl', 'rb') as file:
        # pickle.load() 함수를 사용하여 객체를 로드합니다.
            return pickle.load(file)

def save_record(record):
    with open('files/record.pkl', 'wb') as file:
        # pickle.dump() 함수를 사용하여 객체를 저장합니다.
        pickle.dump(record, file)



            

def intro():
    w, h = SCREEN.get_size()
    intro_button_width = w//3
    intro_button_height = h//16
    print(w,h,intro_button_width,intro_button_height)

    background_sound.play(-1)
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
    


    
    run = True
    event = None
    SCREEN.fill((255,255,255))
    
    while run:
        if not music_on: 
            background_sound.set_volume(0.3)
            music_on = True
        SCREEN.blit(logo_image, logo_rect)
        setting_action = setting_button.draw_and_get_event(SCREEN,event)
        SCREEN.blit(setting_image, setting_rect)
        for button in intro_buttons:
            action = button.draw_and_get_event(SCREEN,event)
            if action:
                print(button.name)
                music_on = False
                background_sound.set_volume(0.1)
                if button.name == 'new game': select_difficulty()
                elif button.name == 'continue': play(difficulty=None, cont_game=True)
                elif button.name == 'how to': how_to()
                elif button.name == 'review': review()
                elif button.name == 'info': info()
                elif button.name == 'quit': run=False
                else:
                    print('error')
        
        if setting_action: setting()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        clock.tick(frame)
        pygame.display.flip()


    pygame.quit()
    sys.exit()


def no_setting():
    w,h = SCREEN.get_size()

    back_button = Button('back',cx=w/2,cy=h*3/4,width=w/3,height=100)

    border = pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
    font = pygame.font.Font('files/font/main_font.ttf', 30)
    text = font.render("아직 구현이 안됐습니다ㅠ", True, BLACK)
    text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    text_rect.center = border.center

    run = True
    event = None
    go_back = False
    SCREEN.fill(WHITE)
    while run:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
        if go_back: 
            SCREEN.fill(WHITE)
            return
        go_back = back_button.draw_and_get_event(SCREEN, event)
        SCREEN.blit(text, text_rect)
        pygame.display.flip()



def draw_table():
    w,h = SCREEN.get_size()
    block_size = (w-100)/7
    for i in range(8):
        pygame.draw.line(SCREEN, BLACK, (50+block_size*i,100), (50+block_size*i,100+6*block_size), width=5)
    for i in range(7):
        pygame.draw.line(SCREEN, BLACK, (50,100+block_size*i), (50+block_size*7,100+block_size*i), width=5)


def draw_cursor(x, player):
    if player == 1: color = P1COLOR
    else: color = P2COLOR
    pygame.draw.polygon(SCREEN, color, ((x,90),(x-10,80),(x+10,80)), width=0)

def draw_circle_with_pos(pos,player):
    w,h = SCREEN.get_size()
    r = (w-100)/7/2/1.05
    if player in [1,-1]: color = P1COLOR
    elif player in [2,-2]: color = P2COLOR
    else:
        R1, G1, B1 = P1COLOR
        R2, G2, B2 = P2COLOR
        color = (255-(R1+R2)//2, 255-(G1+G2)//2, 255-(B1+B2)//2)
    x,y = pos
    x += 0.5
    y += 0.5
    pos = (x,y)
    pygame.draw.circle(SCREEN,color,pos,r)
    if player < 0:
        pygame.draw.circle(SCREEN,WHITE,pos,r*0.8)


# board 상의 좌표를 SCREEN의 좌표로 변경
def cord2pos(cord):
    y,x = cord
    # y = 5-y
    w,h = SCREEN.get_size()
    r = (w-100)/7/2

    # (0,0) 일 때의 position
    pos = [50+r,100+r]
    pos[0] += 2*x*r
    pos[1] += 2*y*r

    return pos

def x2col(x):
    w,h = SCREEN.get_size()
    block_size = (w-100)/7
    x -= 50
    col = -1

    while x >= 0:
        x -= block_size
        col += 1

    if not 0<=col<=6:
        col = -1
    return col

def is_valid_x(x):
    w,h = SCREEN.get_size()
    if x<50 or w-50<x:
        return False
    else: return True


# (board, column, player)를 받아서  (next board, next player, pos, valid move) 를 리턴 
def get_next_state(board, col,player):
    if col == -1: 
        return board, player, (None, None), False
    if board[0][col] != 0:
        return board, player, (None, None), False
    
    next_board = copy.deepcopy(board)
    for row in range(5,-1,-1):
        if next_board[row][col] == 0:
            next_board[row][col] = player
            return next_board, 2//player, (row, col), True

# made by chatgpt and I edit little bit.
# 가로, 세로, 대각선에 완성된 줄이 있는지를 체크한다.
# 연결된 4개를 표시하기 위해 return을 수정하였다. 
# return (이긴 플레이어, 좌표1, 좌표2, 좌표3, 좌표4)
def is_win(board, player):
    for i in range(6):
        for j in range(7):
            if board[i][j] == player:
                # horizontal
                if j + 3 < 7 and board[i][j+1] == board[i][j+2] == board[i][j+3] == player:
                    return player, (i,j), (i,j+1), (i,j+2), (i,j+3)
                # vertical
                if i + 3 < 6 and board[i+1][j] == board[i+2][j] == board[i+3][j] == player:
                    return player, (i,j), (i+1,j), (i+2,j), (i+3,j)
                # diagonal (down right)
                if i + 3 < 6 and j + 3 < 7 and board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3] == player:
                    return player, (i,j), (i+1,j+1), (i+2,j+2), (i+3,j+3)
                # diagonal (up right)
                if i - 3 >= 0 and j + 3 < 7 and board[i-1][j+1] == board[i-2][j+2] == board[i-3][j+3] == player:
                    return player, (i,j), (i-1,j+1), (i-2,j+2), (i-3,j+3)
                

    if 0 not in board[0,:]: 
        player = 3
        return player, None, None, None, None
    return 0, None, None, None, None

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
        if easy_action:
            play(difficulty='easy')
            return
        elif normal_action:
            play(difficulty='normal')
            return
        elif hard_action:
            play(difficulty='hard')
            return
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        if go_back: 
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
    SCREEN.fill(WHITE)
    while run:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
        if go_back: 
            SCREEN.fill(WHITE)
            return
        go_back = back_button.draw_and_get_event(SCREEN, event)
        SCREEN.blit(text, text_rect)
        pygame.display.flip()





def play(difficulty,cont_game=False):
    print('play')
    player = np.random.choice([1,2])
    back_button = Button('<',25,25,50,50)
    go_back = False
    if cont_game:
        load_infos = load_continue()
        if load_infos: 
            boards, player, difficulty = load_infos
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

    
    block_event = False
    run = True
    event = None
    x, y = 50,100
    clicked_x, clicked_y = 0,0
    falling_piece = FallingInfo()
    SCREEN.fill(WHITE)
    draw_table()
    draw_cursor(x,player)
    pygame.display.flip()
    background_sound.set_volume(0)
    game_sound.play(-1)
    next_board = copy.deepcopy(board)
    while run:
        SCREEN.fill(WHITE)
        if falling_piece.stopped(): 
            if block_event: block_event = False
            board = next_board
        win_info = is_win(board,2//player)
        if win_info[0] != 0:
            print("for review")
            game_sound.stop()
            save_review(continue_boards,2//player, difficulty)
            end(board, win_info[0], win_info[1:], difficulty)
            
            return
        
        
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
                if not is_valid_x(clicked_x): continue
                x,_ = pygame.mouse.get_pos()
                col = x2col(x)
                next_board, player, (drop_row, drop_col), is_valid = get_next_state(board,col,player)
                block_event = True
                if is_valid: 
                    continue_boards.append(copy.deepcopy(next_board))
                    falling_piece.set_pos((drop_row,drop_col))
                    falling_piece.calculate_info()
                # print(board)

            
            
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                save_continue(continue_boards, player,difficulty)
                game_sound.stop()
                run = False
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        if go_back: 
            save_continue(continue_boards, player,difficulty)
            SCREEN.fill(WHITE)
            game_sound.stop()
            return
        
        if not falling_piece.stopped():
            falling_piece.calculate_info()
            draw_circle_with_pos(falling_piece.pos, player=2//player)
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    pos = cord2pos((i,j))
                    draw_circle_with_pos(pos, player=board[i][j])
        draw_table()
        if player==1: draw_cursor(x,player)
        elif player==2 and block_event: draw_cursor(x, 2//player)
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
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    pos = cord2pos((i,j))
                    draw_circle_with_pos(pos, player=board[i][j])
        
        # coords[:n]까지 그리기
        for i in range(n):
            pos = cord2pos(coords[i])
            draw_circle_with_pos(pos, player=-player)
        if not t%term:
            if n<=3: connect4_sound[n].play()
            n += 1
            
            if n == 5: return
            


        draw_table()
        clock.tick(frame)
        pygame.display.flip()




def end(board, player, coords, difficulty):
    save_continue([],None, None)
    show_connect4(board, player, coords)
    w,h = SCREEN.get_size()

    record = load_record()

    if player == 1: 
        text_content = "이겼습니다! 축하드립니다!"
        record[difficulty][0] += 1
        win_sound.play()
    elif player == 2: 
        text_content = "아쉽게도 졌네요 ㅠㅠ"
        record[difficulty][2] += 1
        fail_sound.play()
    else: 
        text_content = "비겼습니다! 한번 더하면 이길지도...?"
        record[difficulty][1] += 1
        draw_sound.play()

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
        if go_back: 
            SCREEN.fill(WHITE)
            return
        go_back = back_button.draw_and_get_event(SCREEN, event)
        SCREEN.blit(text, text_rect)
        pygame.display.flip()

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    pos = cord2pos((i,j))
                    draw_circle_with_pos(pos, player=board[i][j])

        for i in range(4):
            pos = cord2pos(coords[i])
            draw_circle_with_pos(pos, player=-player)
        draw_table()
        

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
    draw_table()
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
                    if n<=3: connect4_sound[n].play()
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
            SCREEN.fill(WHITE)
            return
        if go_prev:
            page = page-1 if page>=1 else page
            go_prev = False
            idx = 0
            falling_piece = FallingInfo()
            board, next_board = np.zeros((6,7)), np.zeros((6,7))
            t,n, dingdong = 0,0,False
        if go_next:
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
                    pos = cord2pos((i,j))
                    draw_circle_with_pos(pos, player=board[i][j])
        
        for i in range(n):
            pos = cord2pos(page_3_arr[i])
            draw_circle_with_pos(pos, player=-2//player)
        draw_table()
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
    SCREEN.fill(WHITE)
    while run:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
        if go_back: 
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
    back_button = Button('<',25,25,50,50)
    previous_button = Button('<<',cx=w/4,cy=h*3/4,width=w/2,height=100)
    next_button = Button('>>',cx=w/4*3,cy=h*3/4,width=w/2,height=100)
    recommend_button = Button('만약 AI라면...',cx=w/2,cy=h*3/4+100,width=w/2,height=100)
    font = pygame.font.Font('files/font/main_font.ttf', 30)

    border = pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
    text_content = "{} / {}".format(idx, len(review_boards)-1)
    text = font.render(text_content, True, BLACK)
    text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    text_rect.center = border.center


    

    if (len(review_boards)+player)%2: fp = 1
    else: fp = 0
    last_board = review_boards[-1]
    _, *last_coords = is_win(last_board, player)

    go_back, go_prev, go_next, show_recommend= False, False, False, False
    cord_recommend = (None, None)
    SCREEN.fill(WHITE)
    draw_table()
    run = True
    event = None
    while run:
        SCREEN.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
            
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        if idx != 0:
            go_prev = previous_button.draw_and_get_event(SCREEN, event)
        if idx != len(review_boards)-1:
            go_next = next_button.draw_and_get_event(SCREEN, event)
        if (idx+fp)%2 and idx!=len(review_boards)-1:
            show_recommend = recommend_button.draw_and_get_event(SCREEN, event)
        if go_back: 
            SCREEN.fill(WHITE)
            return
        if go_prev:
            idx = idx-1 if idx>=1 else idx
            go_prev = False
            cord_recommend = (None, None)
        if go_next:
            idx = idx+1 if idx<len(review_boards)-1 else idx 
            go_next = False
            cord_recommend = (None, None)
        if show_recommend and (idx+fp)%2 and idx!=len(review_boards)-1:
            if cord_recommend == (None, None):
                row = 0
                
                # ai 추천은 사람의 입장에서 진행 -> player=1
                col = test_main(review_boards[idx], 1, 'hard')
                for r in range(5,-1,-1):
                    if review_boards[idx][r][col] == 0:
                        row = r
                        break
                pos = cord2pos((row,col))
                cord_recommend = pos
                recommend_sound.play()
        
        if cord_recommend != (None, None):
            draw_circle_with_pos(cord_recommend,player=3)

        for i in range(len(review_boards[idx])):
            for j in range(len(review_boards[idx][0])):
                if review_boards[idx][i][j] != 0:
                    pos = cord2pos((i,j))
                    draw_circle_with_pos(pos, player=review_boards[idx][i][j])
        if idx == len(review_boards)-1:
            for coord in last_coords:
                pos = cord2pos(coord)
                draw_circle_with_pos(pos, player=-player)
        draw_table()
        text_content = "{} / {}".format(idx, len(review_boards)-1)
        text = font.render(text_content, True, BLACK)
        SCREEN.blit(text, text_rect)
        pygame.display.flip()


def info():
    record = load_record()
    w, h = SCREEN.get_size()
    # 버튼 3개 만들고
    easy_button = Button('easy',cx=w/2,cy=h/2/2,width=w/3)
    normal_button = Button('normal',cx=w/2,cy=h/2,width=w/3)
    hard_button = Button('hard',cx=w/2,cy=h/4*3,width=w/3)


    font = pygame.font.Font('files/font/main_font.ttf', 30)

    easy_text_content = "  EASY  {} / {} / {}".format(record['easy'][0],record['easy'][1],record['easy'][2])
    easy_border = pygame.draw.rect(SCREEN, WHITE, (w/3,h/2/2,w/3,100))
    
    easy_text = font.render(easy_text_content, True, BLACK)
    easy_text_rect = easy_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    easy_text_rect.center = easy_border.center

    normal_text_content = "NORMAL  {} / {} / {}".format(record['normal'][0],record['normal'][1],record['normal'][2])
    normal_border = pygame.draw.rect(SCREEN, WHITE, (w/3,h/2,w/3,100))
    
    normal_text = font.render(normal_text_content, True, BLACK)
    normal_text_rect = normal_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    normal_text_rect.center = normal_border.center

    hard_text_content = "  HARD  {} / {} / {}".format(record['hard'][0],record['hard'][1],record['hard'][2])
    hard_border = pygame.draw.rect(SCREEN, WHITE, (w/3,h/4*3,w/3,100))
    
    hard_text = font.render(hard_text_content, True, BLACK)
    hard_text_rect = hard_text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    hard_text_rect.center = hard_border.center


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
    SCREEN.fill((255,255,255))
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        if go_back: 
            SCREEN.fill(WHITE)
            return
        SCREEN.blit(easy_text, easy_text_rect)
        SCREEN.blit(normal_text, normal_text_rect)
        SCREEN.blit(hard_text, hard_text_rect)
        SCREEN.blit(ver_text, ver_text_rect)
        pygame.display.flip()

    

# 아직 setting 구현 x 
def setting():
    no_setting()
    return




# 이미지 로드
# pygame.image.load(image_file)
# pygame.transform.scale(image object, (width, height)) 

# SCREEN.fill((R,G,B))
# SCREEN.fill( (255, 255, 255) )

# 이미지를 화면에 출력 
# SCREEN.blit(image object,  cord or Rect)

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

dx, dy = 0, 0
# 화면 업데이트
# pygame.display.update()
# pygame.display.filp()
pygame.display.set_caption('Connect 4 by Team Connect X of TAVE 11th')
def main():
    intro()
    

if __name__ == '__main__':
    main()
