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

# Clock 객체 생성
clock = pygame.time.Clock()

# pygame.Rect(x,y,width, height)
# myRect = pygame.Rect(150, 200, 200, 100)

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



            

def intro():
    w, h = SCREEN.get_size()
    intro_button_width = w//3
    intro_button_height = h//16
    print(w,h,intro_button_width,intro_button_height)


    intro_buttons = [
        Button('new game',cx=w/2,cy=h/2+intro_button_height*1,width=intro_button_width, height=intro_button_height),
        Button('continue',cx=w/2,cy=h/2+intro_button_height*2,width=intro_button_width, height=intro_button_height),
        Button('how to',cx=w/2,cy=h/2+intro_button_height*3,width=intro_button_width, height=intro_button_height),
        Button('review',cx=w/2,cy=h/2+intro_button_height*4,width=intro_button_width, height=intro_button_height),
        Button('option',cx=w/2,cy=h/2+intro_button_height*5,width=intro_button_width, height=intro_button_height),
        Button('quit',cx=w/2,cy=h/2+intro_button_height*6,width=intro_button_width, height=intro_button_height)
    ]
    logo_image = pygame.image.load('image/logo.png')
    logo_image = pygame.transform.scale(logo_image, (w/1.1, w/1.1/4))
    logo_rect = logo_image.get_rect()
    logo_rect.x = w/2- logo_image.get_width()/2
    logo_rect.y = h/4- logo_image.get_height()/2
    
    run = True
    SCREEN.fill((255,255,255))
    while run:
        SCREEN.blit(logo_image, logo_rect)
        for event in pygame.event.get():
            
            for button in intro_buttons:
                action = button.draw_and_get_event(SCREEN,event)
                if action:
                    print(button.name)
                    if button.name == 'new game': select_difficulty()
                    elif button.name == 'continue': play(difficulty=None, cont_game=True)
                    elif button.name == 'how to': how_to()
                    elif button.name == 'review': review()
                    elif button.name == 'option': option()
                    elif button.name == 'quit': run=False
                    else:
                        print('error')
            if event.type == pygame.QUIT:
                run = False
        clock.tick(60)
        pygame.display.flip()


    pygame.quit()
    sys.exit()

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
    if player == 1: color = P1COLOR
    elif player == 2: color = P2COLOR
    else:
        R1, G1, B1 = P1COLOR
        R2, G2, B2 = P2COLOR
        color = (255-(R1+R2)//2, 255-(G1+G2)//2, 255-(B1+B2)//2)
    x,y = pos
    x += 0.5
    y += 0.5
    pos = (x,y)
    pygame.draw.circle(SCREEN,color,pos,r)


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


# (board, column, player)를 받아서  (next board, next player, valid move) 를 리턴 
def get_next_state(board, col,player):
    if col == -1: 
        return board, player, False
    if board[0][col] != 0:
        return board, player, False
    
    for row in range(5,-1,-1):
        if board[row][col] == 0:
            board[row][col] = player
            return board, 2//player, True

# made by chatgpt and I edit little bit.
# 가로, 세로, 대각선에 완성된 줄이 있는지를 체크한다 
def is_win(board, player):
    for i in range(6):
        for j in range(7):
            if board[i][j] == player:
                # horizontal
                if j + 3 < 7 and board[i][j+1] == board[i][j+2] == board[i][j+3] == player:
                    return player
                # vertical
                if i + 3 < 6 and board[i+1][j] == board[i+2][j] == board[i+3][j] == player:
                    return player
                # diagonal (down right)
                if i + 3 < 6 and j + 3 < 7 and board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3] == player:
                    return player
                # diagonal (up right)
                if i - 3 >= 0 and j + 3 < 7 and board[i-1][j+1] == board[i-2][j+2] == board[i-3][j+3] == player:
                    return player
                

    if 0 not in board[0,:]: player = 3
    return 0

def select_difficulty():
    w, h = SCREEN.get_size()
    # 버튼 3개 만들고
    easy_button = Button('easy',cx=w/2,cy=h/2/2,width=w/3)
    normal_button = Button('normal',cx=w/2,cy=h/2,width=w/3)
    hard_button = Button('hard',cx=w/2,cy=h/4*3,width=w/3)
    back_button = Button('<-')
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
        clock.tick(60)
        pygame.display.flip()

def no_board_to_continue():
    w,h = SCREEN.get_size()

    back_button = Button('<-',cx=w/2,cy=h*3/4,width=w/3,height=100)

    border = pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
    font = pygame.font.SysFont('malgungothic', 30)
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
    back_button = Button('<-')
    go_back = False
    if cont_game:
        load_infos = load_continue()
        if load_infos: 
            boards, player, difficulty = load_infos
            board = boards[-1]
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
    SCREEN.fill(WHITE)
    draw_table()
    draw_cursor(x,player)
    pygame.display.flip()
    while run:
        SCREEN.fill(WHITE)
        if is_win(board,2//player) != 0:
            print("for review")
            save_review(continue_boards,2//player, difficulty)
            end(board, is_win(board,2//player))
            return
        
        if block_event: block_event = False
        if player == 2:
            block_event = True
            col = test_main(board, player,difficulty)
            board, player, is_valid = get_next_state(board,col,player)
            if is_valid: continue_boards.append(copy.deepcopy(board))
            
        for event in pygame.event.get():
            if block_event: break
            if event.type == pygame.MOUSEBUTTONDOWN:
                clicked_x, clicked_y = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONUP:
                print(clicked_x)
                if not is_valid_x(clicked_x): continue
                x,_ = pygame.mouse.get_pos()
                col = x2col(x)
                board, player, is_valid = get_next_state(board,col,player)
                if is_valid: continue_boards.append(copy.deepcopy(board))
                # print(board)

            x,y = pygame.mouse.get_pos()
            
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                save_continue(continue_boards, player,difficulty)
                run = False
        
        go_back = back_button.draw_and_get_event(SCREEN, event)
        if go_back: 
            save_continue(continue_boards, player,difficulty)
            SCREEN.fill(WHITE)
            return
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    pos = cord2pos((i,j))
                    draw_circle_with_pos(pos, player=board[i][j])
        draw_table()
        draw_cursor(x,player)
        pygame.display.flip()

def end(board, player):
    save_continue([],None, None)
    w,h = SCREEN.get_size()
    if player == 1: text_content = "이겼습니다! 축하드립니다!"
    elif player == 2: text_content = "아쉽게도 졌네요 ㅠㅠ"
    else: text_content = "비겼습니다! 한번 더하면 이길지도...?"

    back_button = Button('<-',cx=w/2,cy=h*3/4,width=w/3,height=100)

    border = pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
    font = pygame.font.SysFont('malgungothic', 30)
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
        draw_table()
        

def how_to():
    pass

def no_board_to_review():
    w,h = SCREEN.get_size()

    back_button = Button('<-',cx=w/2,cy=h*3/4,width=w/3,height=100)

    border = pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
    font = pygame.font.SysFont('malgungothic', 30)
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
    back_button = Button('<-')
    previous_button = Button('<<',cx=w/4,cy=h*3/4,width=w/2,height=100)
    next_button = Button('>>',cx=w/4*3,cy=h*3/4,width=w/2,height=100)
    recommend_button = Button('만약 AI라면...',cx=w/2,cy=h*3/4+100,width=w/2,height=100,font='malgungothic')
    font = pygame.font.SysFont('malgungothic', 30)

    border = pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
    text_content = "{} / {}".format(idx, len(review_boards)-1)
    text = font.render(text_content, True, BLACK)
    text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    text_rect.center = border.center

    if (len(review_boards)+player)%2: fp = 1
    else: fp = 0
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
                player = (fp+1)%2+1
                print(review_boards[idx], player)
                col = test_main(review_boards[idx], player, 'hard')
                for r in range(5,-1,-1):
                    if review_boards[idx][r][col] == 0:
                        row = r
                        break
                pos = cord2pos((row,col))
                cord_recommend = pos
        
        if cord_recommend != (None, None):
            draw_circle_with_pos(cord_recommend,player=3)

        for i in range(len(review_boards[idx])):
            for j in range(len(review_boards[idx][0])):
                if review_boards[idx][i][j] != 0:
                    pos = cord2pos((i,j))
                    draw_circle_with_pos(pos, player=review_boards[idx][i][j])
        draw_table()
        text_content = "{} / {}".format(idx, len(review_boards)-1)
        text = font.render(text_content, True, BLACK)
        SCREEN.blit(text, text_rect)
        pygame.display.flip()


def option():
    pass
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
