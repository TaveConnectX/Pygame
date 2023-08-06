import os
import pickle
import pygame
import copy
from classes import *


# continue.pkl을 불러온다. 없다면 False 리턴 
def load_continue():
    if not os.path.isfile('files/continue.pkl'):
        return []
    else:
        with open('files/continue.pkl', 'rb') as file:
        # pickle.load() 함수를 사용하여 객체를 로드합니다.
            return pickle.load(file)

# 뒤로 가기 등을 눌렀을 때 continue_states 배열을 저장
def save_continue(continue_states, player, difficulty, remained_undo):
    if continue_states:
        save_infos = [continue_states,player, difficulty, remained_undo]
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

def load_setting():
    if not os.path.isfile('files/setting.pkl'):
        init_setting = {
            'first_player':3,
            'p1_color':(255,0,0),
            'p2_color':(255,255,0),
            'music':1,
            'effect':1
        }
        with open('files/setting.pkl', 'wb') as file:
            pickle.dump(init_setting, file)
        with open('files/default_setting.pkl', 'wb') as file:
            pickle.dump(init_setting, file)
        return init_setting
    else:
        with open('files/setting.pkl', 'rb') as file:
        # pickle.load() 함수를 사용하여 객체를 로드합니다.
            return pickle.load(file)

def save_setting(setting):
    with open('files/setting.pkl', 'wb') as file:
        # pickle.dump() 함수를 사용하여 객체를 저장합니다.
        pickle.dump(setting, file)



def draw_table(surface):
    w,h = surface.get_size()
    block_size = (w-100)/7
    for i in range(8):
        pygame.draw.line(surface, BLACK, (50+block_size*i,100), (50+block_size*i,100+6*block_size), width=5)
    for i in range(7):
        pygame.draw.line(surface, BLACK, (50,100+block_size*i), (50+block_size*7,100+block_size*i), width=5)





# board 상의 좌표를 SCREEN의 좌표로 변경
def coord2pos(surface, coord):
    y,x = coord
    # y = 5-y
    w,h = surface.get_size()
    r = (w-100)/7/2

    # (0,0) 일 때의 position
    pos = [50+r,100+r]
    pos[0] += 2*x*r
    pos[1] += 2*y*r

    return pos

def x2col(surface, x):
    w,h = surface.get_size()
    block_size = (w-100)/7
    x -= 50
    col = -1

    while x >= 0:
        x -= block_size
        col += 1

    if not 0<=col<=6:
        col = -1
    return col

def is_valid_x(surface, x, y, *buttons):
    w,h = surface.get_size()
    if x<50 or w-50<x:
        return False
    for button in buttons:
        cx, cy, w, h = button.cx, button.cy, button.width, button.height
        if cx-w/2<=x<=cx+w/2 and cy-h/2<=y<=cy+h/2:
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