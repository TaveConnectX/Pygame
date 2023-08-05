import copy
import sys
import os
import numpy as np
import onnxruntime
from env import ConnectFour, MCTS, MCTS_alphago, get_encoded_state

# player 의 이름이 적절하지 않으면 error를 출력
class playerError(Exception):
    def __str__(self):
        return "impossible player"

# 보드판을 보고 가능한 action을 확인 (0~6)
def get_valid_actions(state):
    valid_actions = []
    for col in range(len(state[0])):
        if state[0][col]==0: 
            valid_actions.append(col)

    return valid_actions

def board_normalization(state, player):
    arr = np.array(state, dtype=float)
    arr[arr == 2] = -1
    
    # 2p이면 보드판을 반전시켜서 보이게 하여, 항상 같은 색깔을 보면서 학습 가능
    if player == 2: arr = -1 * arr
    elif player == 1: pass
    else: 
        raise playerError

    return arr


def load_model(difficulty):
    path = 'files/model/'+difficulty+'/'
    file_names = os.listdir(path)

    if difficulty=='normal':
        model = onnxruntime.InferenceSession(path+"normal_model.onnx")
        value_model = onnxruntime.InferenceSession(path+"value_model.onnx")

        return (model, value_model)
    
    else:
        
        try:    
            model = onnxruntime.InferenceSession(path+"{}_model.onnx".format(difficulty))
            
        except Exception as e:
            print(f'모델 로드에서 예외가 발생했습니다: {e}')

        return model


def get_minimax_action(model, state, valid_actions):

    q_values = model.run(None, {model.get_inputs()[0].name: np.expand_dims(get_encoded_state(state), axis=0)})[0][0]
    
    # print(q_values.reshape(7,7))
    q_dict = {}
    # print(valid_actions)
    # print(distinct_actions)
    for a in valid_actions:
        q_dict[a] = []
        for b in valid_actions:
            idx = 7*a + b
            # print(a,b)
            # print(q_value[idx])
            # print(q_dict[a][1])
            
            q_dict[a].append((b, -q_values[idx]))

        # q_dict[a] = torch.tensor(q_dict[a])
        q_dict[a] = np.array(q_dict[a])
        
        op_action, value = q_dict[a][q_dict[a].argmax(axis=0)[1]]
        # op_action, value = softmax_policy(torch.tensor(q_dict[a]), temp=temp)
        # if torch.isnan(value):
        #     print(a,b)
        #     print(q_value.reshape(7,7))
        #     print(q_dict)
        q_dict[a] = (op_action, -1 * value)

    qs_my_turn = [[key, value[1]] for key, value in q_dict.items()]
    # qs_my_turn = torch.tensor(qs_my_turn)
    qs_my_turn = np.array(qs_my_turn)
    action, value = qs_my_turn[qs_my_turn.argmax(axis=0)[1]]
        
    action = int(action)
    

    return action


def get_alphago_action(model, value_model, state, vas):
    args = {
            'C': 1.5,
            'num_searches': 200,
            'dirichlet_epsilon': 0.,
            'dirichlet_alpha': 0.3
        }

    CF = ConnectFour()
    mcts = MCTS_alphago(CF, args, model, value_model)
    mcts_probs = mcts.search(state)
    # action = np.random.choice(range(7),p=mcts_probs)
    action = np.argmax(mcts_probs)
    # print(mcts_probs)
    return action




def get_alphazero_action(model, state, vas):
    args = {
        'C': 1.5,
        'num_searches': 100,
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3
    }
    CF = ConnectFour()
    mcts = MCTS(CF, args, model)
    mcts_probs = mcts.search(state)
    # action = np.random.choice(range(7),p=mcts_probs)
    action = np.argmax(mcts_probs)
    print(mcts_probs)
    
    return action

def get_action(model, state, difficulty, vas):
    if isinstance(model, tuple):
        model, value_model = model


    if difficulty=='easy':
        return get_minimax_action(model, state, vas)
    elif difficulty=='normal':
        return get_alphago_action(model, value_model, state, vas)
    elif difficulty=='hard':
        return get_alphazero_action(model, state, vas)
    else: 
        print('error')
        exit()

def test_main(state, player, difficulty):
    # model type 확인

    
    state = np.array(state)  # list to numpy array

    # 1p, 2p 확인
    # player를 인자로 받도록 변경 
    # player = check_player(state)
    print(state)
    # env가 없으므로 valid action이 뭔지 따로 확인
    valid_actions = get_valid_actions(state)
    # print("valid_actions:",valid_actions)

    state = board_normalization(state,  player)
    
    # alphago는 따로 적용 
    
    # 알맞은 model 할당
    model = load_model(difficulty)
    # print(model)
    # 가중치 load
    # 모델에 forward
    action = get_action(model, state, difficulty, valid_actions)


    

    # for debugging
    # print("model name:", model_name)
    # print("model type:", model_type)
    # print("player:", player)
    # print("Q values:", qvalues.tolist())
    # print("valid actions:", valid_actions)
    # print("maxQ:", torch.max(valid_q_values).item())
    # print("selected action:", valid_actions[torch.argmax(valid_q_values)])

    # 가장 높은 value를 가진 action return
    return action