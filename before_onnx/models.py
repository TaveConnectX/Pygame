import numpy as np
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

device = 'cpu'

# class 가독성을 놎이기 위해 network 부분만 따로 분리
class CFLinear(nn.Module):
    def __init__(self, state_size=6*7, action_size=7,num_layer=13, hidden_size=128):
        super(CFLinear,self).__init__()
        self.model_type = 'Linear'
        self.model_name = 'Linear-v1'
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, hidden_size))
        for _ in range(num_layer-3):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, 32))
        self.layers.append(nn.Linear(32, action_size))

        for layer in self.layers:
            if type(layer) in [nn.Conv2d, nn.Linear]:
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            layer = layer.to(device)
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        y = F.tanh(self.layers[-1](x))
        return y

# class 가독성을 높이기 위해 network 부분만 따로 분리 
class CFCNN(nn.Module):
    def __init__(self, input_channel=3,action_size=7,num_layer=13, hidden_size=128):
        super(CFCNN,self).__init__()
        self.model_type = 'CNN'
        self.model_name = 'CNN-v1'
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2,2), stride=1,padding=1)
        # self.conv2 = nn.Conv2d(32,64,(2,2), stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64,64,(2,2), stride=1, padding=1)
        
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4,4), stride=1,padding=2)
        # self.conv2 = nn.Conv2d(32,64,(4,4), stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64,64,(4,4), stride=1, padding=1)
        # self.linear1 = nn.Linear(64*3*4, 64)
        # self.linear2 = nn.Linear(64, action_size)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(input_channel, hidden_size, kernel_size=3, padding=1))
        for _ in range(num_layer-3):
            self.layers.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1))
        self.layers.append(nn.Conv2d(hidden_size, 32, kernel_size=3, padding=1))
        self.layers.append(nn.Linear(32*6*7, action_size))
        
        

        # relu activation 함수를 사용하므로, He 가중치 사용
        for layer in self.layers:
            if type(layer) in [nn.Conv2d, nn.Linear]:
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            layer = layer.to(device)


    def forward(self, x):
        # print("1st shape:",x.shape)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            # (N, 32, 6, 7)
        # print("after for:", x.shape)
        y = x.flatten(start_dim=2)  # (N, 32, 42)
        # print("after first flatten:", y.shape)
        y = y.flatten(start_dim=1)  # (N, 32*42)
        # print("after 2nd flatten:", y.shape)
        # print()
        y = F.tanh(self.layers[-1](y))   # (N, 7)
        # view로 채널 차원을 마지막으로 빼줌
        # 정확한 이유는 나중에 알아봐야 할듯? 
        # y = y.view(y.shape[0], -1, 42)  # (N, 12, 42)
        # y = y.flatten(start_dim=1)  # (N, 12*42)
        # y = F.relu(self.linear1(y))
        # y = self.linear2(y)
        return y

    # def forward(self,x):
    #     # (N, 1, 6,7)
    #     y = F.relu(self.conv1(x))
    #     # (N, 32, 7,8)
    #     y = F.relu(self.conv2(y))
    #     # (N, 64, 8,9)
    #     y = F.relu(self.conv3(y))
    #     # (N, 64, 9,10)
    #     #print("shape x after conv:",y.shape)
    #     y = y.flatten(start_dim=2)
    #     # (N, 64, 90)
    #     #print("shape x after flatten:",y.shape)
    #     y = y.view(y.shape[0], -1, 64)
    #     # (N, 90, 64)
    #     #print("shape x after view:",y.shape)
    #     y = y.flatten(start_dim=1)
    #     # (N, 90*64)
    #     y = F.relu(self.linear1(y))
    #     # (N, 64)
    #     y = self.linear2(y) # size N, 12
    #     # (N, 12)
    #     return y.cuda()


# class 가독성을 높이기 위해 network 부분만 따로 분리 
# class CNNforMinimax(nn.Module):
#     def __init__(self, action_size=7*7):
#         super(CNNforMinimax,self).__init__()
#         self.model_type = 'CNN'
#         self.model_name = 'CNN-Minimax-v1'

#         self.conv1 = nn.Conv2d(1,42,(4,4), stride=1, padding=2)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
#         self.linear1 = nn.Linear(42*3*4, 42)
#         self.linear2 = nn.Linear(42, action_size)
        
#         self.layers = [
#             self.conv1,
#             self.maxpool1,
#             self.linear1,
#             self.linear2
#         ]

#         # relu activation 함수를 사용하므로, He 가중치 사용
#         for layer in self.layers:
#             if type(layer) in [nn.Conv2d, nn.Linear]:
#                 init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
#             # layer = layer.to(device)

#         self.to(device)

#     def forward(self, x):
#         y = F.relu(self.conv1(x))  # (N, 42, 6, 7)
#         y = self.maxpool1(y)  # (N, 42, 3, 4)
#         y = y.flatten(start_dim=2)  # (N, 42, 12)
#         # view로 채널 차원을 마지막으로 빼줌
#         # 정확한 이유는 나중에 알아봐야 할듯? 
#         y = y.view(y.shape[0], -1, 42)  # (N, 12, 42)
#         y = y.flatten(start_dim=1)  # (N, 12*42)
#         y = F.relu(self.linear1(y))
#         y = self.linear2(y)
#         return y

  

# heuristic model을 이용하기 위한 껍데기
# action을 선택할 때 2차원 배열을 그대로 이용하므로 'cnn'으로 둠
class HeuristicModel():
    def __init__(self):
        self.model_type = 'CNN'
        self.model_name = 'Heuristic-v1'

# random model을 이용하기 위한 껍데기
# action을 선택할 때 2차원 배열을 그대로 이용하므로 'cnn'으로 둠
class RandomModel():
    def __init__(self):
        self.model_type = 'CNN'
        self.model_name = 'Random'

class MinimaxModel():
    def __init__(self):
        self.model_type = 'CNN'
        self.model_name = 'Minimax-tree'

class ResNetforDQN(nn.Module):
    def __init__(self, input_channel=3, num_blocks=5, num_hidden=128, action_size=7):
        super().__init__()
        self.device = 'cpu'
        self.model_type = 'CNN'
        self.model_name = 'DQN-ResNet-v1'
        self.action_size = action_size
        self.start_block = nn.Sequential(
            nn.Conv2d(input_channel, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backbone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_blocks)]
        )

        self.policy = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 7, action_size),
            nn.Tanh()
            
        )

        self.to(device)

        


    def forward(self, x):
        x = self.start_block(x)

        for res_block in self.backbone:
            x = res_block(x)
        
        q = self.policy(x)

        return q

    def predict(self, x):
        x = torch.FloatTensor(x.astype(np.float32)).to(self.device)
        while x.ndim<=3:
            x = x.unsqueeze(0)
        # x = x.view(1, self.size)
        self.eval()
        with torch.no_grad():
            q = self.forward(x)

        return q.data.cpu().numpy()[0]

class AlphaZeroResNet(nn.Module):
    def __init__(self, input_channel=3, num_blocks=5, num_hidden=128):
        super().__init__()
        self.device = 'cpu'
        self.model_name = 'AlphaZero-ResNet-v1'
        self.start_block = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backbone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_blocks)]
        )

        self.policy = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 7, 7)
        )

        self.value = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 6 * 7, 1),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.start_block(x)

        for res_block in self.backbone:
            x = res_block(x)
        
        p = F.softmax(self.policy(x), dim=1)
        v = self.value(x)

        return p, v

    def predict(self, x):
        x = torch.FloatTensor(x.astype(np.float32)).to(self.device)
        while x.ndim<=3:
            x = x.unsqueeze(0)
        # x = x.view(1, self.size)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(x)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden)
        )

    def forward(self, x):
        return F.relu(self.block(x) + x)
    

class DQNModel:
    def __init__(self, use_conv=True, use_resnet=True, use_minimax=True, command=None):
        if command == None:
            self.use_conv = use_conv
            self.use_resnet = use_resnet
            self.use_minimax = use_minimax
            self.command = bin((use_conv<<2) + (use_resnet<<1) + (use_minimax))[2:]
            while len(self.command) < 3:
                self.command = '0' + self.command
        else: 
            self.command = str(command)
            if len(self.command) != 3:
                print("command length error")
                exit()

        if self.command == '111':
            self.model = ResNetforDQN(action_size=49)
            self.model.model_name = 'DQN-resnet-minimax-v1'
        elif self.command == '110':
            self.model = ResNetforDQN(action_size=7)
            self.model.model_name = 'DQN-resnet-v1'
        elif self.command == '101':
            self.model = CFCNN(action_size=49)
            self.model.model_name = 'DQN-CNN-minimax-v1'
        elif self.command == '100':
            self.model = CFCNN(action_size=7)
            self.model.model_name = 'DQN-CNN-v1'
        # linear한 상태로는 resnet을 사용할 수 없음
        elif self.command in ['011', '010']:
            print("impossible command")
            exit()
        elif self.command == '001':
            self.model = CFLinear(action_size=49)
            self.model.model_name = 'DQN-linear-minimax-v1'
        elif self.command == '000':
            self.model = CFLinear(action_size=7)
            self.model.model_name = 'DQN-linear-v1'
        else:
            print("impossible command")
            exit()




class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(42, 84)  # 입력 크기: 42, 출력 크기: 임의로 설정한 중간 층 크기
        self.fc2 = nn.Linear(84, 3)  # 입력 크기: 중간 층 크기, 출력 크기: 클래스 수
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # 2차원 배열을 1차원으로 평탄화
        x = x.flatten()
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x