import torch
import torch.nn as nn
# from torch.distributions import Normal

class Layer(nn.Module):
    def __init__(self,in_channels,out_channels, activation_function="elu"):
        super(Layer,self).__init__()
        self.activation_functions = {
            "elu" : nn.ELU(),
            "relu" : nn.ReLU(inplace=True),
            "leakyrelu" :nn.LeakyReLU(),
            "sigmoid" : nn.Sigmoid(),
            "tanh" : nn.Tanh(),
            "relu6" : nn.ReLU6()
           } 
        self.layer = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            self.activation_functions[activation_function]
        )
    def forward(self,x):
        return self.layer(x)

class Encoder(nn.Module):
    def __init__(
            self, info, cfg):
        super(Encoder,self).__init__()
        encoder_features = cfg["encoder_features"]          # [1500,1000]
        # print(f"encoder_features = {encoder_features}")
        activation_function = cfg["activation_function"]    # [leakyrelu]
        
        self.encoder_layers = nn.ModuleList() 
        in_channels = info                         # encoder가 sparse면 in_channels는 441, encoder가 dense면 in_channels는 676
        for feature in encoder_features:
            self.encoder_layers.append(nn.Linear(in_channels, feature))
            self.encoder_layers.append(nn.LeakyReLU(inplace=True))
            in_channels = feature

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
class ConvHeightmapEncoder(nn.Module):
    def __init__(self, in_channels, encoder_features=[16, 32]):
        # print("in_channels = ", in_channels)                        # 10201
        # print("encoder_features = ", encoder_features)              # 8, 16, 32, 64
        # print("encoder_activation = ", encoder_activation)          # leaky_relu
        super().__init__()
        # self.heightmap_size는 rover_env_cfg.py에서 height_scanner의 size의 제곱근과 같음. Ex) resolution=0.05, size=[5.0, 5.0] 이라고 하면, 한 변이 101개이므로, self.heightmap_size는 101이 나옴.
        self.heightmap_size = torch.sqrt(torch.tensor(in_channels)).int()   # tensor(101, dtype=torch.int32)
        
        # print("self.heightmap_size = ",self.heightmap_size)         # tensor(101, dtype=torch.int32)
        # kernel = 가중치 필터
        kernel_size = 3
        
        # kernel이 움직이는 칸 수
        # 무조건 웬만하면 1로 하자.
        stride = 1
        
        # padding : 배열의 둘레를 확장하고 0으로 채우는 연산. Ex) 3,3일 경우 5,5로 되며 둘레가 다 0으로 채워짐.
        padding = 1
        
        self.encoder_layers = nn.ModuleList()
        in_channels = 1  # 1 channel for heightmap
        
        """
        kernel_size : 합성곱 필터 크기
        stride : 필터가 움직이는 간격
        padding : 배열의 둘레를 확장하기 위한 값
        
        nn.Conv2d : 입력 채널(in_channels)에서 출력 채널(feature)로의 합성곱 연산을 수행하는 nn.Conv2d 레이어를 추가.
        nn.BatchNorm2d : 배치 정규화를 수행하여 학습 속도를 높이고 안정성을 향상
        get_activation : 활성화 함수 추가
        nn.MaxPool2d : 최대 풀링(Max Pooling)을 수행하여 입력 데이터를 압축하고 주요 정보를 강조
        
        """
        for feature in encoder_features:
            # print("\n\n12342352358932y589027589")
            # print("feature = ", feature)    # feature가 불러와질때마다 8, 16, 32, 64가 출력됨.
            self.encoder_layers.append(nn.Conv2d(in_channels, feature, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=False))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            # self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            self.encoder_layers.append(nn.LeakyReLU(inplace=True))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            self.encoder_layers.append(nn.Conv2d(feature, feature, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=False))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            self.encoder_layers.append(nn.LeakyReLU(inplace=True))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            # Pooling = 행렬을 압축해, 특정 데이터를 강조하는 역할을 수행!
            self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            # print("encoder_layers 출력 : \n",self.encoder_layers)
            in_channels = feature
            # print("in_channels = ", in_channels)    # feature가 불러와질때마다 8, 16, 32, 64가 출력됨.
            # print("\n\n")
        out_channels = in_channels
        # print("out_channels = ", out_channels)      # 마지막으로 in_cprint("flatten_size : ", flatten_size)
        
        
        
        """_summary_
        목적 : CNN 레이어를 통과한 후 데이터의 너비(w)와 높이(h)를 계산하기 위함
        방법 : Kernel, stride, padding을 전부 고려해서 계산
        """
        flatten_size = [self.heightmap_size, self.heightmap_size]
        for _ in encoder_features:
            # Conv2D 레이어를 거치면 아래와 같이 너비와 높이가 변함.
            w = (flatten_size[0] - kernel_size + 2 * padding) // stride + 1
            # print("w = ", w)
            h = (flatten_size[1] - kernel_size + 2 * padding) // stride + 1
            # print("h = ", h)
            
            # Conv2D 레이어를 거치면 아래와 같이 너비와 높이가 변함.
            w = (w - kernel_size + 2 * padding) // stride + 1
            # print("w = ", w)
            h = (h - kernel_size + 2 * padding) // stride + 1
            # print("h = ", h)
            
            # Max Pooling을 거치면 아래와 같이 너비와 높이가 변함!
            w = (w - 2) // 2 + 1
            h = (h - 2) // 2 + 1
            flatten_size = [w, h]   # flatten_size :  [tensor(6, dtype=torch.int32), tensor(6, dtype=torch.int32)]
            
        self.conv_out_features = out_channels * flatten_size[0] * flatten_size[1]   # 64*6*6=tensor(2304, dtype=torch.int32)

        features = [80, 60]

        self.mlps = nn.ModuleList()
        in_channels = self.conv_out_features    # in_channels =  tensor(2304, dtype=torch.int32)
        # print("isaac_rover/rover_envs/envs/navigation/learning/skrl/models.py에서 실행중\n"*10)
        # print(f"in_channels = {in_channels}")
        
        for feature in features:
            # print(f"")
            self.mlps.append(nn.Linear(in_channels, feature))
            self.mlps.append(nn.LeakyReLU(inplace=True))
            in_channels = feature
        # Mlp : 2304 -> 80 -> 60

        self.out_features = features[-1]

    def forward(self, x):
        # x is a flattened heightmap, reshape it to 2D
        # view함수는 텐서의 shape을 변경하는 함수임.
        # 처음에 -1은 자동으로 차원을 지정하라는 의미. 즉, 뒤의 값인 1에 맞게 알아서 shape이 변경됨.
        # print("isaac_rover/rover_envs/envs/navigation/learning/skrl/models.py\n" * 20)
        # print("self.heightmap_size : ", self.heightmap_size)
        # print("x.shape : ", x.shape)  # 현재 x의 크기 확인
        # print("변경 전!")
        # print("x.shape = ",x.shape)
        x = x.view(-1, 1, self.heightmap_size, self.heightmap_size)
        # print(f"heightmap_size = {self.heightmap_size}")
        # print("%^&*(^*%*&%*&%^*(%&*(%*&%&*(%*&(%&*(%&*(())))))))")
        # print("x 출력중")
        # print(x)
        # print("변경 후!")
        # print("x.shape = ",x.shape)

        for layer in self.encoder_layers:
            x = layer(x)
            # print("x = layer(x) 결과 출력")
            # print(x.shape)

        x = x.view(-1, self.conv_out_features)
        # print("%^&*(^*%*&%*&%^*(%&*(%*&%&*(%*&(%&*(%&*(())))))))")
        # print("x 출력중")
        # print(x)
        for layer in self.mlps:
            x = layer(x)
            # print("x = layer(x) 결과 출력")
            # print(x)
        return x

class Belief_Encoder(nn.Module):
    def __init__(
            self, info, cfg, input_dim=120):
        super(Belief_Encoder,self).__init__()
        self.hidden_dim = cfg["hidden_dim"]                     # 300
        self.n_layers = cfg["n_layers"]                         # 2
        activation_function = cfg["activation_function"]        # leakyrelu
        proprioceptive = info["proprioceptive"]                 # 4
        input_dim = proprioceptive+input_dim                    # 4+40=44
        
        self.gru = nn.GRU(input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.gb = nn.ModuleList()
        self.ga = nn.ModuleList()
        gb_features = cfg["gb_features"]                        # [128,256,512,1024]
        ga_features = cfg["ga_features"]                        # [128,256,512,1024]

        in_channels = self.hidden_dim                           # 300
        for feature in gb_features:                             # [128,128,120]
            self.gb.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        in_channels = self.hidden_dim                           # 300
        for feature in ga_features:                             # [128,128,120]
            self.ga.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.ga.append(nn.Sigmoid())

    def forward(self, p, l_e, h):
        # p = proprioceptive
        # e = exteroceptive
        # h = hidden state
        # x = input data, h = hidden state
        
        x = torch.cat((p,l_e),dim=2)
        out, h = self.gru(x, h)
        x_b = x_a = out
        
        for layer in self.gb:
            x_b = layer(x_b)
        for layer in self.ga:
            x_a = layer(x_a)
        # print(f"l_e.shape = {l_e.shape}")
        # print(f"x_a.shape = {x_a.shape}")
        x_a = l_e * x_a
        # TODO IMPLEMENT GATE
        belief = x_b + x_a

        return belief, h, out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to('cpu')
        return hidden

class Belief_Decoder(nn.Module):
    def __init__(
            self, info, cfg, n_input=50, hidden_dim=50,n_layers=2,activation_function="leakyrelu"):
        super(Belief_Decoder,self).__init__()
        exteroceptive = info["sparse"] + info["dense"]
        gate_features = cfg["gate_features"] #[128,256,512, exteroceptive]
        decoder_features = cfg["decoder_features"]#[128,256,512, exteroceptive]
        #n_input = cfg[""]
        gate_features.append(exteroceptive)
        decoder_features.append(exteroceptive)
        self.n_input = n_input
        self.gate_encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
    

        in_channels = self.n_input
        for feature in gate_features:
            self.gate_encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        self.gate_encoder.append(nn.Sigmoid())  

        in_channels = self.n_input
        for feature in decoder_features:
            self.decoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        

    def forward(self, e, h):
        gate = h[-1]
        decoded = h[-1]
       # gate = gate.repeat(e.shape[1], 1, 1).permute(1,0,2)
       # decoded = decoded.repeat(e.shape[1], 1, 1).permute(1,0,2)
        for layer in self.gate_encoder:
            gate = layer(gate)

        for layer in self.decoder:
            decoded = layer(decoded)
        x = e*gate
        x = x + decoded
        return x
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(1.0)

class MLP(nn.Module):
    def __init__(
            self, info, cfg, belief_dim):
        super(MLP,self).__init__()
        self.mlp = nn.ModuleList()  # MLP for student policy
        proprioceptive = info["proprioceptive"]             # 4
        action_space = info["actions"]                      # 2
        activation_function = cfg["activation_function"]    # leakyrelu
        network_features = cfg["network_features"]          # [256,160,128]

        in_channels = proprioceptive + belief_dim           # 124
        for feature in network_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(nn.LeakyReLU(inplace=True))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels,action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def forward(self, p, belief):
        # print("hejhej")
        # print(p.shape)
        # print(belief.shape)
        # p.shape = torch.Size([8, 1, 4])
        # belief.shape = torch.Size([8, 1, 40])
        x = torch.cat((p,belief),dim=2)
        # print(x.shape)
        for layer in self.mlp:
            x = layer(x)
        return x, self.log_std_parameter

class Student(nn.Module):
    def __init__(
            self, info, cfg):
        super(Student,self).__init__()

        self.n_re = info["reset"]           # 1
        self.n_pr = info["proprioceptive"]  # 4
        self.n_sp = info["sparse"]          # 441
        self.n_de = info["dense"]           # 676
        self.n_ac = info["actions"]         # 2
        encoder_layers = cfg["encoder"]
        
        self.sparse_encoder = Encoder(self.n_sp, encoder_layers)
        self.dense_encoder = Encoder(self.n_de, encoder_layers)
        encoder_dim = cfg["encoder"]["encoder_features"][-1] * 2        # encoder_dim = 2000
        self.belief_encoder = Belief_Encoder(info, cfg["belief_encoder"], input_dim=encoder_dim)
        self.belief_decoder = Belief_Decoder(info, cfg["belief_decoder"], cfg["belief_encoder"]["hidden_dim"])
        
        # student policy의 최종 MLP
        self.MLP = MLP(info, cfg["mlp"], belief_dim=encoder_dim)
        # print(f"self.MLP = {self.MLP}")
        # print(f"type 출력중!\n{type(self.MLP)}")
        # print("=====================\n"*5)
        
        # print(f"mlp_params = {mlp_params}")
        # print(f"encoder_params1 = {encoder_params1}")
        # print(f"\n")
        # print(f"encoder_params2 = {encoder_params2}")
        # print()
        # print(f"mlp_params key 출력중!")
        # for k, v in mlp_params.items():
        #     print(k)
        
        # print(f"sparse_encoder_params 출력중!")
        # for k, v in sparse_encoder_params.items():
        #     print(k)
            # print(v)
            
        # print(f"self.sparse_encoder 출력중!")
        # print(self.sparse_encoder)
        # print(encoder_params1.keys())
        # print(encoder_params2.keys())
        
        # print(f"self.MLP 출력중!\n{self.MLP}")
        # print(f"self.sparse_encoder 출력중!\n{self.sparse_encoder}")
        # print(f"self.dense_encoder 출력중!\n{self.dense_encoder}")
        # print("Before .to(cuda):", next(self.MLP.parameters()).device)
        # print("Before .to(cuda):", next(self.sparse_encoder.parameters()).device)
        # print("Before .to(cuda):", next(self.dense_encoder.parameters()).device)
        self.MLP.to("cuda")
        self.sparse_encoder.to("cuda")
        self.dense_encoder.to("cuda")

    def forward(self, x, h):
        # print("model에 input 들어왔다잇!")
        n_re = self.n_re    # 1
        n_ac = self.n_ac    # 2
        n_pr = self.n_pr    # 4
        n_sp = self.n_sp    # 441
        n_de = self.n_de    # 676
        # print("num of state")
        # print(n_re, n_ac, n_pr, n_sp, n_de)
        # reset = x[:,:, 0:n_re]
        # print(f"reset = {reset}")
        # actions = x[:,:,n_re:n_re+n_ac]
        # print(f"actions = {actions}")
        proprioceptive = x[:,:,n_re:n_re+n_pr]
        # print(f"proprioceptive = {proprioceptive[0,0,:]}")
        sparse = x[:,:,-(n_sp+n_de):-n_de]
        dense = x[:,:,-n_de:]
        exteroceptive = torch.cat((sparse,dense),dim=2)

        #sparse_gt = gt[:,:,-(n_de+n_sp):-n_de]
        #dense_gt = gt[:,:,-n_de:]
        # n_p = self.n_p
        
        # p = x[:,:,0:n_p]        # Extract proprioceptive information  
        
        # e = x[:,:,n_p:1084]         # Extract exteroceptive information
        
        # Pass exteroceptive information through encoder
        e_l1 = self.sparse_encoder(sparse)  # shape = [batch_size, 1, encoder_output]
        e_l2 = self.dense_encoder(dense)    # shape = [batch_size, 1, encoder_output]
        
        # print(f"e_l1.shape = {e_l1.shape}")
        # print(f"e_l2.shape = {e_l2.shape}")
        #################
        # 원본은 dim=2
        e_l = torch.cat((e_l1,e_l2), dim=2)

        #e_l1_gt = self.encoder1(sparse_gt) # Pass exteroceptive information through encoder
        
       # e_l2_gt = self.encoder2(dense_gt)

       # gt_ex = torch.cat((e_l1_gt,e_l2_gt), dim=2)
        belief, h, out = self.belief_encoder(proprioceptive,e_l,h) # extract belief state
        
        #estimated = self.belief_decoder(exteroceptive,h)
        estimated = self.belief_decoder(exteroceptive,out)
        
        
        actions, log_std = self.MLP(proprioceptive,belief)

        # min_log_std= -20.0
        # max_log_std = 2.0
        # log_std = torch.clamp(log_std, 
        #                         min_log_std,
        #                         max_log_std)
        # g_log_std = log_std
        # # print(actions.shape[0])
        # # print(actions.shape[2])
        # _g_num_samples = actions.shape[0]

        # # # distribution
        # _g_distribution = Normal(actions, log_std.exp())
        # #print(_g_distribution.shape)
        # # # sample using the reparameterization trick
        # actions = _g_distribution.rsample()
        #print((actions-action).mean())
        return actions, estimated, h#, gt_ex, belief


           # hidden_dim=50,n_layers=2,activation_function="leakyrelu"