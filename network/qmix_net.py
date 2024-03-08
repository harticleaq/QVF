import torch.nn as nn
import torch
import torch.nn.functional as F

'''
General hyper-net.
'''
class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.hyper_b2 =nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1)
                                     )

    def forward(self, q_values, states): 

        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)  
        states = states.reshape(-1, self.args.state_shape)  

        w1 = torch.abs(self.hyper_w1(states)) 
        b1 = self.hyper_b1(states)  

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  

        hidden = F.elu(torch.bmm(q_values, w1) + b1) 

        w2 = torch.abs(self.hyper_w2(states))  
        b2 = self.hyper_b2(states)  

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  
        b2 = b2.view(-1, 1, 1)  

        q_total = torch.bmm(hidden, w2) + b2  
        q_total = q_total.view(episode_num, -1, 1) 
        return q_total


'''
Intent attention module, consider action intention of other agents.
'''
class QMixNet2(nn.Module):
    def __init__(self, args):
        super(QMixNet2, self).__init__()
        self.args = args
        
        self.use_imn = False
        self.num_heads = 2
        self.imn_dim = self.args.n_agents * self.args.n_actions
        self.query_dim = 64
        self.value_dim = 64
        self.key_hidden_dim = 64

        self.query = nn.Sequential(nn.Linear(self.args.state_shape, self.query_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.query_dim, self.query_dim))
        self.key_extractors = nn.ModuleList()
        for i in range(self.num_heads):
            self.key_extractors.append(nn.Sequential(nn.Linear(self.imn_dim, self.key_hidden_dim)))

        self.key = nn.Sequential(nn.Linear(self.imn_dim, self.key_hidden_dim),
                                 nn.ReLU(),
                                nn.Linear(self.key_hidden_dim, self.key_hidden_dim))

        self.value = nn.Sequential(nn.Linear(self.imn_dim, self.value_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.value_dim, self.value_dim))

        if self.use_imn:
            input_dim = args.state_shape + self.value_dim + args.n_agents
        else:
            input_dim = args.state_shape + (args.n_agents) * args.n_actions + args.n_agents
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(input_dim, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(input_dim, args.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.args.state_shape, args.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(self.args.state_shape, args.qmix_hidden_dim)
        self.hyper_b2 =nn.Sequential(nn.Linear(self.args.state_shape, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1)
                                     )

    def forward(self, q_values, states, other_actions, agent_id):  
        episode_num, episode_len = q_values.size(0), q_values.size(1)
        q_values = q_values.view(-1, 1, self.args.n_agents)  
        b_states = states.reshape(-1, self.args.state_shape)  
        states = states.reshape(-1, self.args.state_shape).repeat(1, self.args.n_agents).view(-1, self.args.n_agents, self.args.state_shape)
        other_actions = other_actions.reshape(-1, self.args.n_agents, self.args.n_agents * self.args.n_actions)

        if self.use_imn:
            q = self.query(states).reshape(episode_num* episode_len, self.args.n_agents, self.num_heads, -1).permute(0, 2, 1, 3)
            k = self.key(other_actions).reshape(episode_num* episode_len, self.args.n_agents, self.num_heads, -1).permute(0, 2, 1, 3)
            v = self.value(other_actions).reshape(episode_num* episode_len, self.args.n_agents, self.num_heads, -1).permute(0, 2, 1, 3)
            attn = q @ k.transpose(2, 3) * (other_actions.shape[-1] ** -0.5)
            attn = attn.softmax(dim=-1)
            other_actions = (attn @ v).permute(0, 2, 1, 3).reshape(episode_num* episode_len, self.args.n_agents, -1)

        agent_id = agent_id.reshape(-1, self.args.n_agents, self.args.n_agents)
        states = torch.cat((states, other_actions, agent_id), -1)

        w1 = torch.abs(self.hyper_w1(states))  
        b1 = self.hyper_b1(b_states)  

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim) 
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim) 

        hidden = F.elu(torch.bmm(q_values, w1) + b1)  

        w2 = torch.abs(self.hyper_w2(b_states))  
        b2 = self.hyper_b2(b_states)  

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  
        b2 = b2.view(-1, 1, 1)  
        q_total = torch.bmm(hidden, w2) + b2  
        q_total = q_total.view(episode_num, -1, 1) 
        return q_total