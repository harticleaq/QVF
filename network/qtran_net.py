import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

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

    def forward(self, q_values, states):  # states的shape为(episode_num, max_episode_len， state_shape)

        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)  # (episode_num * max_episode_len, 1, n_agents) 
        states = states.reshape(-1, self.args.state_shape)  # (episode_num * max_episode_len, state_shape)

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



class QtranQAlt(nn.Module):
    def __init__(self, args):
        super(QtranQAlt, self).__init__()
        self.args = args

      
        self.action_encoding = nn.Sequential(nn.Linear(self.args.n_actions, self.args.n_actions),
                                             nn.ReLU(),
                                             nn.Linear(self.args.n_actions, self.args.n_actions))

  
        self.hidden_encoding = nn.Sequential(nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim))

       
        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim + self.args.n_agents
        self.q = nn.Sequential(nn.Linear(q_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.n_actions))


    def forward(self, state, hidden_states, actions):  # (episode_num, max_episode_len, n_agents, n_actions)

        episode_num, max_episode_len, n_agents, n_actions = actions.shape

      
        action_encoding = self.action_encoding(actions.reshape(-1, n_actions))
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, n_agents, n_actions)

       
        hidden_encoding = self.hidden_encoding(hidden_states.reshape(-1, self.args.rnn_hidden_dim))
        hidden_encoding = hidden_encoding.reshape(episode_num, max_episode_len, n_agents, self.args.rnn_hidden_dim)

      
        hidden_encoding = hidden_encoding.sum(dim=-2)  # (episode_num, max_episode_len, rnn_hidden_dim)
        hidden_encoding = hidden_encoding.unsqueeze(-2).expand(-1, -1, n_agents, -1)  # (episode_num, max_episode_len, n_agents， rnn_hidden_dim)

       
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, 1, n_agents * n_actions)
        action_encoding = action_encoding.repeat(1, 1, n_agents, 1)  
    
        action_mask = (1 - torch.eye(n_agents))  
        action_mask = action_mask.view(-1, 1).repeat(1, n_actions).view(n_agents, -1)
        if self.args.cuda:
            action_mask = action_mask.cuda()
        action_encoding = action_encoding * action_mask.unsqueeze(0).unsqueeze(0)

        action_encoding = action_encoding.reshape(episode_num, max_episode_len, n_agents, n_agents, n_actions)
        action_encoding = action_encoding.sum(dim=-2)  

        inputs = torch.cat([state, hidden_encoding, action_encoding], dim=-1)
        q = self.q(inputs)
        # q = -torch.exp(q)
        return q


# Joint action-value network
class QtranQBase(nn.Module):
    def __init__(self, args):
        super(QtranQBase, self).__init__()
        self.args = args
        self.b_init_value = 0.01
      
        ae_input = self.args.rnn_hidden_dim + self.args.n_actions
        self.hidden_action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                             nn.ReLU(),
                                             nn.Linear(ae_input, ae_input))

        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim
        self.q = nn.Sequential(nn.Linear(q_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, 1))
        
        # for fc in self.q:
        #     if isinstance(fc, torch.nn.modules.activation.ReLU):
        #         continue
        #     fanin_init(fc.weight)
        #     fc.bias.data.fill_(self.b_init_value)

    def forward(self, state, eval_hiddens, actions):  # (episode_num, max_episode_len, n_agents, n_actions)
        episode_num, max_episode_len, n_agents, _ = actions.shape
        hidden_actions = torch.cat([eval_hiddens, actions], dim=-1)
        hidden_actions = hidden_actions.reshape(-1, self.args.rnn_hidden_dim + self.args.n_actions)
        hidden_actions_encoding = self.hidden_action_encoding(hidden_actions)
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len, n_agents, -1)  
        hidden_actions_encoding = hidden_actions_encoding.sum(dim=-2)
        inputs = torch.cat([state.reshape(episode_num * max_episode_len, -1), hidden_actions_encoding], dim=-1).detach()
        q = self.q(inputs)
        return q


class QtranV(nn.Module):
    def __init__(self, args):
        super(QtranV, self).__init__()
        self.args = args
        self.b_init_value = 0.01

        v_input = self.args.state_shape
        self.v = nn.Sequential(nn.Linear(v_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, 1))
        
        for fc in self.v:
            if isinstance(fc, torch.nn.modules.activation.ReLU):
                continue
            fanin_init(fc.weight)
            fc.bias.data.fill_(self.b_init_value)

    def forward(self, state):
        v = self.v(state)
        return v


