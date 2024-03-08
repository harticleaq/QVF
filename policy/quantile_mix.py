import torch
import os
from network.base_net import RNN
from network.qtran_net import QtranV, QtranQBase
from network.qmix_net import QMixNet, QMixNet2

class QuantileMix:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        rnn_input_shape = self.obs_shape

        if args.last_action:
            rnn_input_shape += self.n_actions 
        if args.reuse_network:
            rnn_input_shape += self.n_agents

        self.eval_rnn = RNN(rnn_input_shape, args)  
        self.target_rnn = RNN(rnn_input_shape, args)

        self.eval_joint_q = QtranQBase(args)  # Joint action-value network
        self.target_joint_q = QtranQBase(args)

        self.V = QtranV(self.args).to(self.args.device)

        if self.args.qmix_type == "mix2":
            self.eval_qmix_net = QMixNet2(args) 
            self.target_qmix_net = QMixNet2(args)
        else:
            self.eval_qmix_net = QMixNet(args) 
            self.target_qmix_net = QMixNet(args)

        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_joint_q.cuda()
            self.target_joint_q.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.env_args["map_name"]

        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_joint_q = self.model_dir + '/joint_q_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_joint_q.load_state_dict(torch.load(path_joint_q, map_location=map_location))
                print('Successfully load the model: {}, {}'.format(path_rnn, path_joint_q))
            else:
                raise Exception("No model!")


        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_joint_q.load_state_dict(self.eval_joint_q.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_rnn.parameters()) + list(self.eval_qmix_net.parameters()) + list(self.V.parameters())
        
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
            self.joint_optimizer = torch.optim.RMSprop(list(self.eval_joint_q.parameters()), lr=args.lr)
        else:
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)
            self.joint_optimizer = torch.optim.Adam(list(self.eval_joint_q.parameters()), lr=args.lr)

        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QuantileMix')
        self.hb_loss = torch.nn.HuberLoss(reduction='none')
        self.beta_mse = torch.nn.MSELoss()

    def quantile_regression(self, y_true, y_pred, alpha=0.5):
        diff = y_true - y_pred
        # hb = self.hb_loss(y_pred, y_true) 
        hb = (y_pred - y_true)**2 
        loss = hb * torch.abs(alpha - (diff < 0.).to(torch.float32)).detach()
        # loss = hb * torch.abs(-alpha + (diff < 0.).to(torch.float32)).detach()
        return loss
    

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  
    
        episode_num = batch['o'].shape[0]
        episode_len = batch['o'].shape[1]
        self.init_hidden(episode_num)
        for key in batch.keys():  
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, avail_u_next, terminated = batch['u'], batch['r'],  batch['avail_u'], \
                                                  batch['avail_u_next'], batch['terminated']
        u_onehot = batch['u_onehot']
        u_onehot_next = batch['u_onehot_next']
        s, s_next = batch['s'], batch['s_next']
        mask = (1 - batch["padded"].float()).squeeze(-1) 
        
        if self.args.cuda:
            u = u.cuda()
            r = r.cuda()
            avail_u = avail_u.cuda()
            avail_u_next = avail_u_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
            s = s.cuda()
            s_next = s_next.cuda()
            u_onehot = u_onehot.cuda()
            u_onehot_next = u_onehot_next.cuda()
        
        individual_q_evals, individual_q_targets, eval_hiddens, target_hiddens = self._get_individual_q(batch, max_episode_len)
        
        individual_q_max = individual_q_evals.max(dim=-1)[0]
        individual_q_clone = individual_q_evals.clone()
        individual_q_clone[avail_u == 0.0] = - 999999
        individual_q_targets[avail_u_next == 0.0] = - 999999


        opt_onehot_eval = torch.zeros(*individual_q_clone.shape)
        opt_action_eval = individual_q_clone.argmax(dim=3, keepdim=True)
        opt_onehot_eval = opt_onehot_eval.scatter(-1, opt_action_eval[:, :].cpu(), 1)


        opt_onehot_target = torch.zeros(*individual_q_targets.shape)
        opt_action_target = individual_q_targets.argmax(dim=3, keepdim=True)
        opt_onehot_target = opt_onehot_target.scatter(-1, opt_action_target[:, :].cpu(), 1)

        if self.args.td_type == "qv":
            joint_q_evals, v, v_next, joint_q_opt = self.get_qtran(batch, eval_hiddens, target_hiddens, opt_onehot_eval)
            y_dqn = r.squeeze(-1) + self.args.gamma * v_next * (1 - terminated.squeeze(-1))
            td_error = joint_q_evals - y_dqn.detach()
            l_td = ((td_error * mask) ** 2).sum() / mask.sum()

            self.joint_optimizer.zero_grad()
            l_td.backward()
            # torch.nn.utils.clip_grad_norm_(self.joint_parameters, self.args.grad_norm_clip)
            self.joint_optimizer.step()

            v_error = self.quantile_regression(joint_q_evals.detach(), v, 0.5)
            l_v = (v_error * mask).sum() / mask.sum()

            self.v_optimizer.zero_grad()
            l_v.backward()
            # torch.nn.utils.clip_grad_norm_(self.v.parameters(), self.args.grad_norm_clip)
            self.v_optimizer.step()
        elif self.args.td_type == "max":
            joint_q_evals, joint_q_opt, target_joint_q_opt = self.get_joint(batch, eval_hiddens, target_hiddens, target_opt_actions = opt_onehot_target)
            y_dqn = r.squeeze(-1) + self.args.gamma * target_joint_q_opt * (1 - terminated.squeeze(-1))
            td_error = joint_q_evals - y_dqn.detach()
            l_td = ((td_error * mask) ** 2).sum() / mask.sum()

            self.joint_optimizer.zero_grad()
            l_td.backward()
            # torch.nn.utils.clip_grad_norm_(self.joint_parameters, self.args.grad_norm_clip)
            self.joint_optimizer.step()
        elif self.args.td_type == "td_lambda":
            joint_q_evals, joint_q_opt, target_joint_q_opt, v = self.get_joint(batch, eval_hiddens, target_hiddens, target_opt_actions = opt_onehot_target, use_v=True)
            y_dqn = self.td_lambda_target(batch, max_episode_len, target_joint_q_opt.unsqueeze(-1), self.args)
            td_error = joint_q_evals - y_dqn.detach()
            l_td = ((td_error * mask) ** 2).sum() / mask.sum()

            self.joint_optimizer.zero_grad()
            l_td.backward()
            # torch.nn.utils.clip_grad_norm_(self.joint_parameters, self.args.grad_norm_clip)
            self.joint_optimizer.step()
          
        q_individual = torch.gather(individual_q_evals, dim=-1, index=u).squeeze(-1)
        if self.args.qmix_type == "mix2":
            agent_id = torch.eye(self.args.n_agents).unsqueeze(0).unsqueeze(0).expand(episode_num, episode_len, -1, -1).to(self.args.device)
            action_mask = 1 - torch.eye(self.args.n_agents)
            action_mask = action_mask.view(-1, 1).repeat(1, self.args.n_actions).view(1, self.args.n_agents, -1).to(self.args.device)
            # other_opt_actions = opt_onehot_eval.repeat(1, 1, 1, self.args.n_agents).to(self.args.device) * action_mask
            other_opt_actions = u_onehot.view(episode_num, episode_len, -1).repeat(1, 1, self.args.n_agents).view(episode_num, episode_len, self.n_agents, -1).to(self.args.device) * action_mask
            q_total_eval = self.eval_qmix_net(q_individual, s, other_opt_actions, agent_id).squeeze(-1)
        else:
            q_total_eval = self.eval_qmix_net(q_individual, s)

        if self.args.q_total_type == "individual":
            individual_q_targets[avail_u_next == 0.0] = - 9999999
            agent_id = torch.eye(self.args.n_agents).unsqueeze(0).unsqueeze(0).expand(episode_num, episode_len, -1, -1).to(self.args.device)
            q_targets = individual_q_targets.max(dim=3)[0]
            action_mask = 1 - torch.eye(self.args.n_agents)
            action_mask = action_mask.view(-1, 1).repeat(1, self.args.n_actions).view(1, self.args.n_agents, -1).to(self.args.device)
            # other_opt_actions = opt_onehot_target.repeat(1, 1, 1, self.args.n_agents).to(self.args.device) * action_mask
            other_opt_actions = u_onehot_next.view(episode_num, episode_len, -1).repeat(1, 1, self.args.n_agents).view(episode_num, episode_len, self.n_agents, -1).to(self.args.device) * action_mask
            q_total_target = self.target_qmix_net(q_targets, s_next, other_opt_actions, agent_id)
            q_total_target = (r + self.args.gamma * q_total_target * (1 - terminated)).squeeze(-1)
        elif self.args.q_total_type == "joint":
            q_total_target = joint_q_evals

        # ---------------------------------------------L_Qjoint-------------------------------------------------------------
        v_error = self.quantile_regression(q_total_target.detach(), v, self.args.alpha)
        l_v = (v_error* mask).sum() / mask.sum()
        
        # ---------------------------------------------L_Qi-------------------------------------------------------------  
        q_error = self.quantile_regression(q_total_target.detach(), q_total_eval, self.args.alpha)
        l_q = (q_error * mask).sum() / mask.sum()
        
        loss = l_q + l_v

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_joint_q.load_state_dict(self.eval_joint_q.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
       
        train_infos = []
        for i in range(self.n_agents):
            train_info = {}
            train_info['q_pred_u_mean'] = 0
            train_info['q_pred_a_mean'] = 0
            train_info['q_pred_a_min'] = 0
            train_info['q_pred_a_max'] = 0
            train_info['target'] = 0
            train_info['target_max'] = 0
            train_info['loss_q'] = 0
            train_info['loss_td'] = 0
            train_info['reward'] = 0
            train_info['beta'] = 0

            train_info['q_pred_u_mean'] += q_individual[:, :, i].mean().item()
            train_info['q_pred_a_mean'] += individual_q_evals[:, :, i].mean().item()
            train_info['q_pred_a_min'] += individual_q_evals[:, :, i].min().item()
            train_info['q_pred_a_max'] += individual_q_evals[:, :, i].max().item()
            train_info['target'] = joint_q_evals.mean().item()
            train_info['target_max'] = joint_q_evals.max().item()
            train_info['loss_td'] = l_td.item()
            train_info['loss_q'] = l_q.item()
            train_info['reward'] = r.mean().item()
            # train_info['beta'] = beta.item()
            train_infos.append(train_info)
        return train_infos

    def td_lambda_target(self, batch, max_episode_len, q_targets, args):
        episode_num = batch['o'].shape[0]
        mask = (1 - batch["padded"].float()).to(self.args.device)
        terminated = (1 - batch["terminated"].float()).to(self.args.device)
        r = batch['r'].to(self.args.device)
        # --------------------------------------------------n_step_return---------------------------------------------------
        n_step_return = torch.zeros((episode_num, max_episode_len, 1, max_episode_len)).to(self.args.device)
        for transition_idx in range(max_episode_len - 1, -1, -1):
            n_step_return[:, transition_idx, :, 0] = (r[:, transition_idx] + args.gamma * q_targets[:, transition_idx] * terminated[:, transition_idx]) * mask[:, transition_idx]        
            for n in range(1, max_episode_len - transition_idx):
                n_step_return[:, transition_idx, :, n] = (r[:, transition_idx] + args.gamma * n_step_return[:, transition_idx + 1, :, n - 1]) * mask[:, transition_idx]
        # --------------------------------------------------n_step_return---------------------------------------------------

        # --------------------------------------------------lambda return---------------------------------------------------
        '''
        lambda_return.shape = (episode_num, max_episode_len，n_agents)
        '''
        lambda_return = torch.zeros((episode_num, max_episode_len, 1)).to(self.args.device)
        for transition_idx in range(max_episode_len):
            returns = torch.zeros((episode_num, 1)).to(self.args.device)
            for n in range(1, max_episode_len - transition_idx):
                returns += pow(args.td_lambda, n - 1) * n_step_return[:, transition_idx, :, n - 1]
            lambda_return[:, transition_idx] = (1 - args.td_lambda) * returns + \
                                            pow(args.td_lambda, max_episode_len - transition_idx - 1) * \
                                            n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]
        # --------------------------------------------------lambda return---------------------------------------------------
        return lambda_return.squeeze(-1)

    def _get_individual_q(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        eval_hiddens = []
        target_hiddens = []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_individual_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            

            if transition_idx == 0:
                _, self.target_hidden = self.target_rnn(inputs, self.eval_hidden)

            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

     
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
            eval_hidden = self.eval_hidden.clone()
            eval_hidden = eval_hidden.view(episode_num, self.n_agents, -1)
            eval_hiddens.append(eval_hidden)

            target_hidden = self.target_hidden.clone()
            target_hidden = target_hidden.view(episode_num, self.n_agents, -1)
            target_hiddens.append(target_hidden)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        eval_hiddens = torch.stack(eval_hiddens, dim=1)
        target_hiddens = torch.stack(target_hiddens, dim=1)
        return q_evals, q_targets, eval_hiddens, target_hiddens

    def _get_individual_inputs(self, batch, transition_idx):
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        if self.args.last_action:
            if transition_idx == 0:  
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_joint(self, batch, eval_hiddens, target_hiddens, opt_actions=None, target_opt_actions=None, use_v=False):
        episode_num, max_episode_len = eval_hiddens.shape[0], eval_hiddens.shape[1]  
        states = batch['s'][:, :max_episode_len]
        states_next = batch['s_next'][:, :max_episode_len]
        u_onehot = batch['u_onehot'][:, :max_episode_len]
        q_opts = None
        
        if self.args.cuda:
            states = states.cuda()
            states_next = states_next.cuda()
            u_onehot = u_onehot.cuda()
            eval_hiddens = eval_hiddens.cuda()
            target_hiddens = target_hiddens.cuda()

        if opt_actions is not None:
            opt_actions = opt_actions[:, :max_episode_len].cuda()
            # q_opts = self.target_joint_q(states, eval_hiddens, opt_actions)
            q_opts = self.eval_joint_q(states, eval_hiddens, opt_actions)
            q_opts = q_opts.view(episode_num, -1, 1).squeeze(-1)
        
        if target_opt_actions is not None:
            target_opt_actions = target_opt_actions[:, :max_episode_len].cuda()
            target_q_opts = self.target_joint_q(states_next, target_hiddens, target_opt_actions)
            target_q_opts = target_q_opts.view(episode_num, -1, 1).squeeze(-1)

        q_evals = self.eval_joint_q(states, eval_hiddens, u_onehot)
        q_evals = q_evals.view(episode_num, -1, 1).squeeze(-1)
        if use_v:
            v = self.V(states).view(episode_num, -1)
        return q_evals, q_opts, target_q_opts, v

    def get_qtran(self, batch, eval_hiddens, target_hiddens, opt_actions=None):
        episode_num, max_episode_len = eval_hiddens.shape[0], eval_hiddens.shape[1]  
        states = batch['s'][:, :max_episode_len]
        states_next = batch['s_next'][:, :max_episode_len]
        u_onehot = batch['u_onehot'][:, :max_episode_len]
        q_opts = None
        
        if self.args.cuda:
            states = states.cuda()
            states_next = states_next.cuda()
            u_onehot = u_onehot.cuda()
            eval_hiddens = eval_hiddens.cuda()
            target_hiddens = target_hiddens.cuda()

        if opt_actions is not None:
            opt_actions = opt_actions[:, :max_episode_len].cuda()
            # q_opts = self.target_joint_q(states, eval_hiddens, opt_actions)
            q_opts = self.target_joint_q(states_next, target_hiddens, opt_actions)
            q_opts = q_opts.view(episode_num, -1, 1).squeeze(-1)

        q_evals = self.eval_joint_q(states, eval_hiddens, u_onehot)
        v = self.v(states)
        v_next = self.v(states_next)
        v = v.view(episode_num, -1, 1).squeeze(-1)
        v_next = v_next.view(episode_num, -1, 1).squeeze(-1)
        return q_evals, v, v_next, q_opts

    def init_hidden(self, episode_num):  
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
        torch.save(self.eval_joint_q.state_dict(), self.model_dir + '/' + num + '_joint_q_params.pkl')
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_params.pkl')
