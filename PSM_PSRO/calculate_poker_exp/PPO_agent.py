import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Normal, Categorical, OneHotCategorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR 
# from .net import PolicyNet, ValueNet
from open_spiel.python.policy import Policy
import random


class Model(nn.Module):
    def __init__(self, obs_shape=180, state_shape=249, action_shape=16, softmax=False):
        super().__init__()
        fc_layer1 = [nn.Linear(obs_shape, 64), nn.ReLU(inplace=True), ]
        fc_layer2 = [nn.Linear(64, 64), nn.ReLU(inplace=True), ]
        fc_layer3 = [nn.Linear(64, 64), nn.ReLU(inplace=True), ]
        fc_layer4 = [nn.Linear(64, action_shape)]

        fc_layers = fc_layer1 + fc_layer2 + fc_layer3 + fc_layer4
        self.act_model = nn.Sequential(*fc_layers)

        fc_layer1 = [nn.Linear(state_shape, 64), nn.ReLU(inplace=True), ]
        fc_layer2 = [nn.Linear(64, 64), nn.ReLU(inplace=True), ]
        fc_layer3 = [nn.Linear(64, 64), nn.ReLU(inplace=True), ]
        fc_layer4 = [nn.Linear(64, 1)]

        fc_layers = fc_layer1 + fc_layer2 + fc_layer3 + fc_layer4
        self.value_model = nn.Sequential(*fc_layers)


        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Forward inference
    def forward(self, obs, state, available_actions=None, sample_actions=None):
        #[B,N,D] for train, [N,D] for eval
        logits = self.act_model(obs) 
        
        # value = self.value_model(state)
        # if available_actions is not None:
        #     logits = logits + (1 - available_actions) * -1e10

        # dist = Categorical(logits=logits)
        # d_actions = dist.probs.argmax(dim=-1) 
        # actions = sample_actions if sample_actions is not None else dist.sample()
        # action_log_probs = dist.log_prob(actions)
        # entropy = dist.entropy()

        return logits

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

def kl_divergence(prob_a, prob_b):
    
    ori_shape = prob_b.shape
    prob_a = prob_a.reshape(-1)
    prob_b = prob_b.reshape(-1)
    prob_b[prob_a==0] = - np.inf

    prob_b = prob_b.reshape(ori_shape)
    prob_b = torch.softmax(prob_b, -1)
    prob_b = prob_b.reshape(-1)
    
    res = (prob_a[prob_a>0] * torch.log(prob_a[prob_a>0]/prob_b[prob_a>0])).sum()

    return res


class PPOAgent(Policy):

    batch_size = 512
    buffer_capacity = int(1e4)
    lr = 5e-3
    epsilon = 0.05
    gamma = 1
    target_update = 5

    def __init__(self, game, playerids, state_dim, hidden_dim,
                 action_dim, device, ckp_dir):

        super().__init__(game, playerids)
        self.device = device
        self.action_dim = action_dim
        self.policy_net = Model(obs_shape=state_dim, state_shape=60, action_shape=action_dim).to(self.device)
        # self.value_net = ValueNet(state_dim, hidden_dim).to(self.device)
            
        self.buffer = []
        self.buffer_count = 0
        self.counter = 0
        # self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        # self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.clip_param = 0.2
        self.entropy_coef = 0.001
        self.max_grad_norm = 10
    
    def action_probabilities(self, state, player_id=None):

        cur_player = state.current_player()
        s = state.information_state_tensor(cur_player)
        legal_actions = state.legal_actions()
        all_act_probs, _, _ = self.select_action(s, legal_actions, noise=False)

        return dict(zip(legal_actions, all_act_probs[legal_actions]))
            
    def select_action(self, state, legal_actions=None, noise=True):
        state = torch.tensor(state).unsqueeze(0).to(self.device)
        if not legal_actions is None:
            with torch.no_grad():
                all_act = self.policy_net(state, 0)[0]
            legal_act = torch.ones_like(all_act) * - np.inf
            legal_act[legal_actions] = all_act[legal_actions]
            action_dist = Categorical(logits=legal_act)
            action = action_dist.mode.item()
            act_prob = action_dist.logits.detach().cpu()
        else:
            with torch.no_grad():
                logits = self.ploicy_net(state)
                action_dist = Categorical(logits=logits)
                action = action_dist.mode().item()
                act_prob = action_dist.logits.detach().cpu()

        return action_dist.probs.detach().cpu(), action, act_prob[action]
    
    def get_value(self, state):
        state = torch.tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.value_net(state).item()
        return value
        
    def store_transition(self, transition):
        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(transition)
            self.buffer_count += 1
        else:
            index = int(self.buffer_count % self.buffer_capacity)
            self.buffer[index] = transition
            self.buffer_count += 1
    
    def clean_buffer(self):
        self.buffer = []

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def update(self, anchor=None, div_weight=None, div_mask=False):
        self.policy_net = self.policy_net.to(self.device)
        self.value_net = self.value_net.to(self.device)
        self.batch_size = len(self.buffer) 
        sample_data = random.sample(self.buffer, self.batch_size)
        state = torch.tensor([t.state for t in sample_data]).to(self.device)
        action = torch.tensor([t.action for t in sample_data]).view(-1,1).to(torch.int64).to(self.device)
        rew = torch.tensor([t.reward for t in sample_data]).to(self.device)
        done = torch.tensor([t.done for t in sample_data]).to(self.device)
        old_value = torch.tensor([t.value for t in sample_data]).to(self.device)
        adv_targ = torch.tensor([t.advantage for t in sample_data]).to(torch.float).to(self.device)
        old_action_log_prob = torch.stack([t.a_log_prob for t in sample_data], axis=0).to(self.device)
        diversity_mask = adv_targ <= 0
        legal_actions = torch.stack([t.legal_action.to(torch.int64).to(self.device) for t in sample_data], axis=0)
        
        logits = self.policy_net(state)

        legal_act = logits.clone()
        legal_act[legal_actions==0] = -np.inf
        action_dist = Categorical(logits=legal_act)

        action_log_prob = action_dist.log_prob(action.squeeze(-1))
        entropy = action_dist.entropy()

        imp_weights = torch.exp(action_log_prob - old_action_log_prob)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        ppo_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        ppo_loss = ppo_loss - self.entropy_coef * entropy.mean()

        # q_values = self.q_net(state).gather(1, action)
        # max_next_q_values = self.tar_q_net(next_state).max(1)[0]
        # q_targets = (reward + self.gamma * max_next_q_values * (1 - done)).view(-1, 1)
        # q_targets = reward.view(-1, 1)

        # dqn_loss = F.mse_loss(q_values, q_targets)
        if not anchor is None:
            # logits = self.policy_net(state)
            logits[legal_actions==0] = -np.inf
            act_prob_main = torch.softmax(logits, -1)
            # act_prob_main[actions_prob==0] = 0
            with torch.no_grad():
                logits_archor = anchor.policy_net(state)
                logits_archor[legal_actions==0] = -np.inf
                act_prob_anchor = torch.softmax(logits_archor, -1)
            if div_mask:
                # cross_entropy_loss = F.cross_entropy(act_prob_main, act_prob_anchor, reduction='none',)
                kl_loss = - kl_divergence(act_prob_main * diversity_mask.unsqueeze(-1), act_prob_anchor * diversity_mask.unsqueeze(-1))/ diversity_mask.sum() * div_weight
            else:
                kl_loss = - kl_divergence(act_prob_main, act_prob_anchor) / self.batch_size * div_weight
        # print(dqn_loss.item())
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optimizer.zero_grad() 
        if not anchor is None:
            loss = ppo_loss + kl_loss
            loss.backward()
        else:        
            ppo_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_pred = self.value_net(state)
        value_loss = F.mse_loss(value_pred.squeeze(-1), adv_targ)
        value_loss.backward()
        self.value_optimizer.step()