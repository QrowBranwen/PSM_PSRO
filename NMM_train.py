import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pickle
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
np.random.seed(123)
import copy
from scipy.spatial.distance import pdist, squareform

dim = 30
payoffs = np.tril(np.random.uniform(-1, 1, (dim,dim)), -1)
payoffs = (payoffs - payoffs.T)

LR = 0.1
TRAIN_ITERS = 20

expected_card = []
sizes = []

# time_string = time.strftime("%Y%m%d-%H%M%S")
# PATH_RESULTS = os.path.join('results', time_string)
PATH_RESULTS = os.path.join('results', "NMM_abl")
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)

device = 'cpu'

# FILE_TRAJ = {
#     'rectified': 'rectified.p',
#     'psro': 'psro.p',
#     'p-psro': 'p_psro.p',
#     'dpp': 'dpp.p',
#     'psd': 'psd.p', 
#              }

def distance_solver(A, b):
    I = np.identity(A.shape[0])
    A_pinv = np.linalg.pinv(A)
    I_minus_AA_pinv = I - A @ A_pinv

    One_tensor = torch.tensor(np.ones(shape=(A.shape[1], 1)), requires_grad=False, dtype=torch.float32)
    A_pinv_tensor = torch.tensor(A_pinv, requires_grad=False, dtype=torch.float32)
    I_minus_AA_pinv_tensor = torch.tensor(I_minus_AA_pinv, requires_grad=False, dtype=torch.float32)

    Sigma_min = min(np.linalg.svd(A.T, full_matrices=True)[1])
    distance = ((Sigma_min ** 2) / A.shape[1]) * ((1 - (One_tensor.t() @ A_pinv_tensor @ b)[0, 0]) ** 2) + torch.square(
        I_minus_AA_pinv_tensor @ b).sum()
    return distance

def kl_divergence(prob_a, prob_b):
    prob_a += 1e-3
    prob_a /= prob_a.sum()
    prob_b += 1e-3
    prob_b = (prob_b.T / prob_b.sum(1)).T
    res = prob_a * torch.log(prob_a/prob_b)
    return res.sum(1)

class MyGaussianPDF(nn.Module):
    def __init__(self, mu):
        super(MyGaussianPDF, self).__init__()
        self.mu = mu
        self.cov = 0.54*torch.eye(2)
        # self.c = (1./(2*np.pi))
        self.c = 1.

    def forward(self, x):
        return self.c*torch.exp(-0.5*torch.diagonal( (x-self.mu)@self.cov@(x-self.mu).t() ))

class GMMAgent(nn.Module):
    def __init__(self, mu):
        super(GMMAgent, self).__init__()
        self.gauss = MyGaussianPDF(mu).to(device)
        self.x = nn.Parameter(0.01*torch.randn(2, dtype=torch.float), requires_grad=False)

    def forward(self):
        return self.gauss(self.x)

class TorchPop:
    def __init__(self, num_learners, seed=0):
        torch.manual_seed(seed)
        self.pop_size = num_learners + 1

        mus = np.array([[2.8722, -0.025255],
                        [1.8105, 2.2298],
                        [1.8105, -2.2298],
                        [-0.61450, 2.8058],
                        [-0.61450, -2.8058],
                        [-2.5768, 1.2690],
                        [-2.5768, -1.2690]]
                       )
        mus = torch.from_numpy(mus).float().to(device)
        self.mus = mus

        self.game = torch.from_numpy(np.array([
                                               [0., 1., 1., 1, -1, -1, -1],
                                               [-1., 0., 1., 1., 1., -1., -1.],
                                               [-1., -1., 0., 1., 1., 1., -1],
                                               [-1., -1., -1., 0, 1., 1., 1.],
                                               [1., -1., -1., -1., 0., 1., 1.],
                                               [1., 1., -1., -1, -1, 0., 1.],
                                               [1., 1., 1., -1., -1., -1., 0.]
                                               ])).float().to(device)

        self.pop = [GMMAgent(mus) for _ in range(self.pop_size)]
        self.pop_hist = [[self.pop[i].x.detach().cpu().clone().numpy()] for i in range(self.pop_size)]

    def get_js_divergence(self, agent1, metanash, K):
        def entropy(p_k):
            p_k = p_k + 1e-8
            p_k = p_k / torch.sum(p_k)
            return -torch.sum(p_k * torch.log(p_k))

        agg_agent = metanash[0] * self.pop[0]()
        for k in range(1, K):
            agg_agent += metanash[k] * self.pop[k]()
        agent1_values = agent1()
        agent1_values = agent1_values / torch.sum(agent1_values)
        agg_agent = agg_agent / torch.sum(agg_agent)
        return 2 * entropy((agent1_values + agg_agent) / 2) - entropy(agent1_values) - entropy(agg_agent)

    def visualise_pop(self, br=None, ax=None, color=None):

        def multivariate_gaussian(pos, mu, Sigma):
            """Return the multivariate Gaussian distribution on array pos."""

            n = mu.shape[0]
            Sigma_det = np.linalg.det(Sigma)
            Sigma_inv = np.linalg.inv(Sigma)
            N = np.sqrt((2 * np.pi) ** n * Sigma_det)
            # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
            return np.exp(-fac / 2) / N

        agents = [agent.x.detach().cpu().numpy() for agent in self.pop]
        agents = list(zip(*agents))

        # Colors
        if color is None:
            colors = cm.rainbow(np.linspace(0, 1, len(agents[0])))
        else:
            colors = [color]*len(agents[0])

        # fig = plt.figure(figsize=(6, 6))
        ax.scatter(agents[0], agents[1], alpha=1., marker='.', color=colors, s=8*plt.rcParams['lines.markersize'] ** 2)
        if br is not None:
            ax.scatter(br[0], br[1], marker='.', c='k')
        for i, hist in enumerate(self.pop_hist):
            if hist:
                hist = list(zip(*hist))
                ax.plot(hist[0], hist[1], alpha=0.8, color=colors[i], linewidth=4)

        # ax = plt.gca()
        for i in range(7):
            ax.scatter(self.mus[i, 0].item(), self.mus[i, 1].item(), marker='x', c='k')
            for j in range(4):
                delta = 0.025
                x = np.arange(-4.5, 4.5, delta)
                y = np.arange(-4.5, 4.5, delta)
                X, Y = np.meshgrid(x, y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                Z = multivariate_gaussian(pos, self.mus[i,:].numpy(), 0.54 * np.eye(2))
                levels = 10
                # levels = np.logspace(0.01, 1, 10, endpoint=True)
                CS = ax.contour(X, Y, Z, levels, colors='k', linewidths=0.5, alpha=0.2)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
                # ax.clabel(CS, fontsize=9, inline=1)
                # circle = plt.Circle((0, 0), 0.2, color='r')
                # ax.add_artist(circle)
        ax.set_xlim([-4.5, 4.5])
        ax.set_ylim([-4.5, 4.5])


    def get_payoff(self, agent1, agent2):
        p = agent1()
        q = agent2()
        return p @ self.game @ q + 0.5*(p-q).sum()

    def get_payoff_aggregate(self, agent1, metanash, K):
        # Computes the payoff of agent1 against the aggregated first :K agents using metanash as weights
        agg_agent = metanash[0]*self.pop[0]()
        for k in range(1, K):
            agg_agent += metanash[k]*self.pop[k]()
        return agent1() @ self.game @ agg_agent + 0.5*(agent1()-agg_agent).sum()

    def get_payoff_aggregate_weights(self, agent1, weights, K):
        # Computes the payoff of agent1 against the aggregated first :K agents using metanash as weights
        agg_agent = weights[0]*self.pop[0]()
        for k in range(1, len(weights)):
            agg_agent += weights[k]*self.pop[k]()
        return agent1() @ self.game @ agg_agent + 0.5*(agent1()-agg_agent).sum()

    def get_br_to_strat(self, metanash, lr, nb_iters=20, BR=None, early_stop=np.inf):
        if BR is None:
            br = GMMAgent(self.mus)
            br.x = nn.Parameter(0.1*torch.randn(2, dtype=torch.float), requires_grad=False)
            br.x.requires_grad = True
        else:
            br =  BR
        optimiser = optim.Adam(br.parameters(), lr=lr)
        best_loss = np.inf
        best_br = None
        count = 0
        for _ in range(nb_iters*50):
            loss = -self.get_payoff_aggregate(br, metanash, self.pop_size,)
            # Optimise !
            if loss < best_loss:
                count = 0
                best_loss = loss
                best_br = copy.deepcopy(br)
            else:
                count += 1

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if count > early_stop:
                return best_br

        return br

    def get_metagame(self, k=None, numpy=False):
        if k==None:
            k = self.pop_size
        if numpy:
            with torch.no_grad():
                metagame = torch.zeros(k, k)
                for i in range(k):
                    for j in range(k):
                        metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j])
                return metagame.detach().cpu().clone().numpy()
        else:
            metagame = torch.zeros(k, k)
            for i in range(k):
                for j in range(k):
                    metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j])
            return metagame

    def add_new(self):
        with torch.no_grad():
            self.pop.append(GMMAgent(self.mus))
            self.pop_hist.append([self.pop[-1].x.detach().cpu().clone().numpy()])
            self.pop_size += 1

    def get_exploitability(self, metanash, lr, nb_iters=20):
        br = self.get_br_to_strat(metanash, lr, nb_iters=nb_iters)
        with torch.no_grad():
            exp = self.get_payoff_aggregate(br, metanash, self.pop_size).item()

        return exp

def fsp_non_symmetric_game(emp_game_matrix, iters=1000):
    row_player_dim = emp_game_matrix.shape[0]
    column_player_dim = emp_game_matrix.shape[1]
    row_avg = np.random.uniform(0, 1, row_player_dim)
    row_avg = row_avg / row_avg.sum()
    column_avg = np.random.uniform(0, 1, column_player_dim)
    column_avg = column_avg / column_avg.sum()
    for i in range(iters):
        # row_avg = np.average(row_pop, axis=0)
        # column_avg = np.average(column_pop, axis=0)
        br_column = get_br_to_strat(row_avg, emp_game_matrix)
        br_row = get_br_to_strat(column_avg, -emp_game_matrix.T)
        row_avg = (row_avg * (i+1) + br_row) / (i+2)
        column_avg = (column_avg * (i+1) + br_column) / (i+2)
    # row_avg = np.average(row_pop, axis=0)
    # column_avg = np.average(column_pop, axis=0)
    # print(f"Nash is {row_avg}")
    return - row_avg @ emp_game_matrix @ column_avg.T


def fp_for_non_symmetric_game(emp_game_matrix, iters=1000):
    row_player_dim = emp_game_matrix.shape[0]
    column_player_dim = emp_game_matrix.shape[1]
    row_avg = np.random.uniform(0, 1, row_player_dim)
    row_avg = row_avg / row_avg.sum()
    column_avg = np.random.uniform(0, 1, column_player_dim)
    column_avg = column_avg / column_avg.sum()
    for i in range(iters):
        # row_avg = np.average(row_pop, axis=0)
        # column_avg = np.average(column_pop, axis=0)
        br_column = get_br_to_strat(row_avg, emp_game_matrix)
        br_row = get_br_to_strat(column_avg, -emp_game_matrix.T)
        row_avg = (row_avg * (i+1) + br_row) / (i+2)
        column_avg = (column_avg * (i+1) + br_column) / (i+2)
    # row_avg = np.average(row_pop, axis=0)
    # column_avg = np.average(column_pop, axis=0)
    # print(f"Nash is {row_avg}")
    return row_avg, column_avg, row_avg @ emp_game_matrix @ column_avg.T


def get_pop_effectivity(self, iters=100, br_iter=20, verbose=False):
    pop_size = len(self.pop)
    curr_pop_metagame = self.get_metagame(numpy=True)
    # metanash = fictitious_play(iters=2000, payoffs=curr_pop_metagame)[0][-1]
    metanash = np.random.random(size=(pop_size,))
    opponent_pop = [self.get_br_to_strat(metanash / sum(metanash), lr=LR, nb_iters=br_iter)]

    meta_game = []
    nash_value_list = []
    with torch.no_grad():
        row_vec = []
        for i in range(pop_size):
            row_vec.append(self.get_payoff(self.pop[i], opponent_pop[0]).item())
    meta_game.append(row_vec)
    for i in range(iters):
        # solve
        nash_row, _, nash_value = fp_for_non_symmetric_game(emp_game_matrix=np.array(meta_game).T, iters=1000)
        # scale for column player
        br_column = self.get_br_to_strat(nash_row, lr=0.1, nb_iters=br_iter)
        opponent_pop.append(br_column)
        row_vec = []
        for j in range(pop_size):
            row_vec.append(self.get_payoff(self.pop[j], br_column).item())
        meta_game.append(row_vec)
        if verbose:
            print(f"{i}th iterations: nash value is {nash_value}")
        nash_value_list.append(nash_value)
    return meta_game, nash_value_list

def gradient_loss_update(torch_pop, k, train_iters=10, lambda_weight=0.1, lr=0.1, div=None, cmd=1):

    # We compute metagame M and then L in a differentiable way
    # We compute expected payoff of agent k-1 against aggregated strat

    # Make strategy k trainable
    # lambda_weight = 0.5
    torch_pop.pop[k].x.requires_grad = True

    # Optimiser
    optimiser = optim.Adam(torch_pop.pop[k].parameters(), lr=lr)
    # Get metagame and metastrat
    with torch.no_grad():
        M = torch_pop.get_metagame(k=k+1)
        L = M @ M.t()  # Compute kernel
        udm = torch.linalg.eigvals((L + 1) ** 3)
        L_card_base = torch.sum(1 / (1 + torch.exp(-udm)) - 0.5).real
    meta_nash = fictitious_play(payoffs=M.detach().cpu().clone().numpy()[:k, :k], iters=2000)[0][-1]

    # if div == "p_dpp":
    #     length = min(len(meta_nash), 5)
    #     topk_ids = torch.topk(torch.tensor(meta_nash), k=length).indices
    #     all_ids = torch.arange(len(meta_nash))
    #     leftover_ids = all_ids[~torch.isin(all_ids, topk_ids)]
    #     if len(leftover_ids) > 0:
    #         leftover_ids = leftover_ids[torch.randperm(len(leftover_ids))]
    #         rand_ids = leftover_ids[:2]
    #     else:
    #         rand_ids = torch.tensor([], dtype=torch.long)
    #     selected_ids = torch.cat([topk_ids, rand_ids])
    #     meta_nash_tensor = torch.tensor(meta_nash, dtype=torch.float)
    #     mask = torch.zeros_like(meta_nash_tensor, dtype=torch.bool)
    #     mask[selected_ids] = True
    #     meta_nash_tensor[~mask] = 0.0
    #     total = meta_nash_tensor.sum()
    #     if total > 0:
    #         meta_nash_tensor = meta_nash_tensor / total
    #     meta_nash = meta_nash_tensor.numpy()
    #     topk_ids = torch.cat((selected_ids, torch.tensor([k])))
    #     M = M[topk_ids][:, topk_ids]

    for iter in range(train_iters):
        # Get metagame and metastrat
        M = torch_pop.get_metagame(k=k+1)
        # meta_nash = fictitious_play(payoffs=M.detach().cpu().clone().numpy()[:k, :k], iters=2000)[0][-1]
        # Compute cardinality of pop up until :k UNION training strategy. We use payoffs as features.
        if div == "p_dpp":
            M_mu = M.mean()
            M_sigma = M.std()
            M = (M - M_mu) / M_sigma
            L = torch.exp(-torch.cdist(M, M, p=2) ** 2 / (2 * 4 ** 2))

            M =  f.normalize(M,dim=1,p=2) #  Normalise
            cosine_sim = M @ M.t()
            if cmd == 1:
                # redundancy_penalty = -(torch.sum(L[-1, :-1] * (cosine_sim[-1, :-1] + 1) / 2 )) ** 0.5
                redundancy_penalty = -(torch.sum(L[-1, :-1] * cosine_sim[-1, :-1]))
            elif cmd == 2:
                redundancy_penalty = -(torch.sum(L[-1, :-1]))
            else:
                redundancy_penalty = -(torch.sum(cosine_sim[-1, :-1]))
            L_card = 1 * redundancy_penalty
            exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[k], meta_nash, k)
            # Loss
            loss = -(lambda_weight * (exp_payoff) + (1. - lambda_weight) * L_card)
        if div == "dpp":
            L = M @ M.t()  # Compute kernel
            udm = torch.linalg.eigvals((L+1) ** 3)
            L_card = torch.sum(1 / (1 + torch.exp(-udm)) - 0.5).real
            exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[k], meta_nash, k)
            # Loss
            loss = -(lambda_weight * exp_payoff + (1. - lambda_weight) * L_card)

        elif div == "psd":
            div_w = 0.3
            x_haved = [torch_pop.pop[i]().detach() for i in range(k)]
            x_haved = torch.stack(x_haved, dim=0)
            psd_div = kl_divergence(torch_pop.pop[k](), x_haved).min() * div_w
            # Compute the expected return given that enemy plays agg_strat (using :k first strats)
            exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[k], meta_nash, k)
            # Loss
            loss = -(lambda_weight * exp_payoff + (1. - lambda_weight) * psd_div)
        elif div == "bd_rd":
            M = M[:, :-1]
            # meta_nash = fictitious_play(payoffs=M.detach().cpu().clone().numpy()[:k, :k], iters=1000)[0][-1]
            old_payoff = M[0:-1].t().detach().cpu().clone().numpy()
            new_vec = M[-1].reshape(-1, 1)
            distance = distance_solver(old_payoff, new_vec)
            js_divergence = torch_pop.get_js_divergence(torch_pop.pop[k], meta_nash, k) * 150
            exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[k], meta_nash, k)
            loss = -(lambda_weight * exp_payoff + (1. - lambda_weight/2) * js_divergence + (1. - lambda_weight/2) * distance)

        elif div is None:
            # Compute the expected return given that enemy plays agg_strat (using :k first strats)
            exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[k], meta_nash, k)
            # Loss
            loss = -(lambda_weight * exp_payoff)

        # Optimise !
        pre_exp_payoff = exp_payoff.detach()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if (iter + 1) % 5 == 0 :
            torch_pop.pop_hist[k].append(torch_pop.pop[k].x.detach().cpu().clone().numpy())

    # Make strategy k non-trainable
    torch_pop.pop[k].x.requires_grad = False
    return exp_payoff.item()


def psro_gradient(iters=5, num_learners=4, lr=.2, train_iters=10, div=None, seed=0, cmd=1):

    # Generate population
    torch_pop = TorchPop(num_learners, seed=seed)
    train_iters = int(train_iters / num_learners)

    # Compute initial exploitability and init stuff
    metagame = torch_pop.get_metagame(numpy=True)
    metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1] #np.array([1., 0.]) #
    exp = torch_pop.get_exploitability(metanash, lr, nb_iters=train_iters)
    # pe = fsp_non_symmetric_game(emp_game_matrix=metagame, iters=1000)
    exps = [exp]
    L_cards = [0.]
    ores = []

    for i in range(iters):
        # Define the weighting towards diversity
        lambda_weight = 0. + (0.7 / (1 + np.exp(-0.25*(i-(25)))) )
        lambda_weight = 1. - lambda_weight
        
        for j in range(num_learners):
            k = torch_pop.pop_size - 1

            exp_payoff = gradient_loss_update(torch_pop, k, train_iters=train_iters, lr=lr,
                                                        lambda_weight=lambda_weight, div=div, cmd=cmd)
                       
        metagame = torch_pop.get_metagame(numpy=True)
        metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
        exp = torch_pop.get_exploitability(metanash, lr, nb_iters=train_iters)
        
        M = metagame 
        L = M @ M.T  
        udm = np.linalg.eigvals((L + 1) ** 3)
        L_card = np.sum(1 / (1 + np.exp(-udm)) - 0.5).real
        M = (M - np.mean(M)) / np.std(M)
        pairwise_dists = squareform(pdist(M, metric='euclidean') ** 2)
        L = np.exp(-pairwise_dists / (2 * 4 ** 2))
        row_norms = np.linalg.norm(M, axis=1, keepdims=True)
        M = M / row_norms
        cosine_similarity = M @ M.T
        L = L * cosine_similarity
        sim_L =  L[-num_learners, :-num_learners]
        
        redundancy_penalty = np.sum(sim_L)
        
        L_cards.append(L_card)
        exps.append(exp)
        ores.append(metagame)
        torch_pop.add_new()

        if i % 1 == 0:
            print('ITERATION: ', i, ' exp full: {:.4f}'.format(exps[-1]), 'ore:{:.4f}'.format(redundancy_penalty),'lw: {:.3f}'.format(lambda_weight))

    return torch_pop, exps, ores, L_cards, L


def gradient_loss_update_rectified(torch_pop, k, weights, train_iters=10, lr=0.1):

    # Make strategy k trainable
    torch_pop.pop[k].x.requires_grad = True

    # Optimiser
    optimiser = optim.Adam(torch_pop.pop[k].parameters(), lr=lr)

    for iter in range(train_iters):

        # Get metagame and metastrat
        M = torch_pop.get_metagame(k=k+1)
        # Compute the expected return given that enemy plays agg_strat (using :k first strats)
        exp_payoff = torch_pop.get_payoff_aggregate_weights(torch_pop.pop[k], weights, k)

        # Loss
        loss = -exp_payoff

        # Optimise !
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    torch_pop.pop_hist[k].append(torch_pop.pop[k].x.detach().cpu().clone().numpy())

    # Make strategy k non-trainable
    torch_pop.pop[k].x.requires_grad = False
    return exp_payoff.item()


# Define the PSRO rectified nash algorithm
def psro_rectified_gradient(iters=10, eps=1e-2, seed=0, train_iters=10,
                         num_pseudo_learners=4, lr=0.3):
    # Generate population
    torch_pop = TorchPop(num_pseudo_learners, seed=seed)

    # Compute initial exploitability and init stuff
    metagame = torch_pop.get_metagame(numpy=True)
    metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
    exp = torch_pop.get_exploitability(metanash, lr, nb_iters=train_iters)
    exps = [exp]

    counter = 0
    while counter < iters * num_pseudo_learners:
        # if counter % (5 * num_pseudo_learners) == 0:
            # print('iteration: ', int(counter / num_pseudo_learners), ' exp: ', exps[-1])
            # print('size of population: ', torch_pop.pop_size)

        new_pop = copy.deepcopy(torch_pop)
        emp_game_matrix = torch_pop.get_metagame(numpy=True)
        averages, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)

        # go through all policies. If the policy has positive meta Nash mass,
        # find policies it wins against, and play against meta Nash weighted mixture of those policies
        for j in range(torch_pop.pop_size):
            if counter > iters * num_pseudo_learners:
                # fig1, axs1 = plt.subplots(1, 1)
                # torch_pop.visualise_pop(br=None, ax=axs1)
                # plt.savefig(os.path.join(PATH_RESULTS, 'trajectories_rectified.pdf'))
                return torch_pop, exps
            # if positive mass, add a new learner to pop and update it with steps, submit if over thresh
            # keep track of counter
            if averages[-1][j] > eps:
                # create learner
                new_pop.add_new()
                idx = new_pop.pop_size - 1
                counter += 1
                print(counter)

                mask = emp_game_matrix[j, :]
                mask += 1e-5
                mask[mask >= 0] = 1
                mask[mask < 0] = 0
                weights = np.multiply(mask, averages[-1])
                weights /= weights.sum()

                exp_payoff = gradient_loss_update_rectified(new_pop, idx, weights,
                                                                    train_iters=train_iters, lr=lr)

                if counter % num_pseudo_learners == 0:
                    metagame = new_pop.get_metagame(numpy=True)
                    metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
                    exp = new_pop.get_exploitability(metanash, lr, nb_iters=train_iters)
                    exps.append(exp)
        torch_pop = copy.deepcopy(new_pop)

    return torch_pop, exps


#Search over the pure strategies to find the BR to a strategy
def get_br_to_strat(strat, payoffs=payoffs, verbose=False):
    row_weighted_payouts = strat@payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br

#Fictituous play as a nash equilibrium solver
def fictitious_play(iters=2000, payoffs=payoffs, verbose=False):
    dim = payoffs.shape[0]
    pop = np.random.uniform(0,1,(1,dim))
    pop = pop/pop.sum(axis=1)[:,None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        br = get_br_to_strat(average, payoffs=payoffs)
        exp1 = average@payoffs@br.T
        exp2 = br@payoffs@average.T
        exps.append(exp2-exp1)
        # if verbose:
        #     print(exp, "exploitability")
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))
    return averages, exps


def run_experiments(num_experiments=1, num_threads=20, iters=40,
                    rectified=False, psro=False, psd_psro=False,
                    pipeline_psro=False, dpp_psro=False, bd_rd=False, p_dpp_psro=False,
                    abl_gauss_psro=False, psm_10_psro=False, abl_cos_psro=False, psm_20_psro=False, psm_25_psro=False,
                    yscale='none', verbose=False, train_iters=10):

    rectified_exps = []
    rectified_cardinality = []

    psro_exps = []
    psro_ores = []
    psro_pes = []
    psro_cardinality = []

    pipeline_exps = []
    pipeline_ores = []
    pipeline_pes = []
    pipeline_cardinality = []

    dpp_exps = []
    dpp_pes = []
    dpp_ores = []
    dpp_cardinality = []

    abl_gauss_exps = []
    abl_gauss_ores = []
    abl_gauss_pes = []
    abl_gauss_cardinality = []

    # psm_10_exps = []
    # psm_10_ores = []
    # psm_10_pes = []
    # psm_10_cardinality = []

    abl_cos_exps = []
    abl_cos_ores = []
    abl_cos_pes = []
    abl_cos_cardinality = []

    psm_20_exps = []
    psm_20_ores = []
    psm_20_pes = []
    psm_20_cardinality = []

    psm_25_exps = []
    psm_25_ores = []
    psm_25_pes = []
    psm_25_cardinality = []

    p_dpp_exps = []
    p_dpp_pes = []
    p_dpp_ores = []
    p_dpp_cardinality = []

    psd_exps = []
    psd_ores = []
    psd_pes = []
    psd_cardinality = []

    bd_rd_exps = []
    bd_rd_ores = []
    bd_rd_pes = []
    bd_rd_cardinality = []

    for i in range(num_experiments):
        print('Experiment: ', i + 1)
        if psd_psro:
            print('PSD PSRO')
            torch_pop, exps, ores, L_cards, pes = psro_gradient(iters=iters, num_learners=num_threads, lr=LR, train_iters=train_iters, seed=i,
                                          div="psd")
            psd_exps.append(exps)
            psd_ores.append(ores)
            psd_pes.append(pes)
            psd_cardinality.append(L_cards)
            pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['psd'])+f'{i}.p', 'wb'))
        if bd_rd:
            print('BD&RD PSRO')
            torch_pop, exps, ores, L_cards, pes = psro_gradient(iters=iters, num_learners=num_threads, lr=LR, train_iters=train_iters, seed=i,
                                          div="bd_rd")
            bd_rd_exps.append(exps)
            bd_rd_ores.append(ores)
            bd_rd_pes.append(pes)
            bd_rd_cardinality.append(L_cards)
            pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['bd_rd'])+f'{i}.p', 'wb'))
        if dpp_psro:
            print('Diverse PSRO')
            torch_pop, exps, ores, L_cards, pes = psro_gradient(iters=iters, num_learners=num_threads, lr=LR, train_iters=train_iters, seed=i,
                                          div="dpp")
            dpp_exps.append(exps)
            dpp_ores.append(ores)
            dpp_pes.append(pes)
            dpp_cardinality.append(L_cards)
            pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['dpp'])+f'{i}.p', 'wb'))
        if p_dpp_psro:
            print('p_ Diverse PSRO')
            torch_pop, exps, ores, L_cards, pes = psro_gradient(iters=iters, num_learners=num_threads, lr=LR, train_iters=train_iters, seed=i,
                                          div="p_dpp", cmd=1)
            p_dpp_exps.append(exps)
            p_dpp_ores.append(ores)
            p_dpp_pes.append(pes)
            p_dpp_cardinality.append(L_cards)
            pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['p_dpp'])+f'{i}.p', 'wb'))

        if abl_gauss_psro:
            print('P-SM 0.5 PSRO')
            torch_pop, exps, ores, L_cards, pes = psro_gradient(iters=iters, num_learners=num_threads, lr=LR, train_iters=train_iters, seed=i,
                                          div="p_dpp", cmd=2)
            abl_gauss_exps.append(exps)
            abl_gauss_ores.append(ores)
            abl_gauss_pes.append(pes)
            abl_gauss_cardinality.append(L_cards)
            pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['abl_gauss'])+f'{i}.p', 'wb'))

        # if psm_10_psro:
        #     print('P-SM 1.0 PSRO')
        #     torch_pop, exps, ores, L_cards, pes = psro_gradient(iters=iters, num_learners=num_threads, lr=LR, train_iters=train_iters, seed=i,
        #                                   div="p_dpp", sigma=1)
        #     psm_10_exps.append(exps)
        #     psm_10_ores.append(ores)
        #     psm_10_pes.append(pes)
        #     psm_10_cardinality.append(L_cards)
        #     pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['psm_10'])+f'{i}.p', 'wb'))

        if abl_cos_psro:
            print('P-SM 1.5 PSRO')
            torch_pop, exps, ores, L_cards, pes = psro_gradient(iters=iters, num_learners=num_threads, lr=LR, train_iters=train_iters, seed=i,
                                          div="p_dpp", cmd=3)
            abl_cos_exps.append(exps)
            abl_cos_ores.append(ores)
            abl_cos_pes.append(pes)
            abl_cos_cardinality.append(L_cards)
            pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['abl_cos'])+f'{i}.p', 'wb'))

        # if psm_20_psro:
        #     print('P-SM 2.0 PSRO')
        #     torch_pop, exps, ores, L_cards, pes = psro_gradient(iters=iters, num_learners=num_threads, lr=LR, train_iters=train_iters, seed=i,
        #                                   div="p_dpp", sigma=2.0)
        #     psm_20_exps.append(exps)
        #     psm_20_ores.append(ores)
        #     psm_20_pes.append(pes)
        #     psm_20_cardinality.append(L_cards)
        #     pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['psm_20'])+f'{i}.p', 'wb'))

        # if psm_25_psro:
        #     print('P-SM 2.5 PSRO')
        #     torch_pop, exps, ores, L_cards, pes = psro_gradient(iters=iters, num_learners=num_threads, lr=LR, train_iters=train_iters, seed=i,
        #                                   div="p_dpp", sigma=2.5)
        #     psm_25_exps.append(exps)
        #     psm_25_ores.append(ores)
        #     psm_25_pes.append(pes)
        #     psm_25_cardinality.append(L_cards)
        #     pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['psm_25'])+f'{i}.p', 'wb'))


        # if rectified:
        #     print('Rectified PSRO')
        #     torch_pop, exps = psro_rectified_gradient(iters=iters, seed=i, train_iters=train_iters,
        #                             num_pseudo_learners=1, lr=LR)
        #     rectified_exps.append(exps)
        #     pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['rectified'])+f'{i}.p', 'wb'))

        # if pipeline_psro:
        #     print('Pipeline PSRO')
        #     torch_pop, exps, ores, L_cards, pes = psro_gradient(iters=iters, num_learners=num_threads, lr=LR, train_iters=train_iters, seed=i,
        #                                   div=None)
        #     pipeline_exps.append(exps)
        #     pipeline_ores.append(ores)
        #     pipeline_pes.append(pes)
        #     pipeline_cardinality.append(L_cards)
        #     pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['p-psro'])+f'{i}.p', 'wb'))

        if psro:
            print('PSRO')
            torch_pop, exps, ores, L_cards, pes = psro_gradient(iters=iters, num_learners=1, lr=LR, train_iters=train_iters, seed=i,
                                          div=None)
            psro_exps.append(exps)
            psro_ores.append(ores)
            psro_pes.append(pes)
            psro_cardinality.append(L_cards)
            pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['psro'])+f'{i}.p', 'wb'))

        d = {
            # 'rectified_exps':rectified_exps,
            # 'rectified_cardinality':rectified_cardinality,
            'pipeline_exps':pipeline_exps,
            'pipeline_cardinality':pipeline_cardinality,
            'pipeline_ores':pipeline_ores,
            'pipeline_pes':pipeline_pes,
            'bd_rd_exps':bd_rd_exps,
            'bd_rd_cardinality':bd_rd_cardinality,
            'bd_rd_ores':bd_rd_ores,
            'bd_rd_pes':bd_rd_pes,
            'dpp_exps':dpp_exps,
            'dpp_cardinality':dpp_cardinality,
            'dpp_ores':dpp_ores,
            'dpp_pes':dpp_pes,
            'p_dpp_exps':p_dpp_exps,
            'p_dpp_cardinality':p_dpp_cardinality,
            'p_dpp_ores':p_dpp_ores,
            'p_dpp_pes':p_dpp_pes,
            'abl_gauss_exps':abl_gauss_exps,
            'abl_gauss_cardinality':abl_gauss_cardinality,
            'abl_gauss_ores':abl_gauss_ores,
            'abl_gauss_pes':abl_gauss_pes,
            # 'psm_10_exps':psm_10_exps,
            # 'psm_10_cardinality':psm_10_cardinality,
            # 'psm_10_ores':psm_10_ores,
            # 'psm_10_pes':psm_10_pes,
            'abl_cos_exps':abl_cos_exps,
            'abl_cos_cardinality':abl_cos_cardinality,
            'abl_cos_ores':abl_cos_ores,
            'abl_cos_pes':abl_cos_pes,
            'psm_20_exps':psm_20_exps,
            'psm_20_cardinality':psm_20_cardinality,
            'psm_20_ores':psm_20_ores,
            'psm_20_pes':psm_20_pes,
            'psm_25_exps':psm_25_exps,
            'psm_25_cardinality':psm_25_cardinality,
            'psm_25_ores':psm_25_ores,
            'psm_25_pes':psm_25_pes,
            'psd_exps': psd_exps,
            'psd_cardinality':psd_cardinality,
            'psd_ores': psd_ores,
            'psd_pes': psd_pes,
            'psro_exps':psro_exps,
            'psro_cardinality':psro_cardinality,
            'psro_ores':psro_ores,
            'psro_pes':psro_pes,
         }
        pickle.dump(d, open(os.path.join(PATH_RESULTS, 'checkpoint_'+str(i)), 'wb'))


def run_traj():
    method_nums = len(FILE_TRAJ)
    titles = {
        'rectified': 'PSRO-rN',
        'dpp': 'UDM-PSRO',
        'p_dpp': 'P-UDM-PSRO',
        # 'p-psro': 'P-PSRO',
        'psro': 'PSRO',
        'psd': 'PSD-PSRO',
        'bd_rd': "BD&RD-PSRO"
    }
    methods = ['psro', 'psd', 'dpp', 'p_dpp', 'bd_rd']
    pops = {}
    fig1, axs1 = plt.subplots(1, method_nums, figsize=(5 * method_nums, 5 * 1), dpi=200)
    axs1 = axs1.flatten()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2'] #, '#7f7f7f', '#bcbd22', '#17becf']
    # for i, key in enumerate(FILE_TRAJ.keys()):
    for i, key in enumerate(methods):
        ax = axs1[i]
        d = pickle.load(open(os.path.join(PATH_RESULTS, FILE_TRAJ[key])+'0.p', 'rb'))
        pops[FILE_TRAJ[key]] = d['pop']
        pops[FILE_TRAJ[key]].visualise_pop(ax=ax, color=colors[i])
        ax.set_title(titles[key])

    fig1.tight_layout()
    fig1.savefig(os.path.join(PATH_RESULTS, 'trajectories.png'))


if __name__ =="__main__":
    # num experiments 3, iters 50
    FILE_TRAJ = {
    # 'rectified': 'rectified.p',
    # 'psro': 'psro.p',
    # 'p-psro': 'p_psro.p',
    'p_dpp': 'p_dpp.p',
    # 'dpp': 'dpp.p',
    # 'psd': 'psd.p', 
    # 'bd_rd': 'bd_rd.p',
    # 'abl_gauss': 'abl_gauss.p',
    # 'abl_cos': 'abl_cos.p',
    # 'psm_20': 'psm_20.p',
    # 'psm_25': 'psm_25.p'
             }

    run_experiments(num_experiments=10, num_threads=1, iters=50,
                    pipeline_psro=('p-psro' in FILE_TRAJ),
                    dpp_psro=('dpp' in FILE_TRAJ),
                    p_dpp_psro=('p_dpp' in FILE_TRAJ),
                    psd_psro=('psd' in FILE_TRAJ),
                    rectified=('rectified' in FILE_TRAJ),
                    psro=('psro' in FILE_TRAJ),
                    bd_rd = ('bd_rd' in FILE_TRAJ),
                    abl_gauss_psro=('abl_gauss' in FILE_TRAJ),
                    abl_cos_psro=('abl_cos' in FILE_TRAJ),
                    psm_20_psro=('psm_20' in FILE_TRAJ),
                    psm_25_psro=('psm_25' in FILE_TRAJ),
                    yscale='none', train_iters=TRAIN_ITERS)
    # run_traj()

    # plt.show()