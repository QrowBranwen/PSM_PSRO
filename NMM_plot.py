import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import os
from scipy import stats
import pickle
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

from scipy.spatial.distance import pdist, squareform

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
np.random.seed(123)

from scipy.spatial.distance import squareform, pdist

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

    def get_br_to_strat(self, metanash, lr, nb_iters=20, BR=None):
        if BR is None:
            br = GMMAgent(self.mus)
            br.x = nn.Parameter(0.1*torch.randn(2, dtype=torch.float), requires_grad=False)
            br.x.requires_grad = True
        else:
            br =  BR
        optimiser = optim.Adam(br.parameters(), lr=lr)
        for _ in range(nb_iters*20):
            loss = -self.get_payoff_aggregate(br, metanash, self.pop_size,)
            # Optimise !
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

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

def plot_nmm_traj():
    exp_data = {
        "psro": (0.0493, 0.0125),
        "p_dpp": (0.0283, 0.0091),
        "dpp": (0.0389, 0.0114),
        "psd": (0.0306, 0.008),
        "bd_rd": (0.0404, 0.012),
        "abl_gauss": (0.0459, 0.0088),
        "abl_cos": (0.0311, 0.01),
    }
    methods = ['psro', 'bd_rd', 'dpp', 'psd','p_dpp',]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd',  'tab:red', '#252d2e', '#7f7f7f', '#bcbd22', '#17becf']
    # methods = [ 'abl_gauss', 'abl_cos', 'p_dpp']
    # colors = ['#bcbd22', '#17becf', 'tab:red',  '#bcbd22', '#17becf']

    pops = {}
    method_nums = len(methods)
    fig1, axs1 = plt.subplots(1, method_nums, figsize=(5 * method_nums, 5 * 1), dpi=200)
    axs1 = axs1.flatten()
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd',  'tab:red', '#252d2e', '#7f7f7f', '#bcbd22', '#17becf']
    
    # for i, key in enumerate(FILE_TRAJ.keys()):
    for i, key in enumerate(methods):
        ax = axs1[i]
        d = pickle.load(open(os.path.join(PATH_RESULTS, FILE_TRAJ[key])+'0.p', 'rb'))
        pops[FILE_TRAJ[key]] = d['pop']
        pops[FILE_TRAJ[key]].visualise_pop(ax=ax, color=colors[i])
        ax.set_title(titles[key],size=20)
        # ax.set_xlabel(xlabels[i], size=20)
        # exp_value = f"Exp: {d['exp_mean']:.2f} ± {d['exp_std']:.2f}"  # 根据数据内容
        exp_mean, exp_std = exp_data[key]
        exp_mean, exp_std = exp_mean * 100, exp_std * 100
        if key == 'p_dpp':
            exp_value = r"Exp: $\mathbf{" + f"{exp_mean:.2f}" + r"} \pm \mathbf{" + f"{exp_std:.2f}" + r"}$"
        else:
            exp_value = f"Exp: {exp_mean:.2f} ± {exp_std:.2f}"
        ax.text(0.5, -0.15, exp_value, size=22, ha='center', transform=ax.transAxes)


    fig1.tight_layout()
    fig1.savefig(os.path.join(PATH_RESULTS, 'NMM_traj.pdf'),dpi=300,bbox_inches='tight')

def plot_sim_curve(measure='ores'):
    methods = ['psro', 'bd_rd', 'dpp', 'psd', 'p_dpp', 'abl_gauss', 'abl_cos']
    sim_values = {method: [] for method in methods}  # 初始化字典存储每个方法的 ores 数据
    udm_values = {method: [] for method in methods}
    filepath = os.path.join(PATH_RESULTS, f'checkpoint_9')
    with open(filepath, 'rb') as f:
        d = pickle.load(f)
    for key in methods:
        all_ores_data = d[f'{key}_{measure}']  # 获取所有组的 ores 数据
        for ores_data in all_ores_data:
            tmp_psm = []
            tmp_udm = []
            for M in ores_data:
                mu = np.mean(M)
                sigma = np.std(M)
                M = (M - mu) / sigma  # 归一化
                L = M @ M.T  # Compute kernel
                udm = np.linalg.eigvals((L + 1) ** 3)
                L_card = np.sum(1 / (1 + np.exp(-udm)) - 0.5).real
                tmp_udm.append(L_card)
                
                pairwise_dists = squareform(pdist(M, metric='euclidean') ** 2)
                sigma = 5 #3*np.std(M)
                L = np.exp(-pairwise_dists / (2 * sigma ** 2))
                row_norms = np.linalg.norm(M, axis=1, keepdims=True)
                M = M / row_norms
                cosine_similarity = M @ M.T
                # L = spearman_correlation_numpy(M)
                sim_L =  L[-1, :-1] * cosine_similarity[-1, :-1]
                tmp_psm.append(np.sum(sim_L))

            sim_values[key].append(np.array(tmp_psm))
            udm_values[key].append(np.array(tmp_udm))
    
    def plot_cure(dataset, measure='ores', color=None):
        plt.figure(figsize=(10, 6))
        for method in methods:
            data = np.array(dataset[method])  # Convert to NumPy array
            mean_values = data.mean(axis=0)  # Compute mean
            std_values = data.std(axis=0)    # Compute standard deviation

            # Plot mean values and error bars
            x = np.arange(len(mean_values))  # Assume x-axis is iteration index
            plt.plot(x, mean_values, label=f"{titles[method]}")
            plt.fill_between(x, mean_values - std_values, mean_values + std_values, alpha=0.2)

        # Beautify plot
        plt.xlabel("Iterations", fontsize=14)
        
        if measure == 'ores':
            plt.ylabel("PSM", fontsize=14)
        else:
            plt.ylabel("UDM", fontsize=14)
        
        # plt.title("Ores Values Across Methods with Error Bars", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        if measure == 'ores':
            plt.savefig(os.path.join(PATH_RESULTS, 'NMM_PSM.pdf'),dpi=300,bbox_inches='tight')
        else:
            plt.savefig(os.path.join(PATH_RESULTS, 'NMM_UDM.pdf'),dpi=300,bbox_inches='tight')
    plot_cure(sim_values, measure='ores')
    plot_cure(udm_values, measure='udm')


def load_and_plot_heatmaps():
    import seaborn as sns
    # methods = ['psro', 'bd_rd', 'dpp', 'psd', 'p_dpp']  # Methods to process
    methods = ['abl_gauss', 'abl_cos', 'p_dpp',]  # Methods to process
    filepath = os.path.join(PATH_RESULTS, f'checkpoint_9')
    # Load data from the file
    with open(filepath, 'rb') as f:
        d = pickle.load(f)

    # Create subplots for heatmaps
    fig, axs = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5), dpi=200)

    cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.8])  # Colorbar position: [left, bottom, width, height]
    
    for i, method in enumerate(methods):
        M = d[f'{method}_ores'][0][-1]  # Extract PES matrix for the method
        M = (M - np.mean(M)) / np.std(M)
        pairwise_dists = squareform(pdist(M, metric='euclidean') ** 2)
        L = np.exp(-pairwise_dists / (2 * 2 ** 2))
        row_norms = np.linalg.norm(M, axis=1, keepdims=True)
        M = M / row_norms
        cosine_similarity = M @ M.T
        L = L * cosine_similarity
        # Plot heatmap for the method, share the colorbar
        sns.heatmap(
            L, 
            ax=axs[i], 
            cmap='coolwarm', 
            cbar=i == 0,  # Only add the colorbar for the first heatmap
            cbar_ax=None if i != 0 else cbar_ax, 
            annot=False
        )
        axs[i].set_title(titles[method], fontsize=16)
        axs[i].set_aspect('equal')  # Ensure consistent size for each heatmap
    
    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 0.915, 1])

    # Adjust layout and display the heatmaps
    # plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(PATH_RESULTS, 'NMM_heatmap_abl.pdf'),dpi=300,bbox_inches='tight')


PATH_RESULTS="results/NMM_gc"
FILE_TRAJ = {
    # 'rectified': 'rectified.p',
    'psro': 'psro.p',
    # 'p-psro': 'p_psro.p',
    'p_dpp': 'p_dpp.p',
    'dpp': 'dpp.p',
    'psd': 'psd.p', 
    'bd_rd': 'bd_rd.p',
    'abl_gauss': 'abl_gauss.p',
    # 'psm_10': 'psm_10.p',
    'abl_cos': 'abl_cos.p',
    # 'psm_20': 'psm_20.p',
    # 'psm_25': 'psm_25.p'
    }


method_nums = 5
titles = {
    # 'rectified': 'PSRO-rN',
    # 'p-psro': 'P-PSRO',
    'psro': 'PSRO',
    'bd_rd': "BD&RD-PSRO",
    'dpp': 'UDM-PSRO',
    'psd': 'PSD-PSRO',
    'p_dpp': 'PSM-PSRO (Ours)',
    'abl_gauss': 'PSM-PSRO w/o. Cosine',
    # 'psm_10': 'PSM-PSRO (1.0)',
    'abl_cos': 'PSM-PSRO w/o. Gaussian',
    # 'psm_20': 'PSM-PSRO (2.0)',
    # 'psm_25': 'PSM-PSRO (2.5)'
}

plot_nmm_traj()
plot_sim_curve(measure='ores')
load_and_plot_heatmaps()