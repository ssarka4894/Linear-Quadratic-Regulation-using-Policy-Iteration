import numpy as np
import matplotlib.pyplot as plt

class Convergence:
    def __init__(self, iter_duraton, gain_diff, spec_radius, discount):
        self.iter_duraton = iter_duraton
        self.gain_diff = gain_diff
        self.spec_radius = spec_radius
        self.discount = discount
        
    def compute_performance(self):
        
        params = {'axes.labelsize': 30,
                  'axes.titlesize': 20,
                  'xtick.labelsize':20,
                  'ytick.labelsize':20,
                  'legend.fontsize':20}
        plt.rcParams.update(params)
        
        # ||K - K*|| vs iteration plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.subplots()
        line_gt,=ax.semilogy(self.iter_duraton ,self.gain_diff, color = [0.6350, 0.0780, 0.1840], linewidth=4, markersize=10)
        ax.set_ylabel(r'$Controller \quad Gain \quad : ||K_i - K^\star||$',fontsize=16)
        ax.set_xlabel('Number of Policy Iteration Steps')
        ax.set_facecolor('#E6E6E6')
        ax.grid()
        fig.suptitle('Controller Gain Difference (Policy Iteration Performance)', fontsize=30)
        plt.show()
        
        
        # rho(A - BK) vs iteration plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.subplots()
        line_gt,=ax.plot(self.iter_duraton,self.spec_radius, color = [0.6350, 0.0780, 0.1840], linewidth=4, markersize=10)
        line_stable, = ax.plot(self.iter_duraton,np.ones_like(self.iter_duraton),'k:')
        ax.set_ylabel(r'$Spectral \quad Radius \quad : \rho(A-BK_i)$',fontsize=16)
        ax.set_xlabel('Number of Policy Iteration Steps')
        ax.set_facecolor('#E6E6E6')
        ax.grid()
        fig.suptitle('Spectral Radius (Policy Iteration Performance)', fontsize=30)
        plt.show()
        
        
        
        # \gamma vs iteration plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.subplots()
        line_gt,=ax.plot(self.iter_duraton,self.discount, color = [0.6350, 0.0780, 0.1840], linewidth=4, markersize=10)
        ax.set_ylabel(r'Gamma: $\gamma_i$',fontsize=16)
        ax.set_xlabel('Number of Policy Iteration Steps')
        ax.set_facecolor('#E6E6E6')
        ax.grid()
        fig.suptitle('Discount Factor (Policy Iteration Performance)', fontsize=30)
        plt.show()
        
        
        
    