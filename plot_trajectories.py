import numpy as np
import matplotlib.pyplot as plt

class Trajectories:
    def __init__(self, Horizon, Uo,Uc, X_nexto, X_nextc):
        self.Horizon = Horizon
        self.Uo = Uo
        self.Uc = Uc
        self.X_nexto = np.array(X_nexto)
        self.X_nextc = np.array(X_nextc)
    
        
    def compute_comparisons(self):
        
        params = {'axes.labelsize': 30,
                  'axes.titlesize': 20,
                  'xtick.labelsize':20,
                  'ytick.labelsize':20,
                  'legend.fontsize':20}
        plt.rcParams.update(params)
        
        
        # System Controls Comparison
        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots()
        line_gt,=ax.plot(np.linspace(1,self.Horizon,self.Horizon),self.Uo, color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
        line_pdp,= ax.plot(np.linspace(1,self.Horizon,self.Horizon),self.Uc, color ='#A2142F', linewidth=5)
        ax.set_ylabel('System Controls')
        ax.set_xlabel('Time')
        ax.set_facecolor('#E6E6E6')
        ax.grid()
        fig.suptitle('LSPI obtained Controller vs DARE Optimal Controller', fontsize=30)
        plt.legend([line_gt, line_pdp], ['Optimal Controller - u* (DARE)', 'Obtained Controller - u '], facecolor='white', framealpha=0.5,
                    loc='best')
        plt.show()
        
        # Sustem States
        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots()
        line_qt,=ax.plot(np.linspace(1,self.Horizon,self.Horizon),self.X_nexto[:,0], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
        line_q_pdp,=ax.plot(np.linspace(1,self.Horizon,self.Horizon),self.X_nextc[:,0], color ='#2A596C', linewidth=5)
        line_dqt,=ax.plot(np.linspace(1,self.Horizon,self.Horizon),self.X_nexto[:,1], color ='#9784A4', linewidth=10, linestyle='dashed', alpha=0.7)
        line_dq_pdp,=ax.plot(np.linspace(1,self.Horizon,self.Horizon),self.X_nextc[:,1], color ='#704684', linewidth=5)
        ax.set_ylabel('System States')
        ax.set_xlabel('Time')
        ax.set_facecolor('#E6E6E6')
        ax.grid()
        fig.suptitle('LSPI obtained Controlled States vs DARE Optimally Controlled States', fontsize=30)
        plt.legend([line_qt, line_q_pdp, line_dqt, line_dq_pdp], 
                    [ 'Optimal ($ x_1$)', 'PI ($x_1$)' , 'Optimal ($ x_2$)', 'PI ($ x_2$)'], facecolor='white', framealpha=0.5, fontsize=10, loc='best')
        plt.show()
        
        

        
        