import numpy as np
import numpy.random as rnd
import scipy.linalg as linal
from LSPI_LinearQuadraticRegulation import LSPI_inf_horizon
from plot_Performance import Convergence
from plot_trajectories import Trajectories


# Given System
A = np.array([[0.9974,0.0539],
              [-0.1078,1.1591]])
B = np.array([[0.0013],
              [0.0539]])
Q = np.array([[0.25,0.0],
              [0.0,0.05]])
R = np.array([[0.05]])


n,p = B.shape

lspi_object = LSPI_inf_horizon(n,degree = 2)

NumEpisodes = 10
state_bound = 100
x = lspi_object.random_state_generator(n,state_bound)
x_i = x


final_time = 100
Gain = np.zeros((p,n))
gamma = 0.


Optimal_QValue = linal.solve_discrete_are(A,B,Q,R)
Gain_opt = -linal.solve(R+B.T@Optimal_QValue@B,B.T@Optimal_QValue@A)

Gain_error = 10

ep =0

X = []
U = []
X_next = []

c = []

for t in range(final_time):
    u = Gain@x + .1 * rnd.randn(p)
    
    x_next = A@x + B @ u 
    
    c.append(x@Q@x + u@R@u)
    X.append(x)
    
    U.append(u)
    
    
    X_next.append(x_next)
    
    if linal.norm(x_next) <= state_bound:
        x = x_next
    else:
        x = lspi_object.random_state_generator(n,state_bound)
        

gain_diff = []
discount = []
spec_radius = []

while Gain_error > 1e-6 :
    ep += 1
    eigMax = np.max(np.abs(linal.eigvals(A+B@Gain)))
    Gain_error = linal.norm(Gain-Gain_opt)
    gain_diff.append(Gain_error)
    discount.append(gamma)
    spec_radius.append(eigMax)
    print('Episode: ',ep,', ||K-K*||:',Gain_error, ', rho(A+BK):', eigMax,', gamma:', gamma)
    gamma,Q_mat = lspi_object.check_bounds(X,U,X_next,c,Gain,gamma,limit=1e4)
    Gain = lspi_object.compute_gain(Q_mat,n)

    
print('Optimal Controller Gains: ', Gain_opt)
print('LSPI policy iteraton obtained Controller Gains: ', Gain)


iter_duration = np.arange(1,len(spec_radius)+1)
plotConvergence = Convergence(iter_duration, gain_diff, spec_radius, discount)
plotConvergence.compute_performance()

X = []
U = []
U1 = []
X_next = []
X_next1 = []
c = []
for t in range(final_time):
    u = Gain_opt@x 
    u1 = Gain@x
    x_next = A@x + B @ u 
    x_next1 = A@x + B @ u 
    c.append(x@Q@x + u@R@u)
    X.append(x)
    U.append(u)
    U1.append(u1)
    X_next.append(x_next)
    X_next1.append(x_next1)
    if linal.norm(x_next) <= state_bound:
        x = x_next
    else:
        x = lspi_object.random_state_generator(n,state_bound)
        

Uo = U
X_nexto = X_next

Uc = U1
X_nextc = X_next1



plotComparisons = Trajectories(final_time, Uo,Uc, X_nexto, X_nextc)
plotComparisons.compute_comparisons()
