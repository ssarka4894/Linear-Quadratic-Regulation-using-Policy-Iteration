import numpy as np
import numpy.random as rnd
import scipy.linalg as linal
from sklearn.preprocessing import PolynomialFeatures


class LSPI_inf_horizon:
    def __init__(self, row = None, degree = None):
        self.row = row
        self.degree = degree

    def extractQ(self,weights):
        elem = weights[1:]
        Q = np.array([[elem[0], 0.5*elem[1], 0.5*elem[2]],
                      [0.5*elem[1], elem[3], 0.5*elem[4]],
                      [0.5*elem[2], 0.5*elem[4], elem[5]]], dtype = float)
        return Q

    def compute_basis(self,X,U,X_next,K,discount):
        Basis = []
        for x,u,x_next in zip(X,U,X_next):
            sa_pair = np.hstack([x,u])
            sa_pair_next = np.hstack([x_next,K@x_next])
            poly = PolynomialFeatures(self.degree)
            index = np.array([1,2,3])
            phi = poly.fit_transform([sa_pair])[0]
            phi = np.delete(phi,index)
            phi_next = poly.fit_transform([sa_pair_next])[0]
            phi_next = np.delete(phi_next,index)
            basis = phi - discount * phi_next
            Basis.append(basis)
        return np.array(Basis)
        
        
        
    def compute_value_and_cost(self,X,U,X_next,c,K,discount):
        quad_state = np.vstack([np.eye(self.row),K])
        Basis = self.compute_basis(X,U,X_next,K,discount)
        weights = linal.lstsq(Basis,c)[0]
        Weight_matrix = self.extractQ(weights)
        Qvalue = quad_state.T @ Weight_matrix @ quad_state
        return Qvalue,Weight_matrix
        
    def check_bounds(self,X,U,X_next,c,K,discount,limit=1):
        
        lower_bound = discount 
        upper_bound = 1.
        
        
        # Get the current P value
        Qvalue_cur,Weight_matrix = self.compute_value_and_cost(X,U,X_next,c,K,discount)
            
        # Try 1 first
        discount = 1.
        Qvalue_new,Weight_matrix = self.compute_value_and_cost(X,U,X_next,c,K,discount)
        

        
        Qvalue_change = linal.norm(Qvalue_new-Qvalue_cur)
        spec_radius = linal.eigvalsh(Qvalue_new,eigvals=[0,0])[0]
        
        if spec_radius > 0:
            return discount, Weight_matrix
        
        
        while ((upper_bound-lower_bound) > 1e-6) or (spec_radius < 0) or (Qvalue_change > limit):
            
            Qvalue_new,Weight_matrix = self.compute_value_and_cost(X,U,X_next,c,K,discount)
            spec_radius = linal.eigvalsh(Qvalue_new,eigvals=[0,0])[0]
            
            Qvalue_change = linal.norm(Qvalue_cur-Qvalue_new)
      
            if (spec_radius < 0) or (Qvalue_change > limit):
                upper_bound = discount
                
            else:
                lower_bound = discount
                
            discount = .5 * (lower_bound+upper_bound)
            
                
            
        return discount, Weight_matrix

    def compute_gain(self,Weight_matrix,row):
        Omega = Weight_matrix[row:,row:]
        Phi = Weight_matrix[row:,:row]
        K = -linal.solve(Omega,Phi)
        return K
        
    def random_state_generator(self,row,b):
        r = rnd.rand() * b
        state = rnd.randn(row)
        return r * state / linal.norm(state)

   
    
    



