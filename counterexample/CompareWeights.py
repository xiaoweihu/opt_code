"""
Coefficients of x_t: 
coeff_const(epsilon), coeff_x_1, coeff_delta2
coeff_xi: [xi_1, ..., ..., xi_T]

Given epsilon, eta, C_1
g_t = -epsilon + epsilon * x_t + C_1 * delta2 + xi_t
x_{t+1} = x_t - eta * g_t = (1 - eta*epsilon) * x_t + eta*epsilon - eta*C_1*delta2 - eta*xi_t

Using window average, given K:
bar_g_t = 1/(K+1) * (g_t + g_{t-1} + ... + g_{t-K})
x_t = x_{t-1} - eta * bar_g_t
"""

from collections import deque

C_1 = 1.
C_2 = 1.
T = 100000
epsilon = 1.5*pow(C_1*C_2*C_2/16., 1./3.)*pow(float(T), -1./3.)
eta = 0.25*pow(float(T), -5./8.)

K = int(pow(float(T), 1./8.))

coeff_const = 0.
coeff_x_1 = 1.
coeff_delta2 = 0.
coeff_xi = [0.] * T

bar_g_const = 0.
bar_g_x_1 = 0.
bar_g_delta2 = 0.
bar_g_xi = [0.] * T

stored_g = deque()

def init():
    global coeff_const, coeff_x_1, coeff_delta2, coeff_xi, bar_g_const, bar_g_x_1, bar_g_delta2, bar_g_xi, stored_g
    coeff_const = 0.
    coeff_x_1 = 1.
    coeff_delta2 = 0.
    coeff_xi = [0.] * T
    
    bar_g_const = 0.
    bar_g_x_1 = 0.
    bar_g_delta2 = 0.
    bar_g_xi = [0.] * T
    
    stored_g = deque()
    
    

def alg(t):
    global coeff_const, coeff_x_1, coeff_delta2, coeff_xi
    coeff_const *= 1 - eta*epsilon
    coeff_x_1 *= 1 - eta*epsilon
    coeff_delta2 *= 1 - eta*epsilon
    coeff_xi = [(1 - eta*epsilon) * xi for xi in coeff_xi]
    coeff_const += eta*epsilon
    coeff_delta2 -= eta*C_1
    coeff_xi[t] -= eta
    return
    
    
def algWA(t):
    global coeff_const, coeff_x_1, coeff_delta2, coeff_xi, bar_g_const, bar_g_x_1, bar_g_delta2, bar_g_xi, stored_g
    # compute g_t
    g_coeff_const = epsilon * coeff_const
    g_coeff_x_1 = epsilon * coeff_x_1
    g_coeff_delta2 = epsilon * coeff_delta2
    g_coeff_xi = [epsilon * xi for xi in coeff_xi]
    g_coeff_const -= epsilon
    g_coeff_delta2 += C_1
    g_coeff_xi[t] += 1
    
    if len(stored_g) >= K+1:
        g = stored_g.popleft()
        bar_g_const -= g[0]/float(K+1.)
        bar_g_x_1 -= g[1]/float(K+1.)
        bar_g_delta2 -= g[2]/float(K+1.)
        bar_g_xi = [bar_g_xi[i] - g[3+i]/float(K+1.) for i in range(T)]
    stored_g.append([g_coeff_const, g_coeff_x_1, g_coeff_delta2] + g_coeff_xi)
    bar_g_const += g_coeff_const/float(K+1.)
    bar_g_x_1 += g_coeff_x_1/float(K+1.)
    bar_g_delta2 += g_coeff_delta2/float(K+1.)
    bar_g_xi = [bar_g_xi[i] + g_coeff_xi[i]/float(K+1.) for i in range(T)]
    coeff_const -= eta*bar_g_const
    coeff_x_1 -= eta*bar_g_x_1
    coeff_delta2 -= eta*bar_g_delta2
    coeff_xi = [coeff_xi[i] - eta*bar_g_xi[i] for i in range(T)]
    return
    
    
    
def writeCoeff():
    init()
    for t in range(T):
        alg(t)
    with open('algWeights', 'a') as fout:
        fout.write(repr(T)+' '+repr(coeff_const)+' '+repr(coeff_x_1)+' '+repr(coeff_delta2))
        for i in range(T):
            fout.write(' '+repr(coeff_xi[i]))
        fout.write('\n')
        
    init()
    for t in range(T):
        algWA(t)
    with open('algWAWeights', 'a') as fout:
        fout.write(repr(T)+' '+repr(coeff_const)+' '+repr(coeff_x_1)+' '+repr(coeff_delta2))
        for i in range(T):
            fout.write(' '+repr(coeff_xi[i]))
        fout.write('\n')
    
    
        
    
if __name__=="__main__":
    writeCoeff()

    