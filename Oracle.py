'''
alg: moving window average, improving bias-variance tradeoff

f = epsilon/2*(y-1)^2
The biased, noisy oracle gives out gradient estimate
g = epsilon*(y-1)-min(epsilon, C_1 delta^2) + xi

'''

import sys
from pylab import *
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

numRuns = 1000
dimension = 1.

# will change later 
C_1 = 1.
C_2 = 1.
epsilon = .01
k = 10

gradientList = []
xList = [0.]
noise = []


def environment(y, t):
    global epsilon
    #return epsilon / 2. * pow(y-1.,2) + noise[t]
    return epsilon * (y-1.) + 2*epsilon*epsilon*log(1.+exp(-(y-1.)/epsilon)) + noise[t]
    
    
def oracle(x, delta):
    global epsilon, C_1, C_2
    if x < 0:
        return epsilon * (1.-exp(-(x-1.)/epsilon)) / (1.+exp(-(x-1.)/epsilon)) + min(epsilon, C_1*delta*delta) + random.choice([-1.,1.])*sqrt(C_2)/delta
    else:
        return min( epsilon * (1.-exp(-(x-1.)/epsilon)) / (1.+exp(-(x-1.)/epsilon)) + min(epsilon, C_1*delta*delta) , epsilon * (1.-exp(-(x+1.)/epsilon)) / (1.+exp(-(x+1.)/epsilon)) - min(epsilon, C_1*delta*delta) )  + random.choice([-1.,1.])*sqrt(C_2)/delta




def algAverage(t, eta, delta):
    global gradientList, xList, k, epsilon, C_1, C_2
    u = random.choice([-1.,1.])
    predict = xList[t] + delta*float(u)
    loss = environment(predict, t)
    
    #gradEst = epsilon * (predict - 1.) - min(epsilon, C_1*delta*delta) + random.choice([-1.,1.])*sqrt(C_2)/delta
    gradEst = oracle(xList[t], delta)
    gradientList.append(gradEst)
    
    averageGrad = 0.
    for i in range(k+1):
        if i>t: break
        averageGrad += gradientList[-1-i]
    averageGrad /= float(k)+1.
    
    update = xList[t] - eta*averageGrad
    xList.append(update)
    return loss, predict

    
def alg(t, eta, delta):
    global gradientList, xList, k, epsilon, C_1, C_2
    u = random.choice([-1.,1.])
    predict = xList[t] + delta*float(u)
    loss = environment(predict, t)
    
    #gradEst = epsilon * (predict - 1.) - min(epsilon, C_1*delta*delta) + random.choice([-1.,1.])*sqrt(C_2)/delta
    gradEst = oracle(xList[t], delta)
    gradientList.append(gradEst)
    
    update = xList[t] - eta*gradEst
    xList.append(update)
    return loss, predict



def writeRegret():
    global gradientList, xList, noise, C_1, C_2, epsilon, k

    for T in [100*i for i in [1, 4, 7, 10, 40, 70, 100, 400, 700]]:
        k = int(pow(T, 1./8.))
        delta = 0.6*pow(T, -3./16.)
        eta = 0.25*pow(T, -5./8.)
        C_1 = 1.
        C_2 = 1.
        epsilon = 1.5*pow(C_1*C_2*C_2/16., 1./3.)*pow(T, -1./3.)
        #epsilon = 1.5*pow(C_1*C_2*C_2/16., 1./3.)*0.1
        
        avgRegret = [0.]*numRuns
        finalPredict = [0.]*numRuns
        for run in range(numRuns):
            noise = random.normal(0,1, T)
            gradientList = []
            xList = [0.]
            for t in range(T):
                loss, predict = algAverage(t, eta, delta)
                regret = loss - noise[t]
                avgRegret[run] += regret
            finalPredict[run] = predict
            if (run+1)%500 == 0:
                print "run: ", run, "T: ", T
                
        avgRegret = [sumRegret/float(T) for sumRegret in avgRegret]
        fout = open('AverageAlgOracleChangingEpsilon', 'a')
        fout.write(repr(T)+' '+repr(mean(avgRegret))+' '+repr(std(avgRegret))+' '+repr(mean(finalPredict))+' '+repr(std(finalPredict))+'\n')
        fout.close()



#run()
writeRegret()




