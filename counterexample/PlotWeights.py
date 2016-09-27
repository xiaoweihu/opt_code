import sys
from numpy import *
from pylab import *
import matplotlib.pyplot as plt

def plotWeights():
    data1 = []
    data2 = []
    line = 3
    with open("./algWeights", 'r') as fin1:
        for i in range(line):
            data1 = fin1.readline()
    with open("./algWAWeights", 'r') as fin2:
        for i in range(line):
            data2 = fin2.readline()    
    
    data1 = data1.split(' ')
    data2 = data2.split(' ')
    print len(data1), len(data2)
    T = int(data1[0])
    print T
    const1 = float(data1[1])
    x1_1 = float(data1[2])
    delta2_1 = float(data1[3])
    xi_1 = [float(xi)*T for xi in data1[4:]]
    print len(xi_1)
    const2 = float(data2[1])
    x1_2 = float(data2[2])
    delta2_2 = float(data2[3])
    xi_2 = [float(xi)*T for xi in data2[4:]]
    print len(xi_2)
    roundNum = [i for i in range(1, 1+T)]
    
    fig, ax = plt.subplots()
    
    ax.plot(roundNum, xi_1, label = "alg")
    ax.plot(roundNum, xi_2, label = "alg with WA")
    ax.legend(loc=0)
    ax.set_xlabel('round number T')
    ax.set_ylabel('coefficients of xi')
    
    show()
    
        
    
if __name__=="__main__":
    plotWeights()

    