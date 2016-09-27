import sys
from numpy import *
from pylab import *
import matplotlib.pyplot as plt

def findPara():
    data = loadtxt("exprmResults")
    numRounds = data.T[0]
    eta = data.T[1]
    delta = data.T[2]
    avgRegret = data.T[4]
    baseline = [pow(t,-0.6) for t in numRounds]
    
    fig, ax = plt.subplots()
    
    ax.plot(log10(numRounds), log10(eta), label="eta (log)")
    #ax.plot(log10(numRounds), log10(delta), label="delta (log)")
    #ax.plot(numRounds, avgRegret, label="average regret")
    ax.plot(log10(numRounds), log10(baseline))
    ax.legend(loc=0)
    ax.set_xlabel('Round number (log T)')
    #ax.set_ylabel('Best fixed eta & delta')
    #ax.set_title("Average regret against optimal fixed eta & delta")

    show()

def plotOptPara():
    predictList = loadtxt("predictX11")
    avgRegList = loadtxt("regretX11")
    T = 1000

    expPredict = predictList.mean(axis=0)
    stdPredict = predictList.std(axis=0)
    expRegret = avgRegList.mean(axis=0)
    stdRegret = avgRegList.std(axis=0)
    roundNum = linspace(1, T, T)
    baseline1 = [pow(t, -0.5) for t in roundNum]
    baseline2 = zeros(T)
    
    fig, ax = plt.subplots()
    
    #ax.errorbar(roundNum, expPredict, yerr=stdPredict, fmt='.', label="Predict")
    #ax.errorbar(roundNum, expRegret, yerr=stdRegret, fmt='.', label="Average Regret")
    #ax.plot(roundNum, expRegret, label = "Average Regret")
    ax.plot(roundNum, expPredict, label = "Predict")
    #ax.plot(roundNum, baseline1, label="1/sqrt(t)")
    ax.plot(roundNum, baseline2, label = "0")
    ax.legend(loc=0)
    ax.set_xlabel('round number (t)')
    ax.set_title('Expected Predict Over 1000 Runs: x^11')
    
    show()

def plotFunction():
    xx = [0.0001*i for i in range(-10000, 10001)]
    #yy1 = [pow(abs(x), 11) for x in xx]
    #yy2 = [pow(abs(x), 10) for x in xx]
    #yy3 = [pow(abs(x), 3) for x in xx]
    #yy4 = [pow(abs(x), 2) for x in xx]
    k = [(3828825.*r**9 - 9069060.*r**7 + 12170070.*r**5 - 8450820.*r**3 + 1924825.*r)/16384. for r in xx]

    fig, ax = plt.subplots()
    #ax.plot(xx, yy1, label = "|x|^11")
    #ax.plot(xx, yy2, label = "|x|^10")
    #ax.plot(xx, yy3, label = "|x|^3")
    #ax.plot(xx, yy4, label = "|x|^2")
    ax.plot(xx, k, label = "|x|^2")
    ax.legend(loc=0)

    show()

def plotFunction1():
    n = 1000000.
    #xx = [pow(n, -1./6.)/10000*i for i in range(-100, 100)]
    xx = [float(i)/1000. for i in range(-1000, 2000)]
    yy = [-pow(x,3)+x for x in xx]
    #yy = [0.25*pow(n,-1./2.)/x + 0.5*x*x + 0.25*pow(x,5)*pow(n,0.5) for x in xx]
    #derivative = [-0.25*pow(n,-1./2.)*pow(x, -2.) + x + 5./4.*pow(n,0.5)*pow(x,4) for x in xx]

    fig, ax = plt.subplots()
    ax.plot(xx, yy, label="y")
    #ax.plot(xx, derivative, label="y\'")
    ax.legend(loc=0)
    
    show()

def plotRegret():
    data1 = loadtxt("./regretHSx3")
    data2 = loadtxt("./regretHSNew")
    data3 = loadtxt("./regretHSepsilonX2")
    roundNum = data1.T[0]
    expRegret1 = data1.T[1]
    expRegret2 = data2.T[1][:len(roundNum)]
    expRegret3 = data3.T[1][:len(roundNum)]
    #expPredict = data.T[2]
    baseline1 = [pow(t, -1.0/3.0) for t in roundNum]
    baseline2 = [pow(t, -1.0/2.0) for t in roundNum]

    fig, ax = plt.subplots()
    
    #ax.plot(log10(roundNum), log10(abs(expPredict)), label = )
    ax.plot(log10(roundNum), log10(expRegret1), label = "|x-1|^3")
    ax.plot(log10(roundNum), log10(expRegret2), label = "new loss")
    ax.plot(log10(roundNum), log10(expRegret3), label = "epsilon*(x-1)^2")
    ax.plot(log10(roundNum), log10(baseline1), label = "T^{-1/3}")
    ax.plot(log10(roundNum), log10(baseline2), label = "T^{-1/2}")
    #ax.set_title('loss: |x|^11, for each T, OGD eta=0.25/t delta=3/16*(1+logT/2)/T init=0')
    ax.set_title('highly smooth algorithm, for each T, init=0, epsilon = T^{-1/3}')
    ax.legend(loc=0)
    ax.set_xlabel('round number T (log)')
    ax.set_ylabel('Average Regret (log)')
    
    #savefig("regret_x11_1000runs.pdf")
    show()


    
if __name__ == '__main__':
    plotFunction1()
