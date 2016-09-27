#In 1-dimension case, SPSA is equivalent to Flaxman's gradient estimator
#eta: learning rate; delta: step size

import sys
from pylab import *
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

numRuns = 1000
numRounds = 10000
d=1 #dimension
range1 = -1. #range of the constrained set
range2 = 1.

sumRegret = 0
expRegret = [0]*numRounds
queryY = 0.
predict = 0.
u = 1
gradEst = 0
noise = random.normal(0,1, numRounds)


def project(x):
    global range1, range2
    if x < range1:
        x = range1
    elif x > range2:
        x = range2
    else:
        x = x
    return x

def environment(y, t):
    return pow(abs(y),10) + noise[t] + 100.

def alg(t, eta, delta):
    global predict, u, gradEst, queryY
    if t%2 == 0 :
        u = random.choice([-1,1])
        predict = queryY + delta*u
        loss = environment(predict, t)
        gradEst = float(d)/delta * loss * float(u)
    else:
        predict = queryY - delta*u
        loss = environment(predict, t)
        gradEst = gradEst - float(d)/delta * loss * float(u)
        queryY = queryY - eta*gradEst
        queryY = project(queryY)
    return loss

def run():
    global sumRegret, noise, expRegret, predict, queryY
    for i in range(numRuns):
        sumRegret = 0
        predict = 0   #initialization
        noise = random.normal(0,1, numRounds)
        for t in range(numRounds):
            loss = alg(t, 0.001, 0.15)
            sumRegret += loss - environment(0, t)     #compete with the hindsight optimum
            expRegret[t] += sumRegret
            if (t+1)%20000 == 0 :
                print "Round: ", t, "  Predict: ", predict, "  Loss: ", loss, "  G: ", gradEst
    expRegret = [sum/numRuns for sum in expRegret]


def plot():
    logRound = log10(range(1, numRounds+1))
    logRegret = log10(expRegret)
    #Round = range(1, numRounds+1)
    #baseline = [pow(i,0.5) for i in range(1, numRounds+1)]
    fig, ax = plt.subplots()
    ax.plot(logRound, logRegret)
    ax.plot(logRound, 0.5*logRound)
    #ax.plot(Round, sumRegret)
    #ax.plot(Round, baseline)
    ax.set_xlabel('Round number (log)')
    ax.set_ylabel('Expected regret (log)')
    ax.set_title('eta=0.00001; delta=0.1; x from -1 to 1, initialized as 0')

def findPara():
    global sumRegret, noise, predict, queryY
    for eta in [0.001]:
        for delta in [0.001,0.005, 0.01,0.03, 0.05,0.08, 0.1, 0.15, 0.2, 0.5, 0.8]:
            sumRegret = 0
            for i in range(numRuns):
                noise = random.normal(0,1, numRounds)
                predict = 0   #initialization
                for t in range(numRounds):
                    loss = alg(t, eta, delta)
                    sumRegret += loss - environment(0, t) #compete with the hindsight optimum
            expectedRegret = sumRegret/numRuns
            print "eta:", eta, "  delta:", delta, "  Regret:", expectedRegret

def plotHeatMap():
    global sumRegret, noise, predict, queryY
    etaList = [pow(10, i) for i in range(-10,0)]
    roundsList = [pow(10, i) for i in range(1,5)]
    deltaList = [0.001,0.005, 0.01,0.03, 0.05,0.08, 0.1, 0.15, 0.2, 0.5, 0.8]
    avgRegretList = []  #store the average regret for each eta and T
    
    for eta in etaList:
        for rounds in roundsList:
            avgRegPerDelta = []
            for delta in deltaList:
                sumRegret = 0
                for i in range(numRuns):
                    noise = random.normal(0,1, rounds)
                    queryY = 0.2  #initialization
                    for t in range(rounds):
                        loss = alg(t, eta, delta)
                        sumRegret += loss - environment(0, t) #compete with the hindsight optimum
                avgRegPerDelta.append(sumRegret/numRuns/rounds)
            avgReg = min(avgRegPerDelta)
            avgRegretList.append(avgReg)
            print "eta:", eta, "  T:", rounds, "  Regret:", avgReg
    y = array(avgRegretList).reshape(-1, len(roundsList))

    fig = figure()
    ax = Axes3D(fig)
    x1,x2 = meshgrid(log10(etaList),log10(roundsList))
    ax.plot_surface(x1, x2, y.T, cstride=1, rstride=1, cmap=cm.jet)
    ax.set_xlabel('eta(log)')
    ax.set_ylabel('rounds(log)')
    ax.set_zlabel('average regret')
    show()


def trial():
    x1range = [1,2]
    x2range = [3,4,5]
    y = [1,1,1,6,6,6]
    y = array(y).reshape(-1, 3)

    fig = figure()
    ax = Axes3D(fig)
    x1,x2 = meshgrid(x1range,x2range)
    ax.plot_surface(x1, x2, y.T, cstride=1, rstride=1, cmap=cm.jet)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.set_zlabel('$max_a Q(s,a)$')
    show()

def writeF():
    global noise, predict, queryY
    for T in [100*i for i in range(1,11)]:
        minAvgRegret = 100.0
        optPredict = 0.0
        optEta = 0.0
        optDelta = 0.0
        for eta in [0.001*et for et in range(1, 11)]:
            for delta in [0.01*de for de in range(10,51)]:
                expPredict = 0.0
                expAvgRegret = 0.0
                for run in range(numRuns):
                    noise = random.normal(0,1, T)
                    queryY = 1.0  #initialization
                    for t in range(T):
                        loss = alg(t, eta, delta)
                        expAvgRegret += loss - environment(0.0, t) #the hindsight optimum
                    expPredict += predict
                expPredict = expPredict/float(numRuns)
                expAvgRegret = expAvgRegret/float(numRuns)/float(T)
                if expAvgRegret < minAvgRegret:
                    minAvgRegret = expAvgRegret
                    optPredict = expPredict
                    optEta = eta
                    optDelta = delta
                
        #fout = open('results.dat', 'a')
        #fout.write(repr(T)+' '+repr(optEta)+' '+repr(optDelta)+' '+repr(optPredict)+' '+repr(minAvgRegret)+'\n')
        print T, optEta, optDelta, optPredict, minAvgRegret
        #fout.close()

def runWithOptPara():
    global noise, predict, gradEst, queryY
    #data = loadtxt("exprmResults")
    T = 10000
    #eta = data.T[1]
    #delta = data.T[2]
    eta = [0.25*pow(t, -1.0) for t in range(1,T/2+1)]
    delta = [pow(3./16.*(1.+log(float(T)/2.))/float(T), 0.25) for t in range(1,T/2+1)]
    predictList = zeros((numRuns, T))
    avgRegList = zeros((numRuns, T))
    gradEstList = zeros((numRuns, T))
    
    #fout1 = open('predictX11', 'a')
    #fout2 = open('regretX11', 'a')

    for run in range(numRuns):
        noise = random.normal(0,1, T)
        queryY = 0.0
        sumRegret = 0.0
        for t in range(T):
            loss = alg(t, eta[t/2], delta[t/2])
            regret = loss - environment(0.0, t)
            sumRegret += regret
            predictList[run][t] = predict
            avgRegList[run][t] = sumRegret/float(t+1)
            gradEstList[run][t] = gradEst
                
            #fout1.write(repr(predictList[run][t])+' ')
            #fout2.write(repr(avgRegList[run][t])+' ')
        #fout1.write('\n')
        #fout2.write('\n')
        if (run+1)%100 == 0:
            print "run: ", run

    #fout1.close()
    #fout2.close()
    expPredict = predictList.mean(axis=0)
    expRegret = avgRegList.mean(axis=0)
    expGradEst = gradEstList.mean(axis=0)
    roundNum = linspace(1, T, T)
    baseline = [pow(t, -1.0/3.0) for t in roundNum]
    baseline2 = [pow(t, -1.0/2.0) for t in roundNum]

    fig, ax = plt.subplots()

    #ax.plot(log10(roundNum), log10(abs(expPredict)), label = "Prediction")
    #ax.plot(log10(roundNum), log10(abs(expGradEst)), label = "Gradient Estimate")
    ax.plot(log10(roundNum), log10(expRegret), label = "Average Regret (log)")
    ax.plot(log10(roundNum), log10(baseline2), label = "1/sqrt(T)")
    ax.plot(log10(roundNum), log10(baseline), label = "1/T^{1/3}")
    #ax.plot(roundNum, baseline)
    ax.set_title('loss: |x|^11  eta=0.25/t  delta=constant init=0')
    ax.legend(loc=0)
    ax.set_xlabel('round number (log)')
    
    savefig("regret_x11_1000runs.pdf")
    show()



def writeRegret():
    global noise, predict, gradEst, queryY
    for T in [100*i for i in range(1,101)]:
        eta = [0.25*pow(t, -1.0) for t in range(1,T/2+1)]
        delta = [pow(3./16.*(1.+log(float(T)/2.))/float(T), 0.25) for t in range(1,T/2+1)]
        sumRegret = 0.
        expPredict = 0.
        for run in range(numRuns):
            noise = random.normal(0,1, T)
            queryY = 0.0
            for t in range(T):
                loss = alg(t, eta[t/2], delta[t/2])
                regret = loss - environment(0.0, t)
                sumRegret += regret
            expPredict += predict
            if (run+1)%200 == 0:
                print "run: ", run, "T: ", T
        avgRegret = sumRegret/float(numRuns)/float(T)
        expPredict = expPredict/float(numRuns)
        fout = open('averageRegretOGD11', 'a')
        fout.write(repr(T)+' '+repr(avgRegret)+' '+repr(expPredict)+'\n')
        fout.close()

#run()
#plot()
#show()

#findPara()

#plotHeatMap()

writeF()

#runWithOptPara()

#writeRegret()
