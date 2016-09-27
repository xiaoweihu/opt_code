'''
alg: algorithm using Flaxman's single point estimate

algBarrier: algorithm using self-concordant barrier
    Bandit Convex Optimization: Towards Tight Bounds
    Algorithm 1
    Constraint set: [-1, 1]
    Self-concordant barriar: R(x) = -ln(1-x)-ln(1+x)
            R'(x) = 2x/(1-x^2)   R''(x) = 2(1+x^2)/(1-x^2)^2
            \niu >= 2x^2/(1+x^2) ==> \niu = 1
    Loss function: sigma-strongly convex, L-smooth
    
algHighlySmooth: algorithm using gradient estimate for highly smooth loss functions
    COLT: highly smooth zeroth order online optimization

'''

import sys
from pylab import *
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

numRuns = 100

dimension = 1.
range1 = -2.     #range of the constrained set
range2 = 2.
upperBound = 1.  #maximum of expectation of loss functions
beta = 3.        #order of smoothness

gradientList = []
xList = [0.]
u = 1. #the ramdom disturbance term
noise = []

epsilon = 0.1


def environment(y, t):
    global epsilon
    #return pow(abs(y-1.),3) + noise[t]
    return epsilon / 2. * pow(y-1.,2) + noise[t]
    #return epsilon * (y-1.) + 2*epsilon*epsilon*log(1.+exp(-(y-1.)/epsilon)) + noise[t]


def project(x):
    global range1, range2
    if x < range1:
        x = range1
    elif x > range2:
        x = range2
    else:
        x = x
    return x


#require same delta for round 2t & (2t+1), t=0,1,...
def alg(t, eta, delta):
    global gradientList, xList, u
    if t%2 == 0 :
        u = float(random.choice([-1,1]))
        predict = xList[t] + delta*u
        loss = environment(predict, t)
        gradEst = float(dimension)/delta * loss * float(u)
        gradientList.append(gradEst)
        xList.append(xList[t])
    else:
        predict = xList[t] - delta*u
        loss = environment(predict, t)
        gradEst = gradientList[t-1] - float(dimension)/delta * loss * float(u)
        gradientList.append(gradEst)
        update = xList[t] - eta*gradEst
        update = project(update)
        xList.append(update)
    return loss, predict


def algBarrier(t, eta, sigma):
    global gradientList, xList
    B = pow(2.*(1.+xList[t]**2)/pow(1.-xList[t]**2,2) + eta*sigma*float(t+1), -0.5)
    u = float(random.choice([-1,1]))
    predict = xList[t] + B * u
    loss = environment(predict, t)
    graEst = loss / B * u
    gradientList.append(graEst)
    #the optimization step
    coeff2 = sum([gradientList[i] - sigma*xList[i] for i in range(0,t+1)])
    coeffs = [float(t+1)*sigma, coeff2, -float(t+1)*sigma-2./eta, -coeff2]
    update = roots(coeffs)
    update = [x for x in update if (isreal(x) and x<1 and x>-1)]
    if len(update) > 1:
        minObj = [sum([gradientList[i]*x+sigma/2.*pow(x-xList[i],2) \
                       for i in range(0,t+1)]) \
                  + (-log(1-x)-log(1+x))/eta for x in update]
        update = update[argmin(minObj)]
    xList += update
    return loss, predict


#beta: the order of smoothness
def gradEst4HS(beta, r, dimension, delta, loss, u):
    k = {
        1: 3. * r,
        2: 3. * r,
        3: 15. * r / 4. * (5. - 7. * r ** 3),
        4: 15. * r / 4. * (5. - 7. * r ** 3),
        5: 195. * r / 64. * (99.*r**4 - 126.*r**2 + 35.),
        6: 195. * r / 64. * (99.*r**4 - 126.*r**2 + 35.),
        10: (3828825.*r**9 - 9069060.*r**7 + 12170070.*r**5 - 8450820.*r**3 + 1924825.*r)/16384.
    }.get(beta)
    return float(dimension) / delta * loss * k * u


def algHighlySmooth(t, eta, delta):
    global xList
    u = float(random.choice([-1,1]))
    r = random.uniform(-1, 1)  #randomly choose r uniform in [-1,1]
    predict = xList[t] + delta * r * u
    loss = environment(predict, t)
    gradEst = gradEst4HS(beta, r, dimension, delta, loss, u)
    update = xList[t] - eta * gradEst
    update = project(update)
    xList.append(update)
    return loss, predict




def run():
    global gradientList, xList, noise
    T = 1000
    sigma = 2.
    L = 2.
    eta = pow((1.+2.*L/sigma)*log(T)/(2. * upperBound**2 * float(T)), 0.5)
    print "eta: ", eta
    
    predictList = zeros((numRuns, T))
    avgRegList = zeros((numRuns, T))
    for run in range(numRuns):
        noise = random.normal(0,1, T)
        gradientList = []
        xList = [0.]
        sumRegret = 0.
        for t in range(T):
            loss, predict = algSCB(t, eta, sigma)
            regret = loss - environment(0., t)
            sumRegret += regret
            predictList[run][t] = predict
            avgRegList[run][t] = sumRegret/float(t+1)
        if (run+1)%100 == 0:
            print "run: ", run
    expPredict = predictList.mean(axis=0)
    expRegret = avgRegList.mean(axis=0)
    roundNum = linspace(1, T, T)
    baseline1 = [pow(t, -1.0/3.0) for t in roundNum]
    baseline2 = [pow(t, -1.0/2.0) for t in roundNum]

    fig, ax = plt.subplots()

    #ax.plot(log10(roundNum), log10(abs(expPredict)), label = "Prediction")
    ax.plot(log10(roundNum), log10(expRegret), label = "Average Regret (log)")
    ax.plot(log10(roundNum), log10(baseline2), label = "1/sqrt(T)")
    ax.plot(log10(roundNum), log10(baseline1), label = "1/T^{1/3}")
    ax.set_title('loss: |x|^2 sigma=2 L=2')
    ax.legend(loc=0)
    ax.set_xlabel('round number (log)')
    
    savefig("BCOregret_x2_1000runs.pdf")
    #show()


def writeRegret():
    global gradientList, xList, noise, epsilon
    sigma = 2.
    L = math.factorial(beta)
    
    
    numRounds = [1]
    r = 1
    while r < 100000:
        r += int(pow(10, floor(log10(r))))
        numRounds.append(r)
    for T in numRounds:
        #parameters
        #alg: smooth
        #eta = [0.25*pow(t, -1.0) for t in range(1,T/2+1)]
        #delta = pow(3./16.*(1.+log(float(T)/2.))/float(T), 0.25)
        #
        #algBarrier:
        #eta = pow((1.+2.*L/sigma)*log(T)/(2. * upperBound**2 * float(T)), 0.5)
        #
        #algHighlySmooth: strongly convex & smooth
        #eta = [1./ sigma / float(t+1) for t in range(T)]
        #delta = pow(log(T) * beta**2 * dimension**2 * math.factorial(beta)/ (float(T) * sigma * L), 1./(beta+2.)) #online setting
        #delta = pow(pow(log(T)*beta/float(T), 0.5)*dimension *math.factorial(beta)/L, 1./beta)   #optimization setting
        #
        #algHighlySmooth: smooth
        delta = pow(dimension*(range2-range1)*sqrt(beta)*math.factorial(beta)/(sqrt(T)*L), 1./(beta+1.))
        eta = delta*(range2-range1) / (pow(beta,3./2.) * dimension * sqrt(T))
        
        epsilon = 1.5*pow(1./16., 1./3.)*pow(T, -1./3.)
        
        sumRegret = 0.
        expPredict = 0.
        for run in range(numRuns):
            noise = random.normal(0,1, T)
            gradientList = []
            xList = [0.]
            for t in range(T):
                #loss, predict = alg(t, eta[t/2], delta)
                #loss, predict = algBarrier(t, eta, sigma)
                loss, predict = algHighlySmooth(t, eta, delta)
                regret = loss - noise[t]
                sumRegret += regret
            expPredict += predict
            if (run+1)%50 == 0:
                print "run: ", run, "T: ", T
        avgRegret = sumRegret/float(numRuns)/float(T)
        expPredict = expPredict/float(numRuns)
        fout = open('regretHSepsilonX2', 'a')
        fout.write(repr(T)+' '+repr(avgRegret)+' '+repr(expPredict)+'\n')
        fout.close()



#run()
writeRegret()




