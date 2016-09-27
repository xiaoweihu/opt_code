'''
multi-dimension case: vectors are represented by arrays

algFlaxman: algorithm using Flaxman's single point estimate

algSPSA: algorithm using SPSA gradient estimate

algBarrier: algorithm using self-concordant barrier
    Bandit Convex Optimization: Towards Tight Bounds
    Algorithm 1
    Constraint set: L2-ball, radius = 1
    Self-concordant barriar: R(x) = -ln( 1 - || x ||_2^2 )
    R'(x) = 2x/(1-||x||_2^2)   R''(x) = 2/(1-||x||_2^2)I + 4/(1-||x||_2^2)^2 xx'
    ==> \niu = 1
    Loss function: sigma-strongly convex, L-smooth

algHighlySmooth: algorithm using gradient estimate for highly smooth loss functions
    NIPS: highly smooth zeroth order online optimization

'''

import sys
from pylab import *
from numpy import *
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

numRuns = 1000

dimension = 5
radius = 1.
upperBound = 1.  #maximum of expectation of loss functions
beta = 2.        #order of smoothness

gradientList = []
xList = [zeros(dimension)]
u = ones(dimension)/sqrt(dimension) #the ramdom perturbation term
SPSAsign = 1.
SPSADelta = ones(dimension)
noise = []


def environment(y, t):
    return pow(linalg.norm(y),2) + noise[t]


def project(x, radius=1.):
    """
        Compute the Euclidean projection on a L2-ball
        Solves the optimization problem:
            min_y || y-x ||_2^2 , s.t. || y ||_2 <= radius
            
        Parameters: 
        x: numpy array, n-dimensional vector to project
        radius: float, optional, default: 1, radius of the constrained set
        
        Returns:
        numpy array, Euclidean projection of x on the constrained set
    """
    assert radius > 0
    n, = x.shape   #will raise Valueerror if x is not 1-D
    norm_x = linalg.norm(x)
    if not norm_x > radius:
        return x
    return x / (norm_x/radius)


def randomUnitVector():
    """
        Randomly choose a vector from a unit n-dimensional sphere
    """
    vec = array([random.normal(0., 1.) for i in range(dimension)])
    mag = linalg.norm(vec)
    return vec / mag


#require same delta for round 2t & (2t+1), t=0,1,...
def algFlaxman(t, eta, delta):
    global gradientList, xList, u
    if t%2 == 0 :
        u = randomUnitVector()
        predict = xList[t] + delta*u
        loss = environment(predict, t)
        gradEst = float(dimension)/delta * loss * u
        gradientList.append(gradEst)
        xList.append(xList[t])
    else:
        predict = xList[t] - delta*u
        loss = environment(predict, t)
        gradEst = gradientList[t-1] - float(dimension)/delta * loss * u
        gradientList.append(gradEst)
        update = xList[t] - eta*gradEst
        update = project(update)
        xList.append(update)
    return loss, predict


def algSPSA(t, eta, delta):
    global gradientList, xList, SPSAsign, SPSADelta
    if t%2 == 0 :
        SPSAsign = random.choice([-1., 1.])
        SPSADelta = array([random.choice([-1., 1.]) for i in range(dimension)]) #Bernouli distribution
        predict = xList[t] + delta * SPSAsign * SPSADelta
        loss = environment(predict, t)
        gradEst = array([SPSAsign * loss / delta / Delta_i for Delta_i in SPSADelta])
        gradientList.append(gradEst)
        xList.append(xList[t])
    else:
        predict = xList[t] - delta * SPSAsign * SPSADelta
        loss = environment(predict, t)
        gradEst = gradientList[t-1] - array([SPSAsign * loss / delta / Delta_i for Delta_i in SPSADelta])
        gradientList.append(gradEst)
        update = xList[t] - eta*gradEst
        update = project(update)
        xList.append(update)
    return loss, predict


def matSqrtInv(A):
    """
        Compute the inverse of square root of a real symmetric matrix A
        Using Eigendecomposition A = Q \Lambda Q'
        
        NOTE: BAD PRECISION
    """
    eigValues, eigVectors = linalg.eig(A)
    eigValSqrt = diag([1./sqrt(i) if i>0 else 1000. for i in eigValues])
    return matrix(eigVectors).T * eigValSqrt * matrix(eigVectors)


def sumArray(A):
    """
        A = [array([1,2]), array([3,4])]
        sumArray(A) =array([4,6])
    """
    sum = zeros(len(A[0]))
    for item in A:
        sum += item
    return sum

def algBarrier(t, eta, sigma):
    global gradientList, xList
    B = 2./(1.-linalg.norm(xList[t])**2) * identity(dimension) + matrix(xList[t]).T * matrix(xList[t])*4./(1.-linalg.norm(xList[t])**2)**2 + eta*sigma*float(t+1)*identity(dimension)
    #B = matSqrtInv(B)
    B = scipy.linalg.sqrtm(B)
    B = linalg.inv(B)
    u = randomUnitVector()
    predict = matrix(xList[t]).T + B * matrix(u).T
    predict = squeeze(asarray(predict))
    loss = environment(predict, t)
    graEst = linalg.inv(B) * matrix(u).T * dimension * loss
    graEst = squeeze(asarray(graEst))
    gradientList.append(graEst)
    
    sum_gx = sumArray(gradientList) - sigma*sumArray(xList)
    norm_sum_gx = linalg.norm(sum_gx)
    direction_x = sum_gx / norm_sum_gx
    coeffs = [-float(t+1)*sigma, norm_sum_gx, float(t+1)*sigma + 2./eta, -norm_sum_gx]
    norm_x = roots(coeffs)
    norm_x = [x for x in norm_x if (isreal(x) and x>0 and x<1)]
    update = [x * direction_x for x in norm_x]
    if len(norm_x) > 1:
        minObj = [sum([dot(gradientList[i], x)+sigma/2.*dot(x-xList[i], x-xList[i])
                       for i in range(0,t)])
                  - log(1. - dot(x, x))/eta for x in update]
        update = update[argmin(minObj)]
    update = update[0]
    xList.append(update)
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
    u = randomUnitVector()
    r = random.uniform(-1, 1)  #randomly choose r uniform in [-1,1]
    predict = xList[t] + delta * r * u
    loss = environment(predict, t)
    gradEst = gradEst4HS(beta, r, dimension, delta, loss, u)
    update = xList[t] - eta * gradEst
    update = project(update)
    xList.append(update)
    return loss, predict


def writeRegret():
    global gradientList, xList, noise
    sigma = 2.
    L = math.factorial(beta)

    for T in [100*i for i in [1000]]:
        #parameters
        #alg: smooth & s.c.
        eta = [1./(2.*sigma*float(t)) for t in range(1,T/2+1)]
        delta = pow((12.*dimension**2*(2.*radius)**2*(1+log(T/2))) / (3.*dimension**2*(L-sigma)**2*(1+log(T/2))+L*float(T)*16.*sigma), 0.25)
        #
        #algBarrier:
        #eta = pow((1.+2.*L/sigma)*log(T)/(2. * dimension**2 * upperBound**2 * float(T)), 0.5)
        #
        #algHighlySmooth: strongly convex & smooth
        #eta = [1./ sigma / float(t+1) for t in range(T)]
        #delta = pow(log(T) * beta**2 * dimension**2 * math.factorial(beta)/ (float(T) * sigma * L), 1./(beta+2.)) #online setting
        #delta = pow(pow(log(T)*beta/float(T), 0.5)*dimension *math.factorial(beta)/L, 1./beta)   #optimization setting
        #
        #algHighlySmooth: smooth
        #delta = pow(dimension*2.*radius*sqrt(beta)*math.factorial(beta)/(sqrt(T)*L), 1./(beta+1.))
        #eta = delta*2.*radius / (pow(beta,3./2.) * dimension * sqrt(T))
        
        sumRegret = 0.
        expPredict = zeros(dimension)
        for run in range(numRuns):
            noise = random.normal(0,1, T)
            gradientList = []
            xList = [zeros(dimension)]
            for t in range(T):
                loss, predict = algSPSA(t, eta[t/2], delta)
                #loss, predict = algBarrier(t, eta, sigma)
                #loss, predict = algHighlySmooth(t, eta, delta)
                regret = loss - noise[t]
                sumRegret += regret
            expPredict += predict
            if (run+1)%100 == 0:
                print "run: ", run, "T: ", T
        avgRegret = sumRegret/float(numRuns)/float(T)
        expPredict = expPredict/float(numRuns)
        with open('averageRegretSPSA2_5d', 'a') as fout:
            fout.write(repr(T)+' '+repr(avgRegret)+' ')
            for item in expPredict:
                fout.write(repr(item) + ' ')
            fout.write('\n')

'''
noise = random.normal(0,1, 10)
for t in range(10):
    loss, predict = algHighlySmooth(t, 0.05, 0.05)
    print "round: ", t, "loss: ", loss, "predict: ", predict


a = matrix([[10,1],[1,30]])
b = matSqrtInv(a)
print b * b * a
'''
writeRegret()


