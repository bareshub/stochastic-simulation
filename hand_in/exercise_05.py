import numpy as np
from scipy import stats
rng=np.random.default_rng()
N=100
true_value=np.e-1

# Part 1 - Crude Monte Carlo
U=rng.uniform(0,1,N)
X=np.exp(U)

mean_mc=np.mean(X)
se_mc=np.std(X,ddof=1)/np.sqrt(N)
var_mc=np.var(X,ddof=1)

print("Part 1 - Crude Monte Carlo")
print("estimate =",mean_mc)
print("true value =",true_value)
print("95% CI =",mean_mc-1.96*se_mc,mean_mc+1.96*se_mc)
print("variance =",var_mc)

# Part 2 - Antithetic Variables
U=rng.uniform(0,1,N)
Y_av=(np.exp(U)+np.exp(1-U))/2

mean_av=np.mean(Y_av)
se_av=np.std(Y_av,ddof=1)/np.sqrt(N)
var_av=np.var(Y_av,ddof=1)

print("\nPart 2 - Antithetic Variables")
print("estimate =",mean_av)
print("true value =",true_value)
print("95% CI =",mean_av-1.96*se_av,mean_av+1.96*se_av)
print("variance =",var_av)
print("variance reduction =",100*(1-var_av/var_mc),"%")


# Part 3 - Control Variates
U=rng.uniform(0,1,N)
X=np.exp(U)
c=-np.cov(X,U,ddof=1)[0,1]/np.var(U,ddof=1)
Y_cv=X+c*(U-0.5)
mean_cv=np.mean(Y_cv)
se_cv=np.std(Y_cv,ddof=1)/np.sqrt(N)
var_cv=np.var(Y_cv,ddof=1)

print("\nPart 3 - Control Variates")
print("c =",c)
print("estimate =",mean_cv)
print("true value =",true_value)
print("95% CI =",mean_cv-1.96*se_cv,mean_cv+1.96*se_cv)
print("variance =",var_cv)
print("variance reduction =",100*(1-var_cv/var_mc),"%")


# Part 4 - Stratified Sampling
k=10
j=np.arange(1,k+1)

U_strat=rng.uniform(0,1,(N,k))/k+(j-1)/k
Y_ss=np.mean(np.exp(U_strat),axis=1)
mean_ss=np.mean(Y_ss)
se_ss=np.std(Y_ss,ddof=1)/np.sqrt(N)
var_ss=np.var(Y_ss,ddof=1)
print("\nPart 4 - Stratified Sampling")
print("estimate =",mean_ss)
print("true value =",true_value)
print("95% CI =",mean_ss-1.96*se_ss,mean_ss+1.96*se_ss)
print("variance =",var_ss)
print("variance reduction =",100*(1-var_ss/var_mc),"%")


# Part 5 - Control variates for blocking probability
print("\nPart 5 - Control Variates for Blocking Probability")

lam=5
T=10
m=10
mean_service=8
n=1000
rng=np.random.default_rng()

def simulate_rep():
    N_i=rng.poisson(lam*T)

    if N_i==0:
        return 0.0,0

    arrivals=np.cumsum(rng.exponential(1.0,N_i))
    services=rng.exponential(mean_service,N_i)

    departures=np.array([])
    blocked=0

    for a,s in zip(arrivals,services):
        departures=departures[departures>a]

        if len(departures)<m:
            departures=np.append(departures,a+s)
        else:
            blocked=blocked+1

    return blocked/N_i,N_i


X_p,N_p=np.array([simulate_rep() for i in range(n)]).T

EN=lam*T
p_hat=np.mean(X_p)
var_mc=np.var(X_p,ddof=1)
c_p=-np.cov(X_p,N_p)[0,1]/np.var(N_p,ddof=1)
Y_p=X_p+c_p*(N_p-EN)

p_cv=np.mean(Y_p)
var_cv=np.var(Y_p,ddof=1)
print("crude MC p_hat =",p_hat)
print("crude MC variance =",var_mc)
print("control variate estimate =",p_cv)
print("control variate variance =",var_cv)
print("c =",c_p)

if var_mc>0:
    print("variance reduction =",100*(1-var_cv/var_mc),"%")
else:
    print("crude MC observed zero variance")

#Part 6


# Part 7 - Importance Sampling for P(Z>a)
print("\nPart 7 - Importance Sampling")

for a in [2,4]:
    true_p=1-stats.norm.cdf(a)

    Z=rng.standard_normal(N)
    est_mc=np.mean(Z>a)
    var_mc7=np.var(Z>a,ddof=1)
    Y=rng.normal(a,1.0,N)
    w=stats.norm.pdf(Y)/stats.norm.pdf(Y,loc=a)
    est_is=np.mean((Y>a)*w)
    var_is7=np.var((Y>a)*w,ddof=1)

    print("\na =",a)
    print("true probability =",true_p)
    print("crude MC estimate =",est_mc)
    print("crude MC variance =",var_mc7)
    print("IS estimate =",est_is)
    print("IS variance =",var_is7)
    if var_mc7>0:
        print("variance reduction =",100*(1-var_is7/var_mc7),"%")
    else:
        print("crude MC observed zero events")


# Part 8 - IS with g(x)=lambda exp(-lambda x)
print("\nPart 8 - Importance Sampling with exponential density")

for lam in [0.5,1.0,2.0,5.0]:
    U=rng.uniform(0,1,N)
    Zl=1-np.exp(-lam)
    Y=-np.log(1-U*Zl)/lam
    g=lam*np.exp(-lam*Y)/Zl
    est=np.mean(np.exp(Y)/g)
    var=np.var(np.exp(Y)/g,ddof=1)
    print("lambda =",lam)
    print("estimate =",est)
    print("variance =",var)




# Part 9 - Pareto IS with first moment distribution
print("\nPart 9 - Pareto IS")

beta=1.0
k=2.05
E_X=beta*k/(k-1)
U=rng.uniform(0,1,N)
Y9=beta*(1-U)**(-1/(k-1))
f9=k/beta*(Y9/beta)**(-k-1)
g9=(k-1)/beta*(Y9/beta)**(-k)
weights=Y9*f9/g9

print("theoretical mean =",E_X)
print("IS estimate =",np.mean(weights))
print("IS variance =",np.var(weights,ddof=1))