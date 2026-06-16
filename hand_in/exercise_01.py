import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

plt.close('all')

# part1 
N_obs=10000
a_1=21;c_1=7;M_1=80;x_1=np.zeros(N_obs,dtype=int);x_1[0]=1 # Part 1: Decent generator
a_2=17;c_2=7;M_2=80;x_2=np.zeros(N_obs,dtype=int);x_2[0]=1 # Part 1: Bad generator
a_3=16;c_3=7;M_3=80;x_3=np.zeros(N_obs,dtype=int);x_3[0]=1 # Part 1: Bad generator
a_4=15;c_4=7;M_4=80;x_4=np.zeros(N_obs,dtype=int);x_4[0]=1 # Part 1: Bad generator

for i in range(1,N_obs):
    x_1[i]=(a_1*x_1[i-1]+c_1)%M_1
    x_2[i]=(a_2*x_2[i-1]+c_2)%M_2
    x_3[i]=(a_3*x_3[i-1]+c_3)%M_3
    x_4[i]=(a_4*x_4[i-1]+c_4)%M_4

# Histogram counts
int1=np.linspace(0,M_1,21);counts_1=np.histogram(x_1,int1)[0]
int2=np.linspace(0,M_2,21);counts_2=np.histogram(x_2,int2)[0]
int3=np.linspace(0,M_3,21);counts_3=np.histogram(x_3,int3)[0]
int4=np.linspace(0,M_4,21);counts_4=np.histogram(x_4,int4)[0]

plt.figure(figsize=(14,4))
plt.subplot(1,4,1)
plt.bar(int1[:-1],counts_1)
plt.title('LCG 1')
plt.ylabel('Counts')
plt.xlabel('Value')
plt.subplot(1,4,2)
plt.bar(int2[:-1],counts_2)
plt.title('LCG 2')
plt.ylabel('Counts')
plt.xlabel('Value')
plt.subplot(1,4,3)
plt.bar(int3[:-1],counts_3)
plt.title('LCG 3')
plt.ylabel('Counts')
plt.xlabel('Value')
plt.subplot(1,4,4)
plt.bar(int4[:-1],counts_4)
plt.title('LCG 4')
plt.ylabel('Counts')
plt.xlabel('Value')
plt.tight_layout()
plt.show()

# Part 2: System available generator:
x_syst1=np.random.randint(0,M_1,(N_obs,1)) # integers uniformly distributed on {0,...,M_1-1}
x_syst2=np.random.randint(0,M_2,(N_obs,1)) # integers uniformly distributed on {0,...,M_2-1}


# Kolmogorov-Smirnov statistical test
x_1_sort=np.sort(x_1)
F=np.arange(1,N_obs+1)/N_obs # Theoretical Uniform Distribution
F1_n=np.searchsorted(x_1_sort,x_1_sort,side='right')/N_obs # CDF
D1_n=np.max(np.abs(F1_n-F)) # Test statistic
D1_n_adjusted=(np.sqrt(N_obs)+0.12+0.11/np.sqrt(N_obs))*D1_n
if D1_n_adjusted<1.480: # Critical value for significance level alpha=0.025
    print('KS test LCG 1: Null hypothesis not rejected')
else:
    print('KS test LCG 1: Null hypothesis rejected')

x_2_sort=np.sort(x_2)
F=np.arange(1,N_obs+1)/N_obs # Theoretical Uniform Distribution
F2_n=np.searchsorted(x_2_sort,x_2_sort,side='right')/N_obs # CDF
D2_n=np.max(np.abs(F2_n-F)) # Test statistic
D2_n_adjusted=(np.sqrt(N_obs)+0.12+0.11/np.sqrt(N_obs))*D2_n
if D2_n_adjusted<1.480: # Critical value for significance level alpha=0.025
    print('KS test LCG 2: Null hypothesis not rejected')
else:
    print('KS test LCG 2: Null hypothesis rejected')

x_3_sort=np.sort(x_3)
F=np.arange(1,N_obs+1)/N_obs # Theoretical Uniform Distribution
F3_n=np.searchsorted(x_3_sort,x_3_sort,side='right')/N_obs # CDF
D3_n=np.max(np.abs(F3_n-F)) # Test statistic
D3_n_adjusted=(np.sqrt(N_obs)+0.12+0.11/np.sqrt(N_obs))*D3_n
if D3_n_adjusted<1.480: # Critical value for significance level alpha=0.025
    print('KS test LCG 3: Null hypothesis not rejected')
else:
    print('KS test LCG 3: Null hypothesis rejected')

x_4_sort=np.sort(x_4)
F=np.arange(1,N_obs+1)/N_obs # Theoretical Uniform Distribution
F4_n=np.searchsorted(x_4_sort,x_4_sort,side='right')/N_obs # CDF
D4_n=np.max(np.abs(F4_n-F)) # Test statistic
D4_n_adjusted=(np.sqrt(N_obs)+0.12+0.11/np.sqrt(N_obs))*D4_n
if D4_n_adjusted<1.480: # Critical value for significance level alpha=0.025
    print('KS test LCG 4: Null hypothesis not rejected')
else:
    print('KS test LCG 4: Null hypothesis rejected')

# Chi^2 test. Degrees of freedom: v=M_1-1-0=M_1-1
O_1=np.histogram(x_1,np.arange(0,M_1+1))[0] # Observations per state
E_1=N_obs*np.ones(M_1)/M_1 # Theoretical Expected counts
T_1=np.sum((O_1-E_1)**2/E_1) # Test statistic
if T_1<chi2.ppf(0.975,M_1-1): # Critical value for significance level alpha=0.025
    print('Chi-square test LCG 1: null hypothesis not rejected, T= '+str(T_1))
else:
    print('Chi-square test LCG 1: null hypothesis rejected, T= '+str(T_1))

# Chi^2 test. Degrees of freedom: v=M_2-1-0=M_2-1
O_2=np.histogram(x_2,np.arange(0,M_2+1))[0] # Observations per state
E_2=N_obs*np.ones(M_2)/M_2 # Theoretical Expected counts
T_2=np.sum((O_2-E_2)**2/E_2) # Test statistic
if T_2<chi2.ppf(0.975,M_2-1): # Critical value for significance level alpha=0.025
    print('Chi-square test LCG 2: null hypothesis not rejected, T= '+str(T_2))
else:
    print('Chi-square test LCG 2: null hypothesis rejected, T= '+str(T_2))

# Chi^2 test. Degrees of freedom: v=M_3-1-0=M_3-1
O_3=np.histogram(x_3,np.arange(0,M_3+1))[0] # Observations per state
E_3=N_obs*np.ones(M_3)/M_3 # Theoretical Expected counts
T_3=np.sum((O_3-E_3)**2/E_3) # Test statistic
if T_3<chi2.ppf(0.975,M_3-1): # Critical value for significance level alpha=0.025
    print('Chi-square test LCG 3: null hypothesis not rejected, T= '+str(T_3))
else:
    print('Chi-square test LCG 3: null hypothesis rejected, T= '+str(T_3))

# Chi^2 test. Degrees of freedom: v=M_4-1-0=M_4-1
O_4=np.histogram(x_4,np.arange(0,M_4+1))[0] # Observations per state
E_4=N_obs*np.ones(M_4)/M_4 # Theoretical Expected counts
T_4=np.sum((O_4-E_4)**2/E_4) # Test statistic
if T_4<chi2.ppf(0.975,M_4-1): # Critical value for significance level alpha=0.025
    print('Chi-square test LCG 4: null hypothesis not rejected, T= '+str(T_4))
else:
    print('Chi-square test LCG 4: null hypothesis rejected, T= '+str(T_4))

# TESTS FOR INDEPENDENCE
#   Graphical methods: scatter plot test
#   Run tests
#   Correlation test

plt.figure(figsize=(14,4))
plt.subplot(1,4,1)
plt.scatter(x_1[:-1],x_1[1:],s=5)
plt.title('LCG 1')
plt.xlabel('x_i')
plt.ylabel('x_i+1')
plt.subplot(1,4,2)
plt.scatter(x_2[:-1],x_2[1:],s=5)
plt.title('LCG 2')
plt.xlabel('x_i')
plt.ylabel('x_i+1')
plt.subplot(1,4,3)
plt.scatter(x_3[:-1],x_3[1:],s=5)
plt.title('LCG 3')
plt.xlabel('x_i')
plt.ylabel('x_i+1')
plt.subplot(1,4,4)
plt.scatter(x_4[:-1],x_4[1:],s=5)
plt.title('LCG 4')
plt.xlabel('x_i')
plt.ylabel('x_i+1')
plt.tight_layout()
plt.show()

z_1=x_1>=M_1/2
runs_1=1+np.sum(z_1[1:]!=z_1[:-1])
n1_1=float(np.sum(z_1))
n2_1=float(N_obs-n1_1)
E_runs_1=1+2*n1_1*n2_1/N_obs
V_runs_1=(2*n1_1*n2_1*(2*n1_1*n2_1-N_obs))/(N_obs**2*(N_obs-1))
Z_runs_1=(runs_1-E_runs_1)/np.sqrt(V_runs_1)
if abs(Z_runs_1)<1.96:
    print('Runs test LCG 1: null hypothesis not rejected, Z= '+str(Z_runs_1))
else:
    print('Runs test LCG 1: null hypothesis rejected, Z= '+str(Z_runs_1))

z_2=x_2>=M_2/2
runs_2=1+np.sum(z_2[1:]!=z_2[:-1])
n1_2=float(np.sum(z_2))
n2_2=float(N_obs-n1_2)
E_runs_2=1+2*n1_2*n2_2/N_obs
V_runs_2=(2*n1_2*n2_2*(2*n1_2*n2_2-N_obs))/(N_obs**2*(N_obs-1))
Z_runs_2=(runs_2-E_runs_2)/np.sqrt(V_runs_2)
if abs(Z_runs_2)<1.96:
    print('Runs test LCG 2: null hypothesis not rejected, Z= '+str(Z_runs_2))
else:
    print('Runs test LCG 2: null hypothesis rejected, Z= '+str(Z_runs_2))

z_3=x_3>=M_3/2
runs_3=1+np.sum(z_3[1:]!=z_3[:-1])
n1_3=float(np.sum(z_3))
n2_3=float(N_obs-n1_3)
E_runs_3=1+2*n1_3*n2_3/N_obs
V_runs_3=(2*n1_3*n2_3*(2*n1_3*n2_3-N_obs))/(N_obs**2*(N_obs-1))
Z_runs_3=(runs_3-E_runs_3)/np.sqrt(V_runs_3)
if abs(Z_runs_3)<1.96:
    print('Runs test LCG 3: null hypothesis not rejected, Z= '+str(Z_runs_3))
else:
    print('Runs test LCG 3: null hypothesis rejected, Z= '+str(Z_runs_3))

z_4=x_4>=M_4/2
runs_4=1+np.sum(z_4[1:]!=z_4[:-1])
n1_4=float(np.sum(z_4))
n2_4=float(N_obs-n1_4)
E_runs_4=1+2*n1_4*n2_4/N_obs
V_runs_4=(2*n1_4*n2_4*(2*n1_4*n2_4-N_obs))/(N_obs**2*(N_obs-1))
Z_runs_4=(runs_4-E_runs_4)/np.sqrt(V_runs_4)
if abs(Z_runs_4)<1.96:
    print('Runs test LCG 4: null hypothesis not rejected, Z= '+str(Z_runs_4))
else:
    print('Runs test LCG 4: null hypothesis rejected, Z= '+str(Z_runs_4))

h=1

rho_1=np.corrcoef(x_1[:-h],x_1[h:])[0,1]
Z_corr_1=rho_1*np.sqrt(N_obs)
if abs(Z_corr_1)<1.96:
    print('Correlation test LCG 1: null hypothesis not rejected, rho= '+str(rho_1))
else:
    print('Correlation test LCG 1: null hypothesis rejected, rho= '+str(rho_1))

rho_2=np.corrcoef(x_2[:-h],x_2[h:])[0,1]
Z_corr_2=rho_2*np.sqrt(N_obs)
if abs(Z_corr_2)<1.96:
    print('Correlation test LCG 2: null hypothesis not rejected, rho= '+str(rho_2))
else:
    print('Correlation test LCG 2: null hypothesis rejected, rho= '+str(rho_2))

rho_3=np.corrcoef(x_3[:-h],x_3[h:])[0,1]
Z_corr_3=rho_3*np.sqrt(N_obs)
if abs(Z_corr_3)<1.96:
    print('Correlation test LCG 3: null hypothesis not rejected, rho= '+str(rho_3))
else:
    print('Correlation test LCG 3: null hypothesis rejected, rho= '+str(rho_3))

rho_4=np.corrcoef(x_4[:-h],x_4[h:])[0,1]
Z_corr_4=rho_4*np.sqrt(N_obs)
if abs(Z_corr_4)<1.96:
    print('Correlation test LCG 4: null hypothesis not rejected, rho= '+str(rho_4))
else:
    print('Correlation test LCG 4: null hypothesis rejected, rho= '+str(rho_4))


# Part 2
x_syst1=x_syst1.flatten()
x_syst2=x_syst2.flatten()

int_syst1=np.linspace(0,M_1,21);counts_syst1=np.histogram(x_syst1,int_syst1)[0]
int_syst2=np.linspace(0,M_2,21);counts_syst2=np.histogram(x_syst2,int_syst2)[0]

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.bar(int_syst1[:-1],counts_syst1)
plt.title('System generator 1')
plt.ylabel('Counts')
plt.xlabel('Value')
plt.subplot(1,2,2)
plt.bar(int_syst2[:-1],counts_syst2)
plt.title('System generator 2')
plt.ylabel('Counts')
plt.xlabel('Value')
plt.tight_layout()
plt.show()

# Kolmogorov-Smirnov statistical test
x_syst1_sort=np.sort(x_syst1)
F=np.arange(1,N_obs+1)/N_obs # Theoretical Uniform Distribution
F_syst1_n=np.searchsorted(x_syst1_sort,x_syst1_sort,side='right')/N_obs # CDF
D_syst1_n=np.max(np.abs(F_syst1_n-F)) # Test statistic
D_syst1_n_adjusted=(np.sqrt(N_obs)+0.12+0.11/np.sqrt(N_obs))*D_syst1_n
if D_syst1_n_adjusted<1.480: # Critical value for significance level alpha=0.025
    print('KS test system generator 1: Null hypothesis not rejected')
else:
    print('KS test system generator 1: Null hypothesis rejected')

x_syst2_sort=np.sort(x_syst2)
F_syst2_n=np.searchsorted(x_syst2_sort,x_syst2_sort,side='right')/N_obs # CDF
D_syst2_n=np.max(np.abs(F_syst2_n-F)) # Test statistic
D_syst2_n_adjusted=(np.sqrt(N_obs)+0.12+0.11/np.sqrt(N_obs))*D_syst2_n
if D_syst2_n_adjusted<1.480: # Critical value for significance level alpha=0.025
    print('KS test system generator 2: Null hypothesis not rejected')
else:
    print('KS test system generator 2: Null hypothesis rejected')

# Chi^2 test. Degrees of freedom: v=M_1-1-0=M_1-1
O_syst1=np.histogram(x_syst1,np.arange(0,M_1+1))[0] # Observations per state
E_syst1=N_obs*np.ones(M_1)/M_1 # Theoretical Expected counts
T_syst1=np.sum((O_syst1-E_syst1)**2/E_syst1) # Test statistic
if T_syst1<chi2.ppf(0.975,M_1-1): # Critical value for significance level alpha=0.025
    print('Chi-square test system generator 1: null hypothesis not rejected, T= '+str(T_syst1))
else:
    print('Chi-square test system generator 1: null hypothesis rejected, T= '+str(T_syst1))

# Chi^2 test. Degrees of freedom: v=M_2-1-0=M_2-1
O_syst2=np.histogram(x_syst2,np.arange(0,M_2+1))[0] # Observations per state
E_syst2=N_obs*np.ones(M_2)/M_2 # Theoretical Expected counts
T_syst2=np.sum((O_syst2-E_syst2)**2/E_syst2) # Test statistic
if T_syst2<chi2.ppf(0.975,M_2-1): # Critical value for significance level alpha=0.025
    print('Chi-square test system generator 2: null hypothesis not rejected, T= '+str(T_syst2))
else:
    print('Chi-square test system generator 2: null hypothesis rejected, T= '+str(T_syst2))

# Scatter plots
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.scatter(x_syst1[:-1],x_syst1[1:],s=5)
plt.title('System generator 1')
plt.xlabel('x_i')
plt.ylabel('x_i+1')
plt.subplot(1,2,2)
plt.scatter(x_syst2[:-1],x_syst2[1:],s=5)
plt.title('System generator 2')
plt.xlabel('x_i')
plt.ylabel('x_i+1')
plt.tight_layout()
plt.show()

# Runs test
z_syst1=x_syst1>=M_1/2
runs_syst1=1+np.sum(z_syst1[1:]!=z_syst1[:-1])
n1_syst1=float(np.sum(z_syst1))
n2_syst1=float(N_obs-n1_syst1)
E_runs_syst1=1+2*n1_syst1*n2_syst1/N_obs
V_runs_syst1=(2*n1_syst1*n2_syst1*(2*n1_syst1*n2_syst1-N_obs))/(N_obs**2*(N_obs-1))
Z_runs_syst1=(runs_syst1-E_runs_syst1)/np.sqrt(V_runs_syst1)
if abs(Z_runs_syst1)<1.96:
    print('Runs test system generator 1: null hypothesis not rejected, Z= '+str(Z_runs_syst1))
else:
    print('Runs test system generator 1: null hypothesis rejected, Z= '+str(Z_runs_syst1))

z_syst2=x_syst2>=M_2/2
runs_syst2=1+np.sum(z_syst2[1:]!=z_syst2[:-1])
n1_syst2=float(np.sum(z_syst2))
n2_syst2=float(N_obs-n1_syst2)
E_runs_syst2=1+2*n1_syst2*n2_syst2/N_obs
V_runs_syst2=(2*n1_syst2*n2_syst2*(2*n1_syst2*n2_syst2-N_obs))/(N_obs**2*(N_obs-1))
Z_runs_syst2=(runs_syst2-E_runs_syst2)/np.sqrt(V_runs_syst2)
if abs(Z_runs_syst2)<1.96:
    print('Runs test system generator 2: null hypothesis not rejected, Z= '+str(Z_runs_syst2))
else:
    print('Runs test system generator 2: null hypothesis rejected, Z= '+str(Z_runs_syst2))

# Correlation test
rho_syst1=np.corrcoef(x_syst1[:-h],x_syst1[h:])[0,1]
Z_corr_syst1=rho_syst1*np.sqrt(N_obs)
if abs(Z_corr_syst1)<1.96:
    print('Correlation test system generator 1: null hypothesis not rejected, rho= '+str(rho_syst1))
else:
    print('Correlation test system generator 1: null hypothesis rejected, rho= '+str(rho_syst1))

rho_syst2=np.corrcoef(x_syst2[:-h],x_syst2[h:])[0,1]
Z_corr_syst2=rho_syst2*np.sqrt(N_obs)
if abs(Z_corr_syst2)<1.96:
    print('Correlation test system generator 2: null hypothesis not rejected, rho= '+str(rho_syst2))
else:
    print('Correlation test system generator 2: null hypothesis rejected, rho= '+str(rho_syst2))