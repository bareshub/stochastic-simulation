clc
clear all
close all

% ---------------------------- EXERCISE 2 -------------------------
% ---- Part 1: sampling from geometric distributions: CRUDE METHOD ----

p1=0.1; p2=0.5; p3=0.9; 
N_obs=10000;

% ------------ Crude method sampling-------------
X1_c=floor(log(rand(1,N_obs))./log(1-p1))+1;
X2_c=floor(log(rand(1,N_obs))./log(1-p2))+1;
X3_c=floor(log(rand(1,N_obs))./log(1-p3))+1;

% --------------- Histogram plots ---------------
% Plots: experimental cumulative histogram (empirical distribution function multiplied by N_obs) 
%        theoretical cumulative histogram (theoretical distribution function multiplied by N_obs)

figure 

subplot(1,3,1)
histogram(X1_c,0:max(X1_c),'Normalization','cumcount')
hold on
stairs(1:max(X1_c),N_obs*(1-(1-p1).^(1:max(X1_c)))) 
title('p=0.1')
xlabel('X')
ylabel('Counts')

subplot(1,3,2)
histogram(X2_c,0:max(X2_c),'Normalization','cumcount')                                  
hold on
stairs(1:max(X2_c),N_obs*(1-(1-p2).^(1:max(X2_c)))) 
title('p=0.5')
xlabel('X')
ylabel('Counts')

subplot(1,3,3)
histogram(X3_c,0:max(X3_c),'Normalization','cumcount')                                    
hold on
stairs(1:max(X3_c),N_obs*(1-(1-p3).^(1:max(X3_c)))) 
title('p=0.9')
xlabel('X')
ylabel('Counts')

% Kolmogorov-Smirnov statistical test
F1_n=histcounts(X1_c,1:(max(X1_c)+1),'Normalization','cumcount')/N_obs; % CDF
D1_n=max(abs(F1_n-(1-(1-p1).^(1:max(X1_c))))); % Test statistic                     
D1_n_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*D1_n;       
if D1_n_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('Null hypothesis not rejected for p=0.1')
end

F2_n=histcounts(X2_c,1:(max(X2_c)+1),'Normalization','cumcount')/N_obs; 
D2_n=max(abs(F2_n-(1-(1-p2).^(1:max(X2_c)))));
D2_n_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*D2_n;
if D2_n_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('Null hypothesis not rejected for p=0.5')
end

F3_n=histcounts(X3_c,(1:max(X3_c)+1),'Normalization','cumcount')/N_obs; 
D3_n=max(abs(F3_n-(1-(1-p3).^(1:max(X3_c)))));
D3_n_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*D3_n;
if D3_n_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('Null hypothesis not rejected for p=0.9')
end