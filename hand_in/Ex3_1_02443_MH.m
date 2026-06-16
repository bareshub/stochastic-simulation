clc
clear all
close all

% ---------------------------- EXERCISE 3 ------------------------------
% --- Part 1: sampling from different distributions: INVERSE METHOD  ---

N_obs=100; lambda=1; mu=0; sigma=1; beta=1; k=[2.05,2.5,3,4]; 
U=rand(1,N_obs); U_2=rand(1,N_obs);

% ------------ Inverse method sampling-------------
X_exp=-log(U)/lambda;
X_gauss=mu+sigma*sqrt(-2*log(U)).*cos(2*pi*U_2);
X_pareto_1=beta*(U.^(-1/k(1))-1);
X_pareto_2=beta*(U.^(-1/k(2))-1);
X_pareto_3=beta*(U.^(-1/k(3))-1);
X_pareto_4=beta*(U.^(-1/k(4))-1);


% Accuracy of the generated distribution: Kolmogorov-Smirnov statistical test

% Now we require points of evaluation, we take the points in which F_n(x) is 
% defined: that is the easy computation and close to the KS method. 

x=sort(X_exp);
Counts_exp=histcounts(X_exp,[-inf x],'Normalization','cumcount');
Fn_exp=Counts_exp/N_obs;
Dn_exp=max(abs(Fn_exp-(1-exp(-lambda*x)))); % Test statistic
Dn_exp_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*Dn_exp;
if Dn_exp_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('KS test exponential distribution: null hypotesis not rejected')
end

x=sort(X_gauss);
Counts_gauss=histcounts(X_gauss,[-inf x],'Normalization','cumcount');
Fn_gauss=Counts_gauss/N_obs;
Dn_gauss=max(abs(Fn_gauss-normcdf(x,mu,sigma))); % Test statistic
Dn_gauss_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*Dn_gauss;
if Dn_gauss_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('KS test Gaussian distribution: null hypotesis not rejected')
end

x=sort(X_pareto_1);
Counts_pareto_1=histcounts(X_pareto_1,[-inf x],'Normalization','cumcount');
Fn_pareto_1=Counts_pareto_1/N_obs;
Dn_pareto_1=max(abs(Fn_pareto_1-(1-(beta./(x+beta)).^k(1)))); % Test statistic
Dn_pareto_1_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*Dn_pareto_1;
if Dn_pareto_1_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('KS test Pareto distribution k=2.05: null hypotesis not rejected')
end

x=sort(X_pareto_2);
Counts_pareto_2=histcounts(X_pareto_2,[-inf x],'Normalization','cumcount');
Fn_pareto_2=Counts_pareto_2/N_obs;
Dn_pareto_2=max(abs(Fn_pareto_2-(1-(beta./(x+beta)).^k(2)))); % Test statistic
Dn_pareto_2_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*Dn_pareto_2;
if Dn_pareto_2_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('KS test Pareto distribution k=2.5: null hypotesis not rejected')
end

x=sort(X_pareto_3);
Counts_pareto_3=histcounts(X_pareto_3,[-inf x],'Normalization','cumcount');
Fn_pareto_3=Counts_pareto_3/N_obs;
Dn_pareto_3=max(abs(Fn_pareto_3-(1-(beta./(x+beta)).^k(3)))); % Test statistic
Dn_pareto_3_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*Dn_pareto_3;
if Dn_pareto_3_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('KS test Pareto distribution k=3: null hypotesis not rejected')
end

x=sort(X_pareto_4);
Counts_pareto_4=histcounts(X_pareto_4,[-inf x],'Normalization','cumcount');
Fn_pareto_4=Counts_pareto_4/N_obs;
Dn_pareto_4=max(abs(Fn_pareto_4-(1-(beta./(x+beta)).^k(4)))); % Test statistic
Dn_pareto_4_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*Dn_pareto_4;
if Dn_pareto_4_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('KS test Pareto distribution k=4: null hypotesis not rejected')
end

% --------------------------- Histograms ------------------------------

figure 

subplot(1,2,1)
histogram(X_exp,'Normalization','pdf')
hold on
x=linspace(min(X_exp),max(X_exp),100);
plot(x,lambda*exp(-lambda*x))
title('Exponential distribution')
xlabel('x')
ylabel('Density')

subplot(1,2,2)
histogram(X_gauss,'Normalization','pdf')
hold on
x=linspace(min(X_gauss),max(X_gauss),100);
plot(x,normpdf(x,mu,sigma))
title('Gaussian distribution')
xlabel('x')
ylabel('Density')

figure

x=linspace(0,4,10);
x1=linspace(0,4,100);

subplot(2,2,1)
histogram(X_pareto_1,[-inf x],'Normalization','pdf')
hold on
plot(x1,k(1)*beta^k(1)./(x1+beta).^(k(1)+1))
title('Pareto distribution, k=2.05')
xlabel('x')
ylabel('Density')
xlim([0 4])
ylim([0 2])

subplot(2,2,2)
histogram(X_pareto_2,[-inf x],'Normalization','pdf')
hold on
plot(x1,k(2)*beta^k(2)./(x1+beta).^(k(2)+1))
title('Pareto distribution, k=2.5')
xlabel('x')
ylabel('Density')
xlim([0 4])
ylim([0 2])

subplot(2,2,3)
histogram(X_pareto_3,[-inf x],'Normalization','pdf')
hold on
plot(x1,k(3)*beta^k(3)./(x1+beta).^(k(3)+1))
title('Pareto distribution, k=3')
xlabel('x')
ylabel('Density')
xlim([0 4])
ylim([0 2])

subplot(2,2,4)
histogram(X_pareto_4,[-inf x],'Normalization','pdf')
hold on
plot(x1,k(4)*beta^k(4)./(x1+beta).^(k(4)+1))
title('Pareto distribution, k=4')
xlabel('x')
ylabel('Density')
xlim([0 4])
ylim([0 2])

% -------- Part 2: Pareto mean and variance estimations ---------

mean_1=sum(X_pareto_1)/N_obs; var_1=sum((X_pareto_1-mean_1).^2)/N_obs;
mean_2=sum(X_pareto_2)/N_obs; var_2=sum((X_pareto_2-mean_2).^2)/N_obs;
mean_3=sum(X_pareto_3)/N_obs; var_3=sum((X_pareto_3-mean_3).^2)/N_obs;
mean_4=sum(X_pareto_4)/N_obs; var_4=sum((X_pareto_4-mean_4).^2)/N_obs;

% -------- Part 3: Confidence intervals and coverage for the gaussian distribution ------

N_int=100; n=10; mu=0; sigma=1; 
alpha=0.05; z_crit=1.9600; chi2_low=3.2470; chi2_high=20.4832;

Conf_mean=zeros(N_int,2); Conf_var=zeros(N_int,2); N_mean=0; N_var=0;

for i=1:N_int
    U1=rand(1,n); U2=rand(1,n); X=sqrt(-2*log(U1)).*cos(2*pi*U2);

    X_mean=sum(X)/n;
    Conf_mean(i,1)=X_mean-z_crit*sigma/sqrt(n);
    Conf_mean(i,2)=X_mean+z_crit*sigma/sqrt(n);
    if Conf_mean(i,1)<=mu && mu<=Conf_mean(i,2)
        N_mean=N_mean+1;
    end

    S=sum((X-mu).^2);
    Conf_var(i,1)=S/chi2_high;
    Conf_var(i,2)=S/chi2_low;

    if Conf_var(i,1)<=sigma^2 && sigma^2<=Conf_var(i,2)
        N_var=N_var+1;
    end
end

coverage_mean=N_mean/N_int;
coverage_var=N_var/N_int;

% ---------- Part 4: Pareto method sampling using composition method -----

N_obs=100; mu=1; beta=1; k=1; U=rand(1,N_obs); U_2=rand(1,N_obs); 
Y=-log(U)/mu; X_comp=-log(U_2)./Y; % COMPOSITION METHOD k=1, beta=1

U_3=rand(1,N_obs); X_inv=beta*(U_3.^(-1/k)-1); % INVERSE METHOD k=1, beta=1

x=sort(X_inv);
Counts_inv=histcounts(X_inv,[-inf x],'Normalization','cumcount');
Fn_inv=Counts_inv/N_obs;
Dn_inv=max(abs(Fn_inv-(1-(beta./(x+beta)).^k))); % Test statistic
Dn_inv_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*Dn_inv;
if Dn_inv_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('KS test Pareto inverse method: null hypotesis not rejected')
end

x=sort(X_comp);
Counts_comp=histcounts(X_comp,[-inf x],'Normalization','cumcount');
Fn_comp=Counts_comp/N_obs;
Dn_comp=max(abs(Fn_comp-(1-(beta./(x+beta)).^k))); % Test statistic
Dn_comp_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*Dn_comp;
if Dn_comp_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('KS test Pareto composition method: null hypotesis not rejected')
end

% --------------------------- Histograms ------------------------------

x=linspace(0,100,100);
f_theo=beta./(x+beta).^2;

figure
histogram(X_inv,[-inf x],'Normalization','pdf')
hold on
histogram(X_comp,[-inf x],'Normalization','pdf')
hold on
plot(x,f_theo)
title('Pareto distribution: inverse and composition methods')
xlabel('x')
ylabel('Density')
legend('Inverse method','Composition method','Theoretical density')