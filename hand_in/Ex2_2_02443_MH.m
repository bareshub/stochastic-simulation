clc
clear all
close all
% -----------------------------------------------------------------------
% ---------------------------- EXERCISE 2 ---------------------------
% -----------------------------------------------------------------------

% ---- Part 2: sampling from discrete distributions: DIFFERENT METHODS ---
% ------------------------------------------------------------------------

N_obs=10000;
x=[1,2,3,4,5,6]; 
p=[7/48, 5/48, 1/8, 1/16, 1/4, 5/16]; 
P=cumsum(p);

% Crude method sampling 
M0=whos; mem0 = sum([M0.bytes])/1024^2; 
tic

U=rand(1,N_obs); X_crude=zeros(1,N_obs);
for i = 1:length(p)                      % Checking for all elements in U if each U(j) is smaller 
    X_crude(U<=P(i) & X_crude==0)=x(i);  % than p(i) create a vector with elements containing 
                                         % the corresponding index of x in the six-point distrib
end    

t_crude=toc;                            % Computationally efficiency: time
M1=whos; mem1 = sum([M1.bytes])/1024^2; 
mem_crude=mem1-mem0;                    % Memory req in MB: all var created         

% Rejection method
M0=whos; mem0 = sum([M0.bytes])/1024^2; 
tic

c=max(p)+0.1;            % set a value of c greater than all p_i
i=1; n=1;           
while i<=N_obs
   I=floor(max(x)*rand)+1;
   if rand<=p(I)/c
      X_rej(i)=I; i=i+1;
   end
end                           

t_rej=toc;                            
M1=whos; mem1 = sum([M1.bytes])/1024^2; 
mem_rej=mem1-mem0;                    

% Alias method
M0=whos; mem0 = sum([M0.bytes])/1024^2; 
tic

F = [7/8, 5/8, 3/4, 3/8, 1, 7/8]; A = [5, 6, 5, 6, 5, 5]; % Alias Tables
I=floor(max(x)*rand(1,N_obs))+1; U_2=rand(1,N_obs);       % Int and U_2 generation
X_alias=I; X_alias(U_2>F(I))=A(I(U_2>F(I)));              % X is assigned to 1 and  if U_2>F(i), to A

t_alias=toc;                           
M1=whos; mem1 = sum([M1.bytes])/1024^2; 
mem_alias=mem1-mem0;                   

% ---- Part 3: comparing the three generation methods ---------------
% -------------------------------------------------------------------

% Computational efficiency: tic-toc command (done above)

% Ease of algorithm implementation

% Memory requirements: whos command (done above)

% Accuracy of the generated distribution: Kolmogorov-Smirnov statistical test
Counts_crude=histcounts(X_crude,1:(max(x)+1),'Normalization','cumcount');  % CDF
Fn_crude=Counts_crude/N_obs;
Dn_crude=max(abs(Fn_crude-P)); % Test statistic                     
Dn_crude_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*Dn_crude;       
if Dn_crude_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('KS test crude method: null hypotesis not rejected')
end

Counts_rej=histcounts(X_rej,1:(max(x)+1),'Normalization','cumcount');  % CDF
Fn_rej=Counts_rej/N_obs;
Dn_rej=max(abs(Fn_rej-P)); % Test statistic                     
Dn_rej_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*Dn_rej;       
if Dn_rej_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('KS test rejection method: null hypotesis not rejected')
end

Counts_alias=histcounts(X_alias,1:(max(x)+1),'Normalization','cumcount');  % CDF
Fn_alias=Counts_alias/N_obs;
Dn_alias=max(abs(Fn_alias-P)); % Test statistic                     
Dn_alias_adjusted=(sqrt(N_obs) + 0.12 + 0.11/sqrt(N_obs))*Dn_alias;       
if Dn_alias_adjusted < 1.480 % Critical value for significance level alpha=0.025
    disp('KS test alias method: null hypotesis not rejected')
end

% --------------------------- Histograms ------------------------------

figure 

subplot(1,3,1)
histogram(X_crude,1:(max(x)+1),'Normalization','cumcount')
hold on
stairs(1:max(x),N_obs*P) 
title('Direct Method')
xlabel('X')
ylabel('Counts')

subplot(1,3,2)
histogram(X_rej,1:(max(x)+1),'Normalization','cumcount')
hold on
stairs(1:max(x),N_obs*P) 
title('Rejection method')
xlabel('x')
ylabel('Counts')

subplot(1,3,3)
histogram(X_alias,1:(max(x)+1),'Normalization','cumcount')
hold on
stairs(1:max(x),N_obs*P) 
title('Alias method')
xlabel('x')
ylabel('Counts')

fprintf('Time crude method: %.4f seconds\n', t_crude);
fprintf('Time rejection method: %.4f seconds\n', t_rej);
fprintf('Time taken for alias method: %.4f seconds\n', t_alias);
fprintf('Memory used by crude method: %.4f MB\n', mem_crude);
fprintf('Memory used by rejection method: %.4f MB\n', mem_rej);
fprintf('Memory used by alias method: %.4f MB\n', mem_alias);
fprintf('D_n_adjusted crude : %.4f\n', Dn_crude_adjusted);
fprintf('D_n_adjusted rej: %.4f\n', Dn_rej_adjusted);
fprintf('D_n_adjusted alias: %.4f\n', Dn_alias_adjusted);