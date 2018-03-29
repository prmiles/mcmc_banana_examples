% Banana example
% 
% This techncal example constructs a non Gaussian target
% distribution by twisting two first dimensions of Gaussian
% distribution. The Jacobian of the transformation is 1, so it is
% easy to calculate the right probability regions for the banana
% and study different adaptive methods.
% We demonstrate sampling using the following algorithms:
%     MH - Metropolis Hastings
%     AM - Adaptive Metropolis
%     DR - Delayed Rejection
%     DRAM - Delayed Rejection Adaptive Metropolis

% setup workspace
clear; close all; clc;

% addpath to mcmcstat package
addpath('mcmcstat/');

% 'banana' sum-of-squares
bananafun = @(x,a,b) [a.*x(:,1),x(:,2)./a-b.*((a.*x(:,1)).^2+a^2),x(:,3:end)];
bananainv = @(x,a,b) [x(:,1)./a,x(:,2).*a+a.*b.*(x(:,1).^2+a^2),x(:,3:end)];
bananass  = @(x,d) bananainv(x-d.mu,d.a,d.b)*d.lam*bananainv(x-d.mu,d.a,d.b)';

a = 1; b = 1;         % banana parameters

npar = 2;             % number of unknowns
rho  = 0.9;            % target correlation
sig  = eye(npar); sig(1,2) = rho; sig(2,1) = rho;
lam  = inv(sig);       % target precision
mu   = zeros(1,npar);  % center

% Define data structure and model parameters array
% the data structure and parameters
data = struct('mu',mu,'a',a,'b',b,'lam',lam);
for i=1:npar
  params{i} = {sprintf('p_%d',i),0};
end

% Define model structure
model.ssfun     = bananass;
model.N         = 1;

% Define options structure with elements common to each sampling algorithm
options.nsimu   = 2000;
options.qcov    = eye(npar)*5; % [initial] proposal covariance

% Run Simulations
% Metropolis Hasting (MH)
options.method  = 'mh';
[mh.results,mh.chain] = mcmcrun(model,data,params,options);
% Adaptive Metropolis (AM)
options.method  = 'am';
options.adaptint = 100; % adaptation interval
[am.results,am.chain] = mcmcrun(model,data,params,options);
% Delayed Rejection (DR)
options.method  = 'dr';
options.ntry = 2; % number of DR steps
[dr.results,dr.chain] = mcmcrun(model,data,params,options);
% Delayed Rejection Adaptive Metropolis (DRAM)
options.method  = 'dram';
options.adaptint = 100; % adaptation interval
options.ntry = 2; % number of DR steps
[dram.results,dram.chain] = mcmcrun(model,data,params,options);

% Chain panel
figure(1); clf
mcmcplot(dram.chain,[],dram.results.names,'chainpanel')

% Pairwise Correlation
figure(2); clf
mcmcplot(dram.chain,[1,2],dram.results.names,'pairs',0)

% Print acceptance statistics
fprintf('\n----------------\n')
fprintf('MH: Number of accepted runs: %i out of %i (%4.2f%s)\n',length(unique(mh.chain(:,1))), options.nsimu, 100*(1-mh.results.rejected),'%');
fprintf('AM: Number of accepted runs: %i out of %i (%4.2f%s)\n',length(unique(am.chain(:,1))), options.nsimu, 100*(1-am.results.rejected),'%');
fprintf('DR: Number of accepted runs: %i out of %i (%4.2f%s)\n',length(unique(dr.chain(:,1))), options.nsimu, 100*(1-dr.results.rejected),'%');
fprintf('DRAM: Number of accepted runs: %i out of %i (%4.2f%s)\n',length(unique(dram.chain(:,1))), ...
    options.nsimu, 100*(1-dram.results.rejected),'%');

% save results for comparison
save('matlab_banana','mh','am','dr','dram');