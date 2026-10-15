%% =========================================================================
% SCRIPT: PLS Modeling of Gene Expression and Cortical Gradient Coupling
% =========================================================================
%
% PURPOSE:
% Investigate the association between regional gene expression profiles
% (AHBH transcriptomic data) and the principal cortical gradient derived
% from the Topographic Alignment Network (TAN) using Partial Least Squares
% (PLS) regression and spatial permutation testing.
%
% This analysis tests whether spatial patterns of gene expression explain
% variance in macroscale cortical organization captured by the TAN gradient.
%
% WORKFLOW:
%   1. Load regional AHBH gene expression matrix (Schaefer-400 atlas).
%   2. Extract the principal gradient from the group-level TAN using
%      diffusion map embedding.
%   3. Fit PLS regression to identify latent gene expression components
%      explaining variance in the cortical gradient.
%   4. Evaluate statistical significance using spin permutation tests
%      (spatially constrained null models).
%   5. Perform bootstrap resampling to estimate gene weight stability and
%      compute Z-scores for gene importance ranking.
%   6. Identify positively and negatively weighted gene sets associated
%      with gradient organization.
%
% REQUIREMENTS:
%   - Statistics and Machine Learning Toolbox (plsregress, corr, randsample)
%   - mafdr (Bioinformatics Toolbox)
%   - BrainSpace (for gradient computation,https://github.com/MICA-MNI/BrainSpace)
%   - Precomputed spin permutation file (1000Null_Schaefer400.csv)
%
% INPUT:
%   - s400_AHBH_expression.csv   : gene expression matrix (ROI × genes)
%   - Schaefer_400_TAN.mat       : topographic alignment network
%   - 1000Null_Schaefer400.csv   : spin permutations
%
% OUTPUT:
%   - PLS_stats_table            : variance + correlation + spin p-values
%   - PLS1_pos_genes             : positively associated genes (high Z)
%   - PLS1_neg_genes             : negatively associated genes (low Z)
%   - gene weight stability (bootstrap Z-scores)
%% =========================================================================

clear;clc


%%%%% get AHBH gene matrix %%%%%%%%%  

%https://abagen.readthedocs.io/en/stable/cli.html#the-abagen-command 
% python code 
%import abagen
%expression = abagen.get_expression_data(atlas,missing='interpolate')

%% ------------------------ Load gene expression ---------------------------
% AHBH gene expression matrix (ROI × gene)
AHBH_expression = readmatrix('s400_AHBH_expression.csv');
left_hemi = AHBH_expression(1:200,:);

% Load TAN and compute group-average network
load('Schaefer_400_TAN.mat')
TAN = nanmean(s400_TAN,3); % Schefer400_TAN.csv

% Compute principal gradient of TAN
G = GradientMaps('kernel','cs','approach','dm');
G_p = G.fit(TAN);
Gradient_TAN = G_p.gradients{1}(:,1);

% Restrict to left hemisphere
left_gradient = Gradient_TAN(1:200);


%% ------------------------ PLS regression --------------------------------
% Predict cortical gradient from gene expression profiles

X = zscore(left_hemi);
Y = zscore(left_gradient);

dim = 2;
[XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = plsregress(X,Y,dim);

% Explained variance
temp = cumsum(100 * PCTVAR(2,1:dim));
Rsquared1   = temp(1);
Rsquared2   = temp(2) - temp(1);
Rsquared1_2 = temp(dim);

% PLS1 PLS2  score Y corr

[R0,p0]=corr([XS(:,1),XS(:,2)],left_gradient);  
[Rs0,ps0]=corr([XS(:,1),XS(:,2)],left_gradient,"type","Spearman");

% spin permutation
spin=readmatrix('1000Null_Schaefer400.csv');
spin=spin(1:200,:);
nperm=1000;
for i=1:nperm

    [XLr,YLr,XSr,YSr, BETAr, PCTVARr, MSEr, statsr]=plsregress(X,Y(spin(:,i)),dim);
    temp=cumsum(100*PCTVARr(2,1:dim));
    Rsq1(i)=temp(1);
    Rsq2(i)=temp(dim)-temp(1);
    Rsq1_2(i)=temp(dim);
    [R,p]=corr([XSr(:,1),XSr(:,2)],left_gradient(spin(:,i)));
    [Rs,ps]=corr([XSr(:,1),XSr(:,2)],left_gradient(spin(:,i)),"type","Spearman");
    R_spin(i,:)=R;
    Rs_spin(i,:)=Rs;
end

% calculate p

p1 = (1 + length(find(abs(Rsq1 - mean(Rsq1)) > abs(Rsquared1 - mean(Rsq1))))) / (n_perm + 1);
p2 = (1 + length(find(abs(Rsq2 - mean(Rsq2)) > abs(Rsquared2 - mean(Rsq2))))) / (n_perm + 1);
p1_2 = (1 + length(find(abs(Rsq1_2 - mean(Rsq1_2)) > abs(Rsquared1_2 - mean(Rsq1_2))))) / (n_perm + 1);

corr_p1 = (1 + length(find(abs(R_spin(:,1) - mean(R_spin(:,1))) > abs(R0(1) - mean(R_spin(:,1)))))) / (n_perm + 1);
corr_sp1 = (1 + length(find(abs(Rs_spin(:,1) - mean(Rs_spin(:,1))) > abs(Rs0(1) - mean(Rs_spin(:,1)))))) / (n_perm + 1);
corr_p2 = (1 + length(find(abs(R_spin(:,2) - mean(R_spin(:,2))) > abs(R0(2) - mean(R_spin(:,2)))))) / (n_perm + 1);
corr_sp2 = (1 + length(find(abs(Rs_spin(:,2) - mean(Rs_spin(:,2))) > abs(Rs0(2) - mean(Rs_spin(:,2)))))) / (n_perm + 1);


% results
PLS_stats=[Rsquared1,p1,Rsquared2,p2,Rsquared1_2,p1_2,R0(1),corr_p1,R0(2),corr_p2,Rs0(1),corr_sp1,Rs0(2),corr_sp2]';
PLS_explain=["PLS1 accounted for variations in  gradient";
    "p spin for PLS1";
    "PLS2 accounted for variations in  gradient";
    "p spin for PLS2";
    "PLS1+PLS2 accounted for variations in  gradient";
    "p spin for PLS1+PSL2";
    "correlation between PLS1 and gradient";
    "p spin for correlation";
    "correlation between PLS2 and gradient";
    "p spin for correlation";
    "spearman correlation between PLS1 and gradient";
    "p spin for spearman correlation";
    "spearman correlation between PLS2 and gradient";
    "p spin for spearman correlation"];
PLS_stats_table=[PLS_explain,PLS_stats];

%% ------------------------ Bootstrap gene stability -----------------------
% Estimate robustness of gene weights in PLS1 component

X = zscore(left_hemi);
Y = zscore(left_gradient);

AHBH = readtable('s400_AHBH_expression.csv');
genename = AHBH.Properties.VariableNames';

[PLS1w, x1] = sort(stats.W(:,1), 'descend');
PLSlids = genename(x1);

bootnum = 1000;
PLS1weights = [];

rng(1);

for i = 1:bootnum

    myresample = randsample(size(X,1), size(X,1), 1);
    Xr = X(myresample,:);
    Yr = Y(myresample,:);

    [~,~,~,~,~,~,~, stats] = plsregress(Xr, Yr, dim);

    newW = stats.W(:,1);
    newW = newW(x1);

    % align sign ambiguity
    if corr(PLS1w, newW) < 0
        newW = -newW;
    end

    PLS1weights = [PLS1weights, newW];
end

% Compute stability-based Z-score for each gene weight

PLS1sw = std(PLS1weights');

temp1 = PLS1w ./ PLS1sw';
[Z1, ind1] = sort(temp1, 'descend');

PlS1 = PLSlids(ind1);

% p-values and FDR correction
p = 2 * (1 - normcdf(abs(Z1)));
fdr_p = mafdr(p, 'BHFDR', true);

% Significant gene sets
PLS1_pos_genes1 = PlS1(Z1 > 5);
PLS1_neg_genes1 = PlS1(Z1 < -5);


