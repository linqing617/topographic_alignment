

%% =========================================================================
% SCRIPT: Principal Gradient Analysis of the Topographic Alignment Network
% =========================================================================
%
% PURPOSE:
% Compute the principal gradient of the group-level Topographic Alignment
% Network (TAN) and evaluate its correspondence with the cortical
% sensory-association hierarchy using spin permutation testing.
%
% WORKFLOW:
%   1. Average the TAN across subjects.
%   2. Compute diffusion map embedding using BrainSpace.
%   3. Extract the principal gradient of the TAN.
%   4. Correlate the gradient with the sensory-association (S-A) axis.
%   5. Assess statistical significance using spin permutation testing.
%
% REQUIREMENTS:
%   - BrainSpace (MATLAB)
%   - Schaefer-400 parcellation
%   - Precomputed spin permutations
%
% INPUT:
%   - Schaefer_400_TAN.mat
%   - s400_S-A_axis.csv
%   - 1000Null_Schaefer400.csv
%
% OUTPUT:
%   - Gradient_TAN
%   - rank_TAN
%   - p (spin-test empirical p-value)
%% =========================================================================
clear; clc;

%% ------------------------------------------------------------------------
% Load Topographic Alignment Network (TAN)
% -------------------------------------------------------------------------

load('Schaefer_400_TAN.mat');

% Average TAN across subjects
TAN = nanmean(s400_TAN, 3);


%% ------------------------------------------------------------------------
% Compute the principal gradient of TAN
% -------------------------------------------------------------------------

% Load Schaefer-400 parcellation
labeling = load_parcellation('schaefer', 400);
labeling = labeling.schaefer_400;

% Load Conte69 cortical surface
[surf_lh, surf_rh] = load_conte69();

% Diffusion map embedding using cosine similarity kernel
G = GradientMaps('kernel', 'cs', 'approach', 'dm');

% Fit gradient model
G_p = G.fit(TAN);

% Extract the first (principal) gradient
Gradient_TAN = G_p.gradients{1}(:,1);


%% ------------------------------------------------------------------------
% Correlation with the sensory-association (S-A) axis
% -------------------------------------------------------------------------

% Load cortical S-A axis ranking
rank = readtable('s400_S-A_axis.csv');
rank = rank.rank;

% Spearman correlation between TAN gradient and S-A axis
rank_TAN = corr(rank, Gradient_TAN, 'type', 'Spearman');


%% ------------------------------------------------------------------------
% Spin permutation test
% -------------------------------------------------------------------------

% Load precomputed spin permutations
spin = readmatrix('1000Null_Schaefer400.csv');

null = zeros(1000,1);

% Compute null distribution by rotating parcel labels
for i = 1:1000
    null(i) = corr(Gradient_TAN(spin(:,i)), ...
                   rank, ...
                   'type', 'Spearman');
end

% Two-sided empirical p-value
p = (sum(abs(null - mean(null)) > abs(rank_TAN - mean(null))) + 1) ...
      / (length(null) + 1);