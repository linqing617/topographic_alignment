%% =========================================================================
% SCRIPT: Construction of the Topographic Alignment (TA) Framework
% =========================================================================
%
% PURPOSE:
% Construct subject-level Topographic Alignment (TA) profiles and the
% corresponding Topographic Alignment Network (TAN) from structural MRI
% and reference functional maps (e.g., Neurosynth meta-analytic maps).
%
% WORKFLOW:
%   1. Load a cortical parcellation (e.g., Schaefer-400).
%   2. Load reference functional term maps in the same anatomical space.
%   3. For each subject, compute within each ROI the voxel-wise cosine
%      similarity between the grey matter pattern and each reference map,
%      yielding the ROI-by-term TA matrix (first-order representation).
%   4. Compute ROI-by-ROI cosine similarity between TA profiles to obtain
%      the subject-level Topographic Alignment Network (TAN).
%
% REQUIREMENTS:
%   - DPABI / DPARSF toolbox (y_ReadAll)
%   - Statistics and Machine Learning Toolbox
%
% INPUT:
%   - Cortical parcellation (NIfTI)
%   - Subject grey matter maps
%   - Reference term maps (e.g., Neurosynth)
%
% OUTPUT:
%   - first_order_net : ROI × term × subject TA matrix
%   - s400_TAN        : ROI × ROI × subject TAN
%
%% =========================================================================
clear; clc;

%% ------------------------- USER-DEFINED PATHS --------------------------
%  >>> Please edit the paths below to match your own local directory <<<

% Root directory of this project (contains helper functions, e.g. topographic_alignment.m)
project_dir = '/path/to/your/project';                

% Path to the brain parcellation file (ROI mask), resliced to the same
% space/resolution as the subject data and the term maps.
% Example used here: Schaefer 2018, 400 parcels, 7 networks, MNI152 1mm.
parcellation_file = fullfile(project_dir, ...
    'reslice__Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii');

% Directory containing subject-level T1 structural data (e.g., HCP raw T1s,
% or their SPM-preprocessed / smoothed grey matter segmentations "s8*").
% Each subject should have its own subfolder containing a 'mri' folder
% with the smoothed grey matter image (filename starting with 's8').
subject_t1_dir = '/path/to/your/T1_data';              

% Directory containing the reference/term maps (e.g., Neurosynth maps),
% already resliced to match the parcellation space/resolution.
% Expected structure: one subfolder per term, each containing a single .nii
neuro_dir = '/path/to/your/term_maps';                 

% Where to save the outputs
output_dir = fullfile(project_dir, 'results');
if ~exist(output_dir, 'dir'); mkdir(output_dir); end

% -------------------------------------------------------------------------

addpath(project_dir);   % make sure function topographic_alignment.m is on the path

%% -------------------- Step 1: Load ROI parcellation ---------------------
t1_mask     = reshape(y_ReadAll(parcellation_file), [], 1);
t1_mask_ind = t1_mask(t1_mask > 0);      % ROI labels at in-mask voxels only
roi         = unique(t1_mask_ind);       % unique ROI IDs, e.g. 1:400

%% -------------------- Step 2: Load subject list & term maps --------------
files = dir(subject_t1_dir);
files(1:2, :) = [];                      % remove '.' and '..'

neuro_files = dir(neuro_dir);
neuro_files(1:2, :) = [];                % remove '.' and '..'

n_voxel = length(t1_mask_ind);
n_term  = length(neuro_files);
term_all = zeros(n_voxel, n_term);

for i = 1:n_term
    fprintf('Loading term map %04d / %04d\n', i, n_term);
    nii_file = dir(fullfile(neuro_files(i).folder, neuro_files(i).name, '*.nii'));
    nii_dat  = reshape(y_ReadAll(fullfile(nii_file(1).folder, nii_file(1).name)), [], 1);
    term_all(:, i) = nii_dat(t1_mask > 0);
end

%% ------------- Step 3: Build first-order matrix --------------
% For every subject, project their grey matter pattern onto each ROI and
% compute cosine similarity with every term map (within that ROI).

n_sub = length(files);
n_roi = length(roi);

wrong_sub       = zeros(n_sub, 1);              % subjects that failed
first_order_net = zeros(n_roi, n_term, n_sub);  % ROI x term x subject

for i = 1:n_sub
    fprintf('Building first-order network: subject %04d / %04d\n', i, n_sub);
    try
        % Expect a single smoothed grey matter image starting with 's8'
        % inside <subject_folder>/mri/
        mri_file = dir(fullfile(files(i).folder, files(i).name, 'mri', 's8*'));
        mri_dat  = reshape(y_ReadAll(fullfile(mri_file.folder, mri_file.name)), [], 1);

        sub_t1_net = topographic_alignment(mri_dat(t1_mask > 0), term_all, roi, t1_mask_ind);
        first_order_net(:, :, i) = sub_t1_net;
    catch ME
        warning('Subject %04d failed: %s', i, ME.message);
        wrong_sub(i) = i;
    end
end

%% ------------- Step 4: Build second-order ROI-by-ROI network -------------
% For each subject, compute ROI-ROI similarity from the first-order
% ROI-by-term matrix (i.e., how similar two ROIs' "TA score profiles" are).

s400_TAN = zeros(n_roi, n_roi, n_sub);

for i = 1:n_sub
    fprintf('Building second-order network: subject %04d / %04d\n', i, n_sub);
    sub_first_order = first_order_net(:, :, i);
    sub_net = squareform(1 - pdist(sub_first_order, 'cosine'));
    s400_TAN(:, :, i) = sub_net;
end
