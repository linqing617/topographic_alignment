
% =============================================================================
% SCRIPT: AD-Related Topographic Alignment GLM Analysis
% =============================================================================
%
% PURPOSE:
% 1. Summarise demographic characteristics of the Control and AD groups.
% 2. Perform point-wise GLM analysis to identify group differences in TA score.
% 3. Apply Benjamini-Hochberg FDR correction to the point-wise p-values.
% 4. Classify significant ROI-term pairs into four patterns according to
%    the signs of the control-group mean and the group-effect t-statistic.
% 5. Quantify the prevalence and effect sizes of the four patterns across ROIs.
% 6. Export significant ROI-term pairs and their associated statistics.
%
% INPUT FILES:
%   - ADNI_first_cortex_combat.mat
%       Harmonised ROI-by-term data and demographic/covariate information.
%   - Schaefer400_7network_name.csv
%       Schaefer 400-parcel atlas labels.
%   - Term.csv
%       Term names corresponding to the columns of the ROI-by-term matrix.
%
% EXPECTED DATA:
%   - dat_harmonized: [nROI x nTerm x nSubjects] data matrix.
%   - info_cov_sorted: subject-level demographic and diagnostic information.
%
% OUTPUT FILES:
%   - AD_demo.csv
%       Demographic summary for the Control and AD groups.
%   - AD_Type1_PosInc.csv
%       Significant pairs with positive control means and positive group effects.
%   - AD_Type2_PosDec.csv
%       Significant pairs with positive control means and negative group effects.
%   - AD_Type3_NegWeak.csv
%       Significant pairs with negative control means and positive group effects.
%   - AD_Type4_NegStr.csv
%       Significant pairs with negative control means and negative group effects.
%   — AD_ROI_pattern_summary.csv
%       ROI-level summary of the four patterns, including mean t-statistics,
%       pattern proportions, mean effect-size indices, extreme effect-size
%       indices, and the overall proportion of significant ROI-term pairs.
% STATISTICAL ANALYSIS:
%   - Point-wise general linear models with diagnosis, age, and sex as predictors.
%   - Benjamini-Hochberg false discovery rate (FDR) correction.
%   - Significance threshold: FDR-adjusted p < 0.05.
%
% NOTE:
%   This script assumes that the subject order in dat_harmonized matches
%   the subject order in info_cov_sorted.
%
% =============================================================================


%% 1. Setup and Load Data

clear;
clc;

% Load harmonised data and subject-level covariate information.

load('ADNI_first_cortex_combat.mat');


%% 2. Summarise Demographic Characteristics


age = info_cov_sorted.Age;

sex_dummy = dummyvar(categorical(info_cov_sorted.Sex));
sex = sex_dummy(:, 2);

group_dummy = dummyvar(categorical(info_cov_sorted.ResearchGroup));
disease = group_dummy(:, 1);

% Extract data for the Control and AD groups.
% disease == 0: Control
% disease == 1: AD

control_net = dat_harmonized(:, :, disease == 0);
AD_net = dat_harmonized(:, :, disease == 1);



G = disease;
S = sex;
A = age;

name = {'Control', 'AD'};

for i = 0:1

    idx = G == i;

    n(i + 1, 1) = sum(idx);

    Age{i + 1, 1} = sprintf( ...
        '%.2f +/- %.2f', ...
        mean(A(idx)), std(A(idx)));

    % Sex distribution: female/male.
    Sex{i + 1, 1} = sprintf( ...
        '%d/%d', ...
        sum(S(idx) == 0), sum(S(idx) == 1));

end

T = table( ...
    name', n, Age, Sex, ...
    'VariableNames', {'Diagnosis', 'n', 'Age', 'Sex_F_M'});

writetable(T, 'AD_demo.csv');


%% 3. Point-Wise GLM Analysis

[n_rows, n_cols, n_subs] = size(dat_harmonized);

fprintf( ...
    'Starting point-wise GLM analysis (%d ROI-term pairs)...\n', ...
    n_rows * n_cols);

% Design matrix:
%   Predictor 1: Diagnosis (Control = 0, AD = 1)
%   Predictor 2: Age
%   Predictor 3: Sex

X = [disease, age, sex];

% Group sample sizes.
m1 = sum(disease == 0);
m2 = sum(disease == 1);

% Initialise statistical maps.
P_map_raw = nan(n_rows, n_cols);
T_map = nan(n_rows, n_cols);
D_map = nan(n_rows, n_cols);

tic;

for i = 1:n_rows

    disp(i);

    for j = 1:n_cols

        % Extract data for the current ROI-term pair.
        y = squeeze(dat_harmonized(i, j, :));

        % Skip pairs containing missing values or with zero variance.
        if any(isnan(y)) || std(y) == 0
            continue;
        end

        % Fit a GLM with diagnosis, age, and sex as predictors.
        [b, dev, stats] = glmfit(X, y);

        % Extract the diagnosis effect.
        P_map_raw(i, j) = stats.p(2);
        T_map(i, j) = stats.t(2);

        % Calculate the effect size
        D_map(i,j) = b(2) / stats.s;
    end
end


fprintf('GLM calculation completed.\n');

% Apply Benjamini-Hochberg FDR correction across all ROI-term pairs.
fprintf('Performing FDR correction using mafdr...\n');

p_vector = P_map_raw(:);

fdr_q_values = mafdr(p_vector, 'BHFDR', true);

Q_map = reshape(fdr_q_values, n_rows, n_cols);

%% 5. Classify Significant ROI-Term Pairs into Four Patterns
% Pattern definitions:
% Type 1: Positive Control mean, positive diagnosis t-statistic.
% Type 2: Positive Control mean, negative diagnosis t-statistic.
% Type 3: Negative Control mean, positive diagnosis t-statistic.
% Type 4: Negative Control mean, negative diagnosis t-statistic.


idx_control = find(disease == 0);
idx_patient = find(disease == 1);

mean_control = mean(dat_harmonized(:, :, idx_control), 3);
mean_patient = mean(dat_harmonized(:, :, idx_patient), 3);

% Mean difference: AD minus Control.
delta_map = mean_patient - mean_control;

alpha = 0.05;

nROI = n_rows;
nFeat = n_cols;

% Initialise summary measures for each ROI.
total_ratio = zeros(nROI, 1);
type_ratio = zeros(nROI, 4);
type_strength = zeros(nROI, 4);
type_d = zeros(nROI, 4);
type_d_max = zeros(nROI, 4);

% Identify ROI-term pairs with significant FDR-adjusted p-values.
current_mask = Q_map < alpha;

n_sig = sum(current_mask(:));

% Retain t-statistics for significant pairs only.
current_T_sig = T_map;
current_T_sig(~current_mask) = NaN;

Sig_Mask = current_mask;
T_sig = current_T_sig;


for i = 1:nROI

    row_mask = current_mask(i, :);
    row_t = T_map(i, :);
    row_mc = mean_control(i, :);
    row_d = D_map(i, :);

    sig_idx = row_mask;

    total_count = sum(sig_idx);
    total_ratio(i) = total_count / nFeat;

    if total_count == 0
        continue;
    end

    mc = row_mc(sig_idx);
    t = row_t(sig_idx);
    d = row_d(sig_idx);

    % Type 1: Positive Control mean and positive diagnosis effect.
    idx1 = (mc > 0) & (t > 0);

    if any(idx1)
        type_ratio(i, 1) = sum(idx1) / nFeat;
        type_strength(i, 1) = mean(t(idx1));
        type_d(i, 1) = mean(d(idx1));
        type_d_max(i, 1) = max(d(idx1));
    end

    % Type 2: Positive Control mean and negative diagnosis effect.
    idx2 = (mc > 0) & (t < 0);

    if any(idx2)
        type_ratio(i, 2) = sum(idx2) / nFeat;
        type_strength(i, 2) = mean(t(idx2));
        type_d(i, 2) = mean(d(idx2));
        type_d_max(i, 2) = min(d(idx2));
    end

    % Type 3: Negative Control mean and positive diagnosis effect.
    idx3 = (mc < 0) & (t > 0);

    if any(idx3)
        type_ratio(i, 3) = sum(idx3) / nFeat;
        type_strength(i, 3) = mean(t(idx3));
        type_d(i, 3) = mean(d(idx3));
        type_d_max(i, 3) = max(d(idx3));
    end

    % Type 4: Negative Control mean and negative diagnosis effect.
    idx4 = (mc < 0) & (t < 0);

    if any(idx4)
        type_ratio(i, 4) = sum(idx4) / nFeat;
        type_strength(i, 4) = mean(t(idx4));
        type_d(i, 4) = mean(d(idx4));
        type_d_max(i, 4) = min(d(idx4));
    end

end

fprintf('------------------------------------------------\n');
fprintf('Signed GLM Analysis Summary:\n');
fprintf('FDR < %.3f: %d significant ROI-term pairs\n', ...
    alpha, n_sig);
fprintf('------------------------------------------------\n');



T = table( ...
    type_strength(:, 1), type_strength(:, 2), ...
    type_strength(:, 3), type_strength(:, 4), ...
    type_ratio(:, 1), type_ratio(:, 2), ...
    type_ratio(:, 3), type_ratio(:, 4), ...
    type_d(:, 1), type_d(:, 2), ...
    type_d(:, 3), type_d(:, 4), ...
    type_d_max(:, 1), type_d_max(:, 2), ...
    type_d_max(:, 3), type_d_max(:, 4), ...
    total_ratio, ...
    'VariableNames', { ...
        'pos_inc_T', 'pos_dec_T', ...
        'neg_weak_T', 'neg_strength_T', ...
        'pos_inc_ratio', 'pos_dec_ratio', ...
        'neg_weak_ratio', 'neg_strength_ratio', ...
        'pos_inc_d', 'pos_dec_d', ...
        'neg_weak_d', 'neg_strength_d', ...
        'pos_inc_d_max', 'pos_dec_d_max', ...
        'neg_weak_d_max', 'neg_strength_d_max', ...
        'total_ratio'});
writetable(T, 'AD_ROI_pattern_summary.csv');

%% 6. Export Significant ROI-Term Pairs by Pattern
region_table = readtable('Schaefer400_7network_name.csv');
region_names = string(region_table.label);

term_table = readtable('Term.csv');
term_names = string(term_table.Term);


output_types = { ...
    'AD_Type1_PosInc', ...
    'AD_Type2_PosDec', ...
    'AD_Type3_NegWeak', ...
    'AD_Type4_NegStr'};

type_idx = 1:4;

fprintf('Exporting CSV files with effect sizes...\n');

for t = type_idx

    rows = [];
    cols = [];
    tvals = [];
    mc_vals = [];
    mp_vals = [];
    qvals = [];
    d_vals = [];

    for i = 1:nROI

        % Select significant pairs belonging to the current pattern.
        sig_idx = (current_mask(i, :) & ...
            ((t == 1 & mean_control(i, :) > 0 & T_map(i, :) > 0) | ...
             (t == 2 & mean_control(i, :) > 0 & T_map(i, :) < 0) | ...
             (t == 3 & mean_control(i, :) < 0 & T_map(i, :) > 0) | ...
             (t == 4 & mean_control(i, :) < 0 & T_map(i, :) < 0)));

        idx_j = find(sig_idx);

        if ~isempty(idx_j)

            rows = [rows; repmat(i, length(idx_j), 1)];
            cols = [cols; idx_j(:)];

            tvals = [tvals; T_map(i, idx_j)'];
            mc_vals = [mc_vals; mean_control(i, idx_j)'];
            mp_vals = [mp_vals; mean_patient(i, idx_j)'];
            qvals = [qvals; Q_map(i, idx_j)'];

            d_vals = [d_vals; D_map(i, idx_j)'];

        end
    end

    % Construct the output table.
    Region_ID = rows;
    Region = region_names(rows);

    Term_ID = cols;
    Term = term_names(cols);

    Tval = tvals;
    Qval = qvals;
    EffectSize_d = d_vals;

    Control_mean = mc_vals;
    Patient_mean = mp_vals;

    T_out = table( ...
        Region_ID, Region, Term_ID, Term, ...
        Tval, EffectSize_d, Qval, ...
        Control_mean, Patient_mean);

    % Save the pattern-specific results.
    filename = [output_types{t}, '.csv'];

    writetable(T_out, filename);

    fprintf( ...
        'CSV saved: %s (%d rows)\n', ...
        filename, height(T_out));

end

fprintf('All analyses and exports completed.\n');
