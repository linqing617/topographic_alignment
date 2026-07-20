%% ========================================================================
%  SCRIPT: Network-level Term Frequency Analysis of TA Profiles
%  ========================================================================%
% PURPOSE:
% Characterize the functional specialization of cortical networks by
% summarizing the dominant functional terms represented in regional
% Topographic Alignment (TA) profiles.
%
% WORKFLOW:
%   1. Average the first-order TA matrix across subjects.
%   2. For each ROI, extract the top-n and bottom-n ranked functional
%      terms according to TA values.
%   3. Pool regional term lists within each Yeo-7 functional network.
%   4. Compute term frequencies for each network to identify consistently
%      represented functional annotations.
%   5. Export raw term lists and frequency tables for downstream
%      visualization (e.g., word clouds).
%
% REQUIREMENTS:
%   - Statistics and Machine Learning Toolbox
%
% INPUT:
%   - first_order_net.mat
%   - Term.csv
%   - schaefer400_7network_name.csv
%
% OUTPUT:
%   - Network-level raw term lists
%   - Network-level term frequency tables
%% =========================================================================


clear; clc

%% ============ Load Data ============
load('s400_first_order_net.mat')            % first-order TA matrix
first_mean = nanmean(first_order_net, 3);

labels = readtable('Term.csv');              % 539 term names
labels = labels.Term;

[top10_terms, last10_terms] = extractTerms(first_mean, labels, 10);

name = readtable('schaefer400_7network_name.csv'); % Yeo-7 network parcellation index
region = name.label;
index  = name.Index;

net_names = {'Vis','SM','DA','VA','LIB','Con','DM'};

%% ============ Top10 word list + frequency count ============
word_cell = cell(7,1);
for i = 1:7
    sub_term = top10_terms(index==i,:);
    word_cell{i} = sub_term(:);
end

for i = 1:numel(net_names)
    words = word_cell{i};
    words = strrep(words, '''', '');   % remove stray single quotes
    words = strtrim(words);            % trim leading/trailing whitespace

    % --- Raw word list (duplicates kept, one entry per region) ---
    T = table(words, 'VariableNames', {'Word'});
    writetable(T, ['shaefer400_top10_', net_names{i}, '.csv']);

    % --- Word frequency table (unique words + counts, sorted descending) ---
    freq_table = wordFrequency(words);
    writetable(freq_table, ['shaefer400_top10_freq_', net_names{i}, '.csv']);
end

%% ============ Last10 word list + frequency count ============
word_cell = cell(7,1);
for i = 1:7
    sub_term = last10_terms(index==i,:);
    word_cell{i} = sub_term(:);
end

for i = 1:numel(net_names)
    words = word_cell{i};
    words = strrep(words, '''', '');
    words = strtrim(words);

    T = table(words, 'VariableNames', {'Word'});
    writetable(T, ['shaefer400_last10_', net_names{i}, '.csv']);

    freq_table = wordFrequency(words);
    writetable(freq_table, ['shaefer400_last10_freq_', net_names{i}, '.csv']);
end

disp('Done: top10/last10 word lists and frequency tables for each network have been saved.')

