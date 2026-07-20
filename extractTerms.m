function [top_terms, last_terms] = ...
         extractTerms(A, labels, k)
% extractTopLastTerms
% ---------------------------------------------------------
% Extract Top-k and Last-k elements from each row of matrix A
%
% INPUT
%   A       : [numRows × numCols] numeric matrix
%   labels  : 1 × numCols cell array, labels for each column
%   k       : number of top / last elements to extract
%
% OUTPUT
%   top_terms    : numRows × k cell array, Top-k labels
%   last_terms   : numRows × k cell array, Last-k labels
%   top_indices  : numRows × k matrix, indices of Top-k values
%   last_indices : numRows × k matrix, indices of Last-k values
%   top_values   : numRows × k matrix, Top-k values
%   last_values  : numRows × k matrix, Last-k values
% ---------------------------------------------------------

[numRows, numCols] = size(A);

% Check if k is valid
if k > numCols
    error('k (%d) is larger than the number of columns (%d).', k, numCols);
end

% Preallocate memory
top_indices  = zeros(numRows, k);
last_indices = zeros(numRows, k);
top_values   = zeros(numRows, k);
last_values  = zeros(numRows, k);

top_terms  = cell(numRows, k);
last_terms = cell(numRows, k);

for i = 1:numRows

    % Sort values in descending order
    [sortedVals, sortedIdx] = sort(A(i, :), 'descend');

    % ---------------- Top k ----------------
    top_indices(i, :) = sortedIdx(1:k);
    top_values(i, :)  = sortedVals(1:k);

    for j = 1:k
        top_terms{i, j} = labels{sortedIdx(j)};
    end

    % ---------------- Last k ----------------
    last_indices(i, :) = sortedIdx(end-k+1:end);
    last_values(i, :)  = sortedVals(end-k+1:end);

    for j = 1:k
        last_terms{i, j} = labels{sortedIdx(end-k+j)};
    end
end
end
