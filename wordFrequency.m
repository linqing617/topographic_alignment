function freq_table = wordFrequency(words)
    % Counts how many times each unique term occurs within a network's
    % pooled term list. This frequency will later be normalized to the
    % maximum frequency within the network to weight terms in the word
    % cloud visualization (i.e., freq_normalized = Count / max(Count)).
    %
    % words: cell array of char/string, may contain duplicates
    words = strrep(words, '''', '');
    words = strtrim(words);
    words = words(~cellfun(@isempty, words));  % drop empty strings, if any

    [uwords, ~, ic] = unique(words);
    counts = accumarray(ic, 1);

    freq_table = table(uwords, counts, 'VariableNames', {'Word','Count'});
    freq_table = sortrows(freq_table, 'Count', 'descend');
end
