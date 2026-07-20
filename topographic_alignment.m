function sub_dis = topographic_alignment(sub_dat, term_all, roi, t1_mask_ind)
% MAKE_NET  Compute a subject's ROI-by-term similarity ("first-order") matrix.
%
%   sub_dis = topographic_alignment(sub_dat, term_all, roi, t1_mask_ind)
%
%   Inputs:
%     sub_dat      - [n_voxel x 1] subject grey matter values at in-mask
%                    voxels (same voxel order as t1_mask_ind).
%     term_all     - [n_voxel x n_term] term/reference maps at the same
%                    in-mask voxels.
%     roi          - [n_roi x 1] list of unique ROI IDs in the parcellation.
%     t1_mask_ind  - [n_voxel x 1] ROI label of each in-mask voxel (same
%                    order as sub_dat / term_all rows).
%
%   Output:
%     sub_dis      - [n_roi x n_term] matrix, where each entry is the
%                    cosine similarity (1 - cosine distance) between the
%                    subject's grey matter pattern and a given term map,
%                    computed within that ROI.

    roi_dis = zeros(size(term_all, 2), length(roi));

    for m = 1:length(roi)
        sub_roi   = sub_dat(t1_mask_ind == roi(m));
        neuro_roi = term_all(t1_mask_ind == roi(m), :);

        % Only keep voxels where the term map is non-zero (i.e., in the
        % term map's valid coverage), to avoid biasing the similarity
        % with structurally-zero background voxels.
        voxel = neuro_roi(:, 1) ~= 0;
        sub_roi   = sub_roi(voxel);
        neuro_roi = neuro_roi(voxel, :);

        roi_dis(:, m) = 1 - pdist2(zscore(sub_roi)', zscore(neuro_roi)', 'cosine')';
    end

    sub_dis = roi_dis';   % [n_roi x n_term]
end