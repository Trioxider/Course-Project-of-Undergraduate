function [res1, res2, size1, size2] = splitf(pq)
% SPLITF  Splits a probability vector into two parts with sums as close as possible.
%   [res1, res2, size1, size2] = splitf(pq)
%   Inputs:
%     pq     –  a row vector of probabilities (not necessarily sorted)
%   Outputs:
%     res1   – subgroup 1 (first k elements)
%     res2   – subgroup 2 (remaining elements)
%     size1  – length(res1)
%     size2  – length(res2)

    total = sum(pq);
    csum  = cumsum(pq);
    % find split index where cumulative sum is closest to half
    [~, idx] = min(abs(csum - total/2));
    
    % partition
    res1  = pq(1:idx);
    res2  = pq(idx+1:end);
    size1 = numel(res1);
    size2 = numel(res2);
end
