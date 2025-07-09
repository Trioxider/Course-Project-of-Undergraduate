clear; clc;

load('orderlist.mat');  % Load vector orderList
max_N = 4;

% ----------------- Step 1: Calculate single-symbol entropy H(S) -----------------
% Compute probabilities for single symbols (N=1)
[unique_symbols, ~, idx] = unique(orderList);
probs_single = accumarray(idx, 1) / length(orderList);
H_S = -sum(probs_single .* log2(probs_single));  % Single-symbol entropy

% Preallocate arrays for extended entropy and coding metrics
H_extended = zeros(1, max_N);  % H_N for N=1,2,3,4
avg_len_all = zeros(4, max_N);  % Rows: methods (1-Constant, 2-Shannon, 3-Fano, 4-Huffman)
eff_all = zeros(4, max_N);

for N = 1:max_N
    % ----------------- Step 2: Calculate N-extended entropy H_N -----------------
    % Generate N-grams using sliding window
    L = length(orderList);
    ext_data = zeros(L - N + 1, N);
    for i = 1:(L - N + 1)
        ext_data(i, :) = orderList(i:i+N-1);
    end
    
    % Compute N-gram probabilities
    [symbols, ~, idx] = unique(ext_data, 'rows');
    probs = accumarray(idx, 1) / size(ext_data, 1);
    probs = probs(probs > 0);  % Remove zero probabilities
    H_N = -sum(probs .* log2(probs)) / N;  % Entropy of N-extended source
    H_extended(N) = H_N;
    
    % Sort probabilities for coding algorithms
    [sorted_probs, ~] = sort(probs, 'descend');
    sorted_probs = sorted_probs(:)';  
    
    % ----------------- Coding Methods -----------------
    % Constant Length Code
    const_len = ceil(log2(length(sorted_probs)));
    avg_len_all(1, N) = const_len / N;
    eff_all(1, N) = H_N / avg_len_all(1, N);
    
    % Shannon Code (SBE)
    SBE_codes = SBE(sorted_probs);
    SBE_lens = cellfun(@strlength, SBE_codes);
    avg_len_all(2, N) = sum(sorted_probs .* SBE_lens) / N;
    eff_all(2, N) = H_N / avg_len_all(2, N);
    
    % Fano Code
    fano_codes = fano_optimized(sorted_probs);
    fano_lens = cellfun(@strlength, fano_codes);
    avg_len_all(3, N) = sum(sorted_probs .* fano_lens) / N;
    eff_all(3, N) = H_N / avg_len_all(3, N);
    
    % Huffman Code
    huff_symbols = num2cell(1:length(sorted_probs));
    dict = huffmandict(huff_symbols, sorted_probs);
    lens = cellfun(@length, dict(:,2));
    avg_len_all(4, N) = sorted_probs * lens / N;
    eff_all(4, N) = H_N / avg_len_all(4, N);
end

% ----------------- Print Results -----------------
fprintf('Single-symbol entropy H(S) = %.4f bits/symbol\n', H_S);
fprintf('================================================================\n');
for N = 1:max_N
    fprintf('Extension N = %d | Extended entropy H_%d = %.4f bits\n', N, N, H_extended(N));
    fprintf('----------------------------------------\n');
    fprintf('Method\t\tAvg Length\tEfficiency\n');
    fprintf('Constant\t%.4f\t\t%.4f\n', avg_len_all(1, N), eff_all(1, N));
    fprintf('Shannon\t\t%.4f\t\t%.4f\n', avg_len_all(2, N), eff_all(2, N));
    fprintf('Fano\t\t%.4f\t\t%.4f\n', avg_len_all(3, N), eff_all(3, N));
    fprintf('Huffman\t\t%.4f\t\t%.4f\n', avg_len_all(4, N), eff_all(4, N));
    fprintf('----------------------------------------\n\n');
end

% ----------------- Plot Results -----------------
N_vals = 1:max_N;
figure;
plot(N_vals, avg_len_all', '-o', 'LineWidth', 2);
legend('Constant', 'Shannon', 'Fano', 'Huffman');
xlabel('Extension N'); ylabel('Average Code Length (bits)');
title('Average Coding Length vs Extension');
grid on;

figure;
plot(N_vals, eff_all', '-s', 'LineWidth', 2);
legend('Constant', 'Shannon', 'Fano', 'Huffman');
xlabel('Extension N'); ylabel('Coding Efficiency (H_N / \bar{L})');
title('Coding Efficiency vs Extension');
grid on;