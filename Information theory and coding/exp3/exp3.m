clear; clc;

load('orderlist.mat');  % load vector orderList
max_N = 4;
H = 1.46;  % Entropy rate in bits/symbol(estimation)

avg_len_all = zeros(4, max_N);  % rows: method (1-const, 2-SBE, 3-Fano, 4-Huffman), cols: N
eff_all = zeros(4, max_N);

for N = 1:max_N
    % Step 1: Create N-extended source
    ext_data = orderList(1:end - mod(length(orderList), N));
    ext_data = reshape(ext_data, N, [])';
    [symbols, ~, idx] = unique(ext_data, 'rows');
    probs = accumarray(idx, 1) / size(ext_data, 1);
    
    % Remove zero-probability symbols
    nonzero = probs > 0;
    probs = probs(nonzero);
    symbols = symbols(nonzero, :);
    probs = probs(:)';
    
    % Sort probabilities (needed for SBE and Fano)
    [sorted_probs, sort_idx] = sort(probs, 'descend');
    
    % ---------------- Constant length code ----------------
    const_len = ceil(log2(length(probs)));
    avg_len_all(1, N) = const_len;
    eff_all(1, N) = H * N / const_len;

    % ---------------- Shannon (SBE) code ----------------
    SBE_codes = SBE(sorted_probs);
    SBE_lens = cellfun(@strlength, SBE_codes);
    avg_len_all(2, N) = sum(sorted_probs .* SBE_lens);
    eff_all(2, N) = H * N / avg_len_all(2, N);

    % ---------------- Fano code ----------------
    fano_codes = fano(sorted_probs);
    fano_lens = cellfun(@strlength, fano_codes);
    avg_len_all(3, N) = sum(sorted_probs .* fano_lens);
    eff_all(3, N) = H * N / avg_len_all(3, N);

    % ---------------- Huffman code ----------------
    huff_symbols = num2cell(1:length(probs));
    dict = huffmandict(huff_symbols, sorted_probs);
    lens = cellfun(@length, dict(:,2));   % M×1
    avg_len_all(4, N) = sorted_probs * lens;   % (1×M) * (M×1) → scalar
    eff_all(4, N) = H * N / avg_len_all(4, N);
end

% Plotting
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
xlabel('Extension N'); ylabel('Coding Efficiency');
title('Coding Efficiency vs Extension');
grid on;
