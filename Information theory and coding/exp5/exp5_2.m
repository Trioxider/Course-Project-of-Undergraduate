%% 4.2 Simulation for varying m
clear; % Clear workspace variables
clc;   % Clear command window
m_values = 3:7; % Range of parity bit numbers to simulate
p_fixed = 0.1;  % Fixed crossover probability for BSC
num_bits = 1e7; % Total number of information bits
% Generate random information bits (Bernoulli p=0.5)
info = rand(1, num_bits) > 0.5; 

% Preallocate arrays for results
error_m = zeros(size(m_values));       % Simulated error probabilities
theoretical = zeros(size(m_values));   % Theoretical error probabilities

% Loop through each m value
for j = 1:length(m_values)
    m = m_values(j);     % Current number of parity bits
    n = 2^m - 1;        % Codeword length
    k = n - m;          % Information bits per block
    
    % Calculate padding length to make total bits multiple of k
    pad_len = mod(k - mod(num_bits, k), k);
    % Pad information bits with zeros
    info_bits_pad = [info, zeros(1, pad_len)];
    
    % Encode information bits using Hamming code
    encoded = hamming_encoder(info, m);
    % Simulate BSC channel transmission
    received = bsc(encoded, p_fixed);
    % Decode received sequence
    decoded = hamming_decoder(received, m);
    
    % Reshape original and decoded data into blocks (each column is a block)
    info_matrix = reshape(info_bits_pad, k, []);
    decoded_matrix = reshape(decoded, k, []);
    % Calculate block error rate (codeword error probability)
    error_m(j) = mean(any(info_matrix ~= decoded_matrix, 1));
    
    % Compute theoretical codeword error probability:
    % 1 - P(no errors) - P(exactly one error)
    theoretical(j) = 1 - (1-p_fixed)^n - n*p_fixed*(1-p_fixed)^(n-1);
end

% Plot simulation vs theoretical results
figure;
plot(m_values, error_m, 'bo-', 'LineWidth', 1.5); hold on;
plot(m_values, theoretical, 'r--', 'LineWidth', 1.5);
xlabel('Check digit number m');
ylabel('Codeword Error Probability');
legend('Simulation', 'Theory','Location', 'southwest');
title(sprintf('Error Probability vs. m for p=%.2f of Hamming Codes', p_fixed));