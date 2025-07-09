% 4.1 Simulation for m=3 with i.i.d. and non-i.i.d. bits
clear; % Clear workspace variables
clc;   % Clear command window
m = 3; % Number of parity bits (determines code parameters)
n = 2^m - 1; % Codeword length (number of bits in encoded block)              
k = n - m;   % Message length (number of information bits per block)               
num_bits = 1e7; % Total number of information bits (reduced for faster execution)
p_values = 0:0.01:0.2; % Crossover probability range for BSC channel

% ========== i.i.d. case (equiprobable 0s and 1s) ==========
% Generate i.i.d. information bits (Bernoulli p=0.5)
info_iid = (rand(1, num_bits) > 0.5); 

% Pad information bits to make length multiple of k
pad_len = mod(k - mod(num_bits, k), k);
info_bits_pad = [info_iid, zeros(1, pad_len)]; % Zero padding
info_matrix = reshape(info_bits_pad, k, []); % Reshape into message blocks

% Encode padded information bits using Hamming code
encoded_iid = hamming_encoder(info_bits_pad, m);

% ========== Non-i.i.d. case (biased toward 0s) ==========
% Generate non-i.i.d. information bits (Bernoulli p=0.2 -> 80% 0s)
info_non_iid = (rand(1, num_bits) > 0.2); 

% Apply same padding and reshaping as i.i.d. case
info_bits_non_pad = [info_non_iid, zeros(1, pad_len)]; 
info_non_matrix = reshape(info_bits_non_pad, k, []); 
encoded_non_iid = hamming_encoder(info_bits_non_pad, m);

% Preallocate error probability arrays
error_sim_iid = zeros(size(p_values)); % i.i.d. simulation results
error_theory = zeros(size(p_values));  % Theoretical error probability

% ========== Simulation loop for i.i.d. case ==========
for i = 1:length(p_values)
    p = p_values(i); % Current crossover probability
    % Simulate BSC channel for i.i.d. encoded data
    received_iid = bsc(encoded_iid, p);
    % Decode received sequence
    decoded_iid = hamming_decoder(received_iid, m);
    
    % Reshape decoded data into message blocks
    decoded_matrix = reshape(decoded_iid, k, []);
    % Calculate codeword error rate (block error rate)
    error_sim_iid(i) = mean(any(info_matrix ~= decoded_matrix, 1));
    
    % Calculate theoretical codeword error probability:
    % 1 - P(no errors) - P(exactly 1 error)
    error_theory(i) = 1 - (1-p)^n - n*p*(1-p)^(n-1);
end

% ========== Simulation loop for non-i.i.d. case ==========
error_sim_non_iid = zeros(size(p_values)); % Non-i.i.d. results
for i = 1:length(p_values)
    p = p_values(i);
    % Simulate BSC channel for non-i.i.d. encoded data
    received_non_iid = bsc(encoded_non_iid, p);
    % Decode received sequence
    decoded_non_iid = hamming_decoder(received_non_iid, m);
    
    % Reshape decoded data and calculate block error rate
    decoded_non_matrix = reshape(decoded_non_iid, k, []);
    error_sim_non_iid(i) = mean(any(info_non_matrix ~= decoded_non_matrix, 1));
end

% ========== Plot results ==========
figure;
plot(p_values, error_sim_iid, 'bo-', p_values, error_theory, 'r--', p_values, error_sim_non_iid, 'gx-'); 
xlabel('Crossover Probability (p)');
ylabel('Codeword Error Probability');
legend('i.i.d. Simulation', 'Theoretical', 'Non-i.i.d. Simulation');
title('Error Probability vs. p for Hamming Code (m=3)');