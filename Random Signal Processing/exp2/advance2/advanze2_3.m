clc; 
clear all; 
close all; 
load noise.mat; 
theta = []; % Initialize theta array for storing DOA values
rng(0); % Set the random number generator seed for reproducibility
c = 340; 
N = 44100; 
d = 0.085; 

% Create a table listing all delays corresponding to DOA values
lag_range = -11:11; 
DOA_table = zeros(length(lag_range), 2); % Initialize DOA table with two columns for delay and DOA

% Calculate DOA values for each delay
for i = 1:length(lag_range)
    Lag = lag_range(i);
    Lag_samples = Lag; 
    theta_value = asin(Lag_samples * c / (N * d)); % Calculate DOA
    theta(i) = theta_value; % Store the calculation result in the theta array
    DOA_table(i, :) = [Lag, rad2deg(theta_value)]; 
end

% Display DOA table
fprintf('Lag (samples) | DOA (degrees)\n');
for i = 1:size(DOA_table, 1)
    fprintf('%11d | %11.2f\n', DOA_table(i, 1), DOA_table(i, 2));
end

% Read the original audio signal
[sig_ori, ~] = audioread('test_audio.wav');
sig_ori = sig_ori'; 
Lsig = length(sig_ori); 

% Set the conditions for the signal-to-noise ratio (SNR)
SNR_dB = [30, 0, -20]; 

num_mic = 8; % Number of microphones
correct_ratio = []; % Initialize the array to store the correct ratio

% Perform 100 tests for each SNR condition
for i = 1:length(SNR_dB)
    snr = SNR_dB(i); % Assign the current element to the variable snr
    correct_num = 0; % Initialize the count of correct DOA estimations 
    for j = 1:100
        % Randomly select a DOA
        true_DOA_index = randi(23);
        true_DOA = theta(true_DOA_index); % The true DOA for this trial
        DOA_estimate = []; % Initialize the array to store the estimated DOA values

        % Loop through each microphone starting from the second one
        for mic = 2:num_mic
            sig_received = []; % Initialize the array to store the received signals
            D = d * mic; % Calculate the distance for the current microphone pair

            % Calculate the signal power and noise power
            signal_power = sig_ori*sig_ori'/Lsig;
            noise_power = signal_power / (10^(snr/10));
            
            % Add noise to construct the received signals at two microphones
            sig_noise1 = sig_ori + sqrt(noise_power) * randn(1,Lsig);
            sig_noise2 = sig_ori + sqrt(noise_power) * randn(1,Lsig);
            
            % Calculate the time delay in samples
            L_TD = round(sin(true_DOA) * D * N / c);

            % Generate noise for the time delay
            sig_temp = sqrt(noise_power) * randn(L_TD, 1);

            % Construct the signal with the delay
            if L_TD > 0
                sig_received(1, :) = [sig_temp', sig_noise1];
                sig_received(2, :) = [sig_noise2, sig_temp'];
            elseif L_TD < 0
                sig_received(1, :) = [sig_noise1, sig_temp'];
                sig_received(2, :) = [sig_temp', sig_noise2];
            else
                sig_received(1, :) = sig_noise1;
                sig_received(2, :) = sig_noise2;
            end
            
            % Calculate cross-correlation to estimate the delay
            R_12 = xcorr(sig_received(1, :), sig_received(2, :), 'coeff');
            [~, index] = max(R_12); % Find the index of the maximum value in the cross-correlation
            estimated_lag = index - (Lsig + 1); % Estimate the lag
            
            % Calculate the estimated DOA
            DOA_estimate(mic-1) = asin(estimated_lag * c / (N * D)); % Calculate the estimated DOA
        end
        DOA_estimate = mean(DOA_estimate); % Take the mean of the DOA estimates
        error = abs(true_DOA - DOA_estimate); 
        if error <= 2 
            correct_num = correct_num + 1; % Increment the count of correct estimations
        end
    end
    correct_ratio(i) = correct_num / 100; % Calculate the correct ratio
    fprintf("SNR=%d dB, Correct Ratio: %.2f\n", snr, correct_ratio(i)); 
end