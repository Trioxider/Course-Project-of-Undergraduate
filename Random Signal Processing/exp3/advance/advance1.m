clc;
clear;
close all;

% Set parameters
fs = 50000; % Sampling frequency
f0 = 1000; % Initial frequency
k = 12000; % Frequency modulation rate
tmin = 0; % Signal start time
tmax = 0.1; % Signal duration
Ts = 1/fs; % Sampling period
t = tmin:Ts:tmax-Ts; % Time vector

% Generate a chirp signal
X = cos(2*pi*(f0*t + 0.5*k*t.^2));

% Initialize variables
N = 500; % Number of independent runs
K = 0; % Number of successful runs
MSE_total = 0; % Total Mean Squared Error (MSE)
SNR_range = -3500:500:0; % Range of Signal-to-Noise Ratios (SNR)
% Results storage
MSE_results = zeros(size(SNR_range));
success_rate_results = zeros(size(SNR_range));

% Loop to test different SNRs
for snr_idx = 1:length(SNR_range)
    snr_db = SNR_range(snr_idx);
    snr = 10^(snr_db/10); % Calculate the signal-to-noise ratio
    K = 0;
    MSE_total = 0;
    % Add noise
    Ps = mean(X.^2); % Average power of the signal
    sigma_n2 = Ps / snr; % Noise power
    sigma = sqrt(sigma_n2);

    for i = 1:N
        % Random end time
        delay = 0.11 + rand() * (1 - 0.11);
        t_s = tmax + delay;

        Y = [zeros(1, round(delay*fs)), X];
        Nt = sigma * randn(size(Y)); % Noise signal
        Y = Y + Nt;
        % Design of the matched filter
        matched_filter = fliplr(X); % Time-reversed signal
        
        % Matched filtering
        Y_filtered = conv(Y, matched_filter, 'full');
        
        % Find the peak position of the filtered signal (assumed to be the signal end time)
        [~ , end_idx] = max(Y_filtered);
        t_e = (end_idx - length(matched_filter)  + 1) * Ts + tmax;
        
        % Calculate MSE
        mse = (t_e - t_s)^2;
        MSE_total = MSE_total + mse;
        
        % Calculate success rate
        if abs(t_e - t_s) < 0.03
            K = K + 1;
        end
    end
    
    % Calculate MSE and success rate at the current SNR
    MSE_results(snr_idx) = MSE_total / N;
    success_rate_results(snr_idx) = K / N;
end

% Display results
fprintf('SNR Range: ');
disp(SNR_range);
fprintf('MSE Results: ');
disp(MSE_results);
fprintf('Success Rate Results: ');
disp(success_rate_results);

% Plot the charts
figure;
subplot(2,1,1);
bar(SNR_range, MSE_results);
title('MSE vs SNR');
xlabel('SNR (dB)');
ylabel('MSE');

subplot(2,1,2);
bar(SNR_range, success_rate_results);
title('Success Rate vs SNR');
xlabel('SNR (dB)');
ylabel('Success Rate');