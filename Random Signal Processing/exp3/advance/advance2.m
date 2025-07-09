clc;
clear;
close all;

% Set parameters
fs = 50000; % Sampling frequency in Hz
f0 = 1000; % Initial frequency in Hz
k = 12000; % Frequency modulation rate
tmin = 0; % Signal start time in seconds
tmax = 0.1; % Signal duration in seconds
Ts = 1/fs; % Sampling period
t = tmin:Ts:tmax-Ts; % Time vector

% Generate a chirp signal
X = cos(2*pi*(f0*t + 0.5*k*t.^2));

% Initialize variables
N = 500; % Number of independent runs
K = 0; % Number of successful detections
MSE_total = 0; % Cumulative Mean Squared Error (MSE)
SNR_range = -80:10:0; % Range of Signal-to-Noise Ratios (SNR) in dB
% Storage for results
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

    for k = 1:N
        % Random delay time between 0.11s and 1s
        start_time = 0.01;
        delay = start_time + (1 - 0.11) * rand;
        t_s = tmax + delay;

        % Generate the received signal with noise
        y = [zeros(1, round(delay*fs)), X, zeros(1, round((1-t_s)*fs))];
        y = y + sigma * randn(size(y));
            
        % Define window size and step
        window_duration = 0.1; % Window duration is 0.1 seconds
        window_size = round(window_duration * fs); % Calculate window size based on the sampling rate
        overlap_step = round(window_size / 2); % Calculate overlap step as half of the window size
        
        % Initialize max power and corresponding time index
        max_power = 0;
        
        % Slide window over the signal
        for start_idx = 1:overlap_step:(length(y) - window_size + 1)
            % Extract signal segment
            signal_seg = y(start_idx:(start_idx + window_size - 1));
            
            % Calculate periodogram and frequencies
            [Pxx, ~] = periodogram(signal_seg, [], [], fs);
            
            % Calculate the power of the current signal segment
            current_power = sum(Pxx(:));
            
            % Update max power and corresponding time index
            if current_power > max_power
                max_power = current_power;
                event_time_idx = (start_idx + window_size - 1);
            end
        end

        t_e = event_time_idx / fs; 
        % Calculate the error (end time)
        mse = (t_e - t_s)^2;
        MSE_total = MSE_total + mse;
        
        % Check if detection is successful
        if abs(t_e - t_s) < 0.03
            K = K + 1;
        end
    end
    
    % Calculate MSE and success rate for the current SNR
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

% Plot the results
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