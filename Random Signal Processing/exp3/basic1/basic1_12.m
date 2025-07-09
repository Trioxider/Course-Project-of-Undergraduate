clc;
clear;
close all;

% Set parameters
fs = 1000; 
f1 = 50; 
f2 = 75; 
noise_std = 0.1; 
t = 0:1/fs:2-1/fs; 
x = cos(2*pi*f1*t) + cos(2*pi*f2*t) + noise_std * randn(size(t)); % Signal with added noise

% Plot periodograms for rectangular and Hamming windows
subplot(1,2,1)
periodogram(x, rectwin(length(x))); % Periodogram with rectangular window
set(gca, 'fontsize', 12, 'fontname', 'times');
title('Periodogram with rectangle window', 'fontsize', 14); % Title for the plot

subplot(1,2,2)
periodogram(x, hamming(length(x))); % Periodogram with Hamming window
set(gca, 'fontsize', 12, 'fontname', 'times');
title('Periodogram with Hamming window', 'fontsize', 14); % Title for the plot

%%
% Change the value of noise standard deviation (sigma)
noise_std_list = [0.1, 0.5, 1, 5];
for i = 1:length(noise_std_list)
    noise_std_value = noise_std_list(i);
    x_test = cos(2*pi*f1*t) + cos(2*pi*f2*t) + noise_std_value * randn(size(t)); % Signal with noise
    subplot(2,2,i)
    periodogram(x_test, rectwin(length(x_test))); % Periodogram with rectangular window
    set(gca, 'fontsize', 12, 'fontname', 'times');
    title(sprintf('Periodogram with variance = %g', noise_std_value), 'fontsize', 14); % Title with noise variance
end

%%
% Change the value of the sampling rate
sampling_rate_list = [1000, 3000, 5000, 10000];
for i = 1:length(sampling_rate_list)
    sampling_rate_value = sampling_rate_list(i);
    t = 0:1/sampling_rate_value:2-1/sampling_rate_value; % Sample points adjusted for sampling rate
    x_test = cos(2*pi*f1*t) + cos(2*pi*f2*t) + noise_std * randn(size(t)); % Signal with adjusted sampling rate
    subplot(2,2,i+2)
    periodogram(x_test, rectwin(length(x_test))); % Periodogram with rectangular window
    set(gca, 'fontsize', 12, 'fontname', 'times');
    title(sprintf('Periodogram with sampling rate = %d', sampling_rate_value), 'fontsize', 14); % Title with sampling rate
end

%%
% Change the value of signal length
signal_length_list = [2, 5, 8, 10];
for i = 1:length(signal_length_list)
    signal_length_value = signal_length_list(i);
    t_value = 0:1/fs:signal_length_value-1/fs; % Sample points for adjusted signal length
    x_test = cos(2*pi*f1*t_value) + cos(2*pi*f2*t_value) + noise_std * randn(size(t_value)); % Signal with adjusted length
    subplot(2,2,i+4)
    periodogram(x_test, rectwin(length(x_test))); % Periodogram with rectangular window
    set(gca, 'fontsize', 12, 'fontname', 'times');
    title(sprintf('Periodogram with signal length = %d', signal_length_value), 'fontsize', 14); % Title with signal length
end

%%
% Change the value of FFT length
fft_length_list = [128, 256, 512, 1024];
for i = 1:length(fft_length_list)
    fft_length_value = fft_length_list(i);
    x_test = cos(2*pi*f1*t) + cos(2*pi*f2*t) + noise_std * randn(size(t)); % Original signal
    padding_length = fft_length_value - length(x_test); % Calculate padding needed for FFT length
    x_test_fft = [x_test, zeros(padding_length, 1)]; % Pad the signal with zeros
    X_fft = fft(x_test_fft, [], 2); % Compute the FFT
    P_dB = 10 * log10(abs(X_fft).^2 / (length(X_fft) * noise_std^2)); % Compute the periodogram in dB
    subplot(2, 2, i)
    plot((fs/2)*linspace(0,1,fft_length_value/2+1), P_dB(1:fft_length_value/2+1)); % Plot the first half of the periodogram
    title(sprintf('Periodogram with FFT length = %d', fft_length_value), 'fontsize', 14); % Title with FFT length
    set(gca, 'fontsize', 12, 'fontname', 'times');
    set(gcf, 'Units', 'centimeter', 'Position', [10 10 28 10]);
end

