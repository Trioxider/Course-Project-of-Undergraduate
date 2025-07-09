clc;
clear;
close all;
fs = 1000; % sample rate
f1 = 50; % frequency of the first cosine wave
f2 = 75; % frequency of the second cosine wave
noise_std = 0.1;
t = 0:1/fs:2-1/fs; % sample points
x = cos(2*pi*f1*t) + cos(2*pi*f2*t) + noise_std * randn(size(t)); % length 2s

% Plot custom periodogram
[P,fs1] = my_periodogram(x,fs);
% Convert periodogram power to dB
P_dB = 10 * log10(P / (noise_std^2));
subplot(3,1,1);
plot(fs1,P_dB);
title('Custom Periodogram');

% Plot custom correlogram
[S,fs2] = my_correlogram(x,fs);
% Convert correlogram power to dB
S_dB = 10 * log10(S / (noise_std^2));
subplot(3,1,2);
plot(fs2,S_dB);
title('Custom Correlogram');

% Plot MATLAB default periodogram
subplot(3,1,3);
periodogram(x, rectwin(length(x)), 'twosided');
title('Default Periodogram');

% Set the font size of the plot
set(gca,'FontSize',12);

%%
% Change the value of noise standard deviation (sigma)
noise_std_list = [0.1, 0.5, 1, 5];
figure('Position', [100, 100, 1000, 600]); % [x y width height]
for i = 1:length(noise_std_list)
    noise_std_value = noise_std_list(i);
    x_test = cos(2*pi*f1*t) + cos(2*pi*f2*t) + noise_std_value * randn(size(t)); % Signal with noise
    [P,fs1] = my_periodogram(x_test,fs);
    % Convert periodogram power to dB
    P_dB = 10 * log10(P / (noise_std_value^2));
    subplot(4,1,i)
    plot(fs1/2,P_dB);
    title(sprintf('Periodogram with variance = %g', noise_std_value), 'fontsize', 14); % Title with noise variance
end

%%
% Change the value of the sampling rate
sampling_rate_list = [1000, 3000, 5000, 10000];
figure('Position', [100, 100, 1000, 600]); % [x y width height]
for i = 1:length(sampling_rate_list)
    sampling_rate_value = sampling_rate_list(i);
    t = 0:1/sampling_rate_value:2-1/sampling_rate_value; % Sample points adjusted for sampling rate
    x_test = cos(2*pi*f1*t) + cos(2*pi*f2*t) + noise_std * randn(size(t)); % Signal with adjusted sampling rate
    subplot(4,1,i)
    [P,fs1] = my_periodogram(x_test,fs);
    % Convert periodogram power to dB
    P_dB = 10 * log10(P / (noise_std^2));
    plot(fs1,P_dB);
    title(sprintf('Periodogram with sampling rate = %d', sampling_rate_value), 'fontsize', 14); % Title with sampling rate
end

%%
% Change the value of signal length
signal_length_list = [2, 5, 8, 10];
figure('Position', [100, 100, 1000, 600]); % [x y width height]figure('Position', [100, 100, 1000, 600]); % [x y width height]
for i = 1:length(signal_length_list)
    signal_length_value = signal_length_list(i);
    t_value = 0:1/fs:signal_length_value-1/fs; % Sample points for adjusted signal length
    x_test = cos(2*pi*f1*t_value) + cos(2*pi*f2*t_value) + noise_std * randn(size(t_value)); % Signal with adjusted length
    subplot(4,1,i)
    [P,fs1] = my_periodogram(x_test,fs);
    % Convert periodogram power to dB
    P_dB = 10 * log10(P / (noise_std^2));
    plot(fs1,P_dB);
    title(sprintf('Periodogram with signal length = %d', signal_length_value), 'fontsize', 14); % Title with signal length
end


% Custom periodogram function
function [P,fs1] = my_periodogram(x,fs)
    N = length(x);
    % Calculate Fourier transform
    X = fft(x, N);
    % Calculate power spectral density
    P = abs(X/N).^2;
    % Normalize power spectral density
    P = P/N;
    fs1 = (0:N-1) * (fs / N);
end

% Custom correlogram function
function [S,fs2] = my_correlogram(x,fs)
    N = length(x);
    % Calculate autocorrelation function
    Rxx = xcorr(x, 'coeff');
    % Calculate correlogram
    S = fft(Rxx, N);
    % Normalize correlogram
    S = abs(S/N).^2;
    fs2 = (0:N-1) * (fs / N);
end