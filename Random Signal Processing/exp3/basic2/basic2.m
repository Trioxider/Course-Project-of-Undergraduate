clc;
clear;
close all;
fs = 1000; % sample rate
f1 = 50; % frequency of the first sine wave
f2 = 75; % frequency of the second cosine wave
noise_std = 0.1;
M = 100; % running times
t = 0:1/fs:2-1/fs; % sample points
% x = sin(2*pi*f1*t)+2*cos(2*pi*f2*t)+4*cos(2*pi*f1*t) + noise_std * randn(size(t)); % length 2s
P_runs = zeros(length(t), 100);
P_sum = zeros(1,length(t)); % Initialize the power spectrum and array

for m = 1:M
    % Generate an interference frequency ωI, uniformly distributed in [50π, 80π]
    wI = 50*pi + (80*pi - 50*pi) * rand;
    
    % Generate the signal X(t)
    x = sin(2*pi*f1*t) + 2*cos(2*pi*f2*t) + 4*cos(wI*t) + noise_std * randn(size(t));
    
    % Calculate the periodogram (using a custom function)
    P = my_periodogram(x, fs);
    P_runs(:, m) = P;
    % Accumulate the results of the periodogram
    P_sum = P_sum + P;
end

% Calculate the power spectrum, which is the average of all periodograms
P_spectrum = mean(P_runs, 2);

% Plot the first, fiftieth, and one hundredth periodograms and the power spectrum
subplot(4, 1, 1);
plot((0:length(t)-1)*(fs/length(t)), P_runs(:, 1));
title('Periodogram of the 1st run');
xlabel('Frequency (Hz)');
ylabel('Power');

subplot(4, 1, 2);
plot((0:length(t)-1)*(fs/length(t)), P_runs(:, 50));
title('Periodogram of the 50th run');
xlabel('Frequency (Hz)');
ylabel('Power');

subplot(4, 1, 3);
plot((0:length(t)-1)*(fs/length(t)), P_runs(:, 100));
title('Periodogram of the 100th run');
xlabel('Frequency (Hz)');
ylabel('Power');

subplot(4, 1, 4);
plot((0:length(t)-1)*(fs/length(t)), P_spectrum);
title('Average Power Spectrum over 100 runs');
xlabel('Frequency (Hz)');
ylabel('Average Power');

% Set the font size of the plot
set(gca,'FontSize',12);

%%
M = 100; % running times
noise_std_list = [0.01, 1, 10, 100];
figure('Position', [100, 100, 1000, 600]); % [x y width height]
for i = 1:length(noise_std_list)
    noise_std_value = noise_std_list(i);
    P_runs = zeros(length(t), 100);
    P_sum = zeros(1,length(t)); % Initialize the power spectrum and array
    for m = 1:M
        % Generate an interference frequency ωI, uniformly distributed in [50π, 80π]
        wI = 50*pi + (80*pi - 50*pi) * rand;
        
        % Generate the signal X(t)
        x = sin(2*pi*f1*t) + 2*cos(2*pi*f2*t) + 4*cos(wI*t) + noise_std_value * randn(size(t));
        
        % Calculate the periodogram (using a custom function)
        P = my_periodogram(x, fs);
        P_runs(:, m) = P;
        % Accumulate the results of the periodogram
        P_sum = P_sum + P;
    end
    % Calculate the power spectrum, which is the average of all periodograms
    P_spectrum = mean(P_runs, 2);
    subplot(4, 1, i);
    plot((0:length(t)-1)*(fs/length(t)), P_spectrum);
    title(sprintf('Average Power Spectrum with variance = %d',noise_std_value));
    xlabel('Frequency (Hz)');
    ylabel('Average Power');
end
set(gca,'FontSize',12);




% Custom periodogram function
function [P_dB,fs1] = my_periodogram(x,fs)
    N = length(x);
    % Calculate the Fourier transform
    X = fft(x, N);
    % Calculate the power spectral density
    P = abs(X/N).^2;
    % Normalize the power spectral density
    P = P/N;
    fs1 = (0:N-1) * (fs / N);
    P_dB = 10 * log10(P / (0.1^2));
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