% 1. Predictive Coding (LPC) Section
% 2. 8×8 DCT Transform Coding (Sub-image and Full Image)

clear; close all; clc;

%% ========== 1. Predictive Coding (LPC) ==========
% 1.1 Read audio and plot original signal
[x, Fs] = audioread('audio.wav');   % x: M×1 vector, Fs: Sampling rate
t = (0:length(x)-1)/Fs;
figure('Name','Original Audio Signal');
plot(t, x);
xlabel('Time (s)'); ylabel('Amplitude');
title('Original Audio Signal');
grid on;

% 1.2 Calculate approximate entropy of original signal
[Nv, Edges] = histcounts(x, 128, 'Normalization','probability');
H_x = -sum(Nv .* log2(Nv + eps));
fprintf('Original signal approximate entropy H_x = %.4f bit\n', H_x);

% 1.3 LPC prediction, p=12
p = 12;
a = lpc(x, p);     % Return coefficient vector a of length p+1, a(1)=1

% 1.4 Error signal calculation
B = [0, -a(2:end)];       % Prepare filter numerator
x_hat = filter(B, 1, x);  % Obtain predicted signal
e = x - x_hat;            % Calculate error signal

% Plot predicted and error signals
figure('Name','Predicted Signal');
plot(t, x_hat);
xlabel('Time (s)'); ylabel('Predicted Amplitude');
title('Predicted Signal');
grid on;

figure('Name','Prediction Error Signal');
plot(t, e);
xlabel('Time (s)'); ylabel('Error Amplitude');
title('Prediction Error Signal');
grid on;

% 1.5 Calculate entropies for predicted and error signals
Np = histcounts(x_hat, Edges, 'Normalization','probability');
H_p = -sum(Np .* log2(Np + eps));
fprintf('Predicted signal approximate entropy H_p = %.4f bit\n', H_p);

Ne = histcounts(e, Edges, 'Normalization','probability');
H_e = -sum(Ne .* log2(Ne + eps));
fprintf('Error signal approximate entropy H_e = %.4f bit\n', H_e);

% 1.6 Plot PDF comparisons
mid = (Edges(1:end-1) + Edges(2:end)) / 2;
figure('Name','PDF Comparison: Original vs Predicted');
bar(mid, Nv, 'FaceAlpha',0.6); hold on;
bar(mid, Np, 'FaceAlpha',0.6);
xlabel('Amplitude'); ylabel('Probability');
legend('Original Signal','Predicted Signal');
title('Probability Distribution Comparison');
grid on;

figure('Name','PDF Comparison: Original vs Error');
bar(mid, Nv, 'FaceAlpha',0.6); hold on;
bar(mid, Ne, 'FaceAlpha',0.6);
xlabel('Amplitude'); ylabel('Probability');
legend('Original Signal','Error Signal');
title('Error Signal Distribution Analysis');
grid on;

%% ========== 2. Transform Coding (DCT) ==========

% 2.1 Sub-image DCT demonstration
load('subimage.mat');   % Load test subimage (variable X)
Y = dct2(X);
X_rec = idct2(Y);

% Display DCT results
figure('Name','DCT Transform Demonstration');
subplot(1,3,1);
imshow(mat2gray(X)); title('Original Subimage');
subplot(1,3,2);
imshow(log(abs(Y)), []); title('DCT Coefficient Matrix');
subplot(1,3,3);
imshow(mat2gray(X_rec)); title('IDCT Reconstruction');

% 2.2 Full image block processing
I_rgb = imread('Mona Lisa.jpg');
I_gray = rgb2gray(I_rgb);
[H, W] = size(I_gray);
blk = 8;  % Block size

% Initialize coefficient matrix
dct_coeffs = zeros(H,W);

% Block DCT processing
for i = 1:blk:H
    for j = 1:blk:W
        sub_img = I_gray(i:i+7, j:j+7);
        Y = dct2(sub_img);
        dct_coeffs(i:i+7, j:j+7) = Y;
    end
end

%% Method 1: Threshold compression
thresholds = [20, 50, 100, 300];

figure;
for idx = 1:length(thresholds)
    k = thresholds(idx);
    compressed_img = zeros(H,W);
    zero_count = 0;
    
    for i = 1:8:H
        for j = 1:8:W
            sub_dct = dct_coeffs(i:i+7, j:j+7);
            mask = abs(sub_dct) < k;
            zero_count = zero_count + sum(mask(:));
            sub_dct(mask) = 0;
            reconstructed_sub = idct2(sub_dct);
            compressed_img(i:i+7, j:j+7) = reconstructed_sub;
        end
    end
    
    compressed_img_norm = mat2gray(compressed_img);
    compression_rate = (zero_count / (H*W)) * 100;
    
    subplot(2, 2, idx);
    imshow(compressed_img_norm);
    title(['k=', num2str(k), ', CR: ', num2str(compression_rate, '%.2f'), '%']);
end

%% Compression Method 2: Triangular Retention
I_gray = im2double(rgb2gray(I_rgb));

% Block DCT processing
for i = 1:blk:H
    for j = 1:blk:W
        sub_img = I_gray(i:i+7, j:j+7);
        Y = dct2(sub_img);
        dct_coeffs(i:i+7, j:j+7) = Y;
    end
end


retention_levels = [1, 3, 6, 10];
figure('Name','Triangular Retention Compression');
for L = retention_levels
    % Calculate triangular mask
    t = floor((sqrt(8*L+1)-1)/2);
    mask = triu(ones(blk, blk))';
    mask = double(cumsum(mask, 1) + cumsum(mask, 2) <= t+1);
    
    % Apply mask to all blocks
    masked_coeffs = zeros(size(dct_coeffs));
    total_zeros = 0;
    for i = 1:blk:H
        for j = 1:blk:W
            row_range = i:min(i+blk-1, H);
            col_range = j:min(j+blk-1, W);
            current_block = dct_coeffs(row_range, col_range);
            masked_block = current_block .* mask(1:length(row_range), 1:length(col_range));
            masked_coeffs(row_range, col_range) = masked_block;
            total_zeros = total_zeros + sum(masked_block(:) == 0);
        end
    end
    
    % Reconstruct image
    reconstructed = zeros(size(I_gray));
    for i = 1:blk:H
        for j = 1:blk:W
            row_range = i:min(i+blk-1, H);
            col_range = j:min(j+blk-1, W);
            reconstructed(row_range, col_range) = idct2(masked_coeffs(row_range, col_range));
        end
    end
    
    % Calculate compression metrics
    compression_rate = (total_zeros / (H*W)) * 100;
    
    % Display results
    subplot(2,2,find(retention_levels==L));
    imshow(reconstructed);
    title(sprintf('L=%d (t=%d)\nCR= %.1f%%', L, t, compression_rate));
end