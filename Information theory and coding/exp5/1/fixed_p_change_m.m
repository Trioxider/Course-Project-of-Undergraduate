%% 实验 4.2：不同 m 值的汉明码性能比较（统计码字错误率 FER）
clear; clc; close all;

% 固定参数
p = 0.1;                    % BSC 信道的错误概率
num_bits = 1e7;           % 每种 m 生成固定数量的码字
m_values = 3:7;             % 不同的 m 值

% 生成信息比特，总长度为num_bits
info_bits = rand(1, num_bits) > 0.5;

% 预分配结果存储
error_rates = zeros(size(m_values));
theoretical = zeros(size(m_values));

for idx = 1:length(m_values)
    m = m_values(idx);
    n = 2^m - 1;
    k = n - m;
    
    % 填充源码长度到 k 的整数倍
    pad_len = mod(k - mod(num_bits, k), k);
    info_bits_pad = [info_bits, zeros(1, pad_len)];

    % 编码
    encoded = hamming_encoder(info_bits_pad, m);

    % 通过 BSC 信道
    received = bsc(encoded, p);

    % 解码
    decoded = hamming_decoder(received, m);

    % 分组比较：每组 k 比特组成一个码字，是否完全还原
    info_matrix = reshape(info_bits_pad, k, []);
    decoded_matrix = reshape(decoded, k, []);
    error_rates(idx) = mean(any(info_matrix ~= decoded_matrix, 1));

    % 理论 FER（≥2位错误）
    theoretical(idx) = 1 - (1-p)^n - n*p*(1-p)^(n-1);
end

%% 可视化结果
figure;
plot(m_values, error_rates, 'bo-', 'LineWidth', 1.5); hold on;
plot(m_values, theoretical, 'r--', 'LineWidth', 1.5);
xlabel('校验位数 m');
ylabel('码字错误率');
legend('仿真值', '理论值', 'Location', 'southwest');
title(sprintf('固定 p = %.2f，不同 m 值下的 Hamming 码性能比较', p));
grid on;

