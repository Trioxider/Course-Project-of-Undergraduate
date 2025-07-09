%% 4.1 - 均匀分布 (信息比特独立同分布 p=0.5)
clear; 
clc;

m = 3;                      % 汉明码参数
n = 2^m - 1;                % 码字长度
k = n - m;                  % 信息位长度
num_bits = 1e7;             % 信息比特数
p_values = 0:0.01:0.2;      % BSC错误概率范围

% 生成随机比特流（均匀分布）
info_bits = rand(1, num_bits) > 0.5;

% 补齐长度到 k 的整数倍
pad_len = mod(k - mod(num_bits, k), k);
info_bits_pad = [info_bits, zeros(1, pad_len)];

% 编码
encoded = hamming_encoder(info_bits_pad, m);

% 分块原始信息用于统计 FER
info_matrix = reshape(info_bits_pad, k, []);

% 初始化误码率数组
error_rates = zeros(size(p_values));

for i = 1:length(p_values)
    p = p_values(i);

    % BSC信道
    received = bsc(encoded, p);

    % 解码
    decoded = hamming_decoder(received, m);
    decoded_matrix = reshape(decoded, k, []);

    % 统计码字错误率（整组出错算1个）
    error_rates(i) = mean(any(info_matrix ~= decoded_matrix, 1));
end

% 理论误码率（m=3）
theoretical = 1 - (1 - p_values).^n - n .* p_values .* (1 - p_values).^(n - 1);

% 绘图
figure;
plot(p_values, error_rates, 'b-o', 'LineWidth', 1.5); hold on;
plot(p_values, theoretical, 'r--', 'LineWidth', 1.5);
xlabel('BSC交叉概率 p');
ylabel('码字错误率');
title('m=3，Hamming码性能 (信息比特均匀分布)');
legend('仿真值', '理论值', 'Location', 'northwest');
grid on;

%% 4.1 - 非均匀分布 (信息比特非 i.i.d.，p(1)=0.8)
info_bits_nonuni = rand(1, num_bits) > 0.2;  % P(1) = 0.8

% 补齐到 k 的整数倍
pad_len = mod(k - mod(num_bits, k), k);
info_bits_nonuni_pad = [info_bits_nonuni, zeros(1, pad_len)];

% 编码
encoded_nonuni = hamming_encoder(info_bits_nonuni_pad, m);
info_matrix_nonuni = reshape(info_bits_nonuni_pad, k, []);
error_rates_nonuni = zeros(size(p_values));

for i = 1:length(p_values)
    p = p_values(i);

    % BSC 信道
    received_nonuni = bsc(encoded_nonuni, p);

    % 解码
    decoded_nonuni = hamming_decoder(received_nonuni, m);
    decoded_matrix_nonuni = reshape(decoded_nonuni, k, []);

    % 统计码字错误率
    error_rates_nonuni(i) = mean(any(info_matrix_nonuni ~= decoded_matrix_nonuni, 1));
end

% 绘图
figure;
plot(p_values, error_rates_nonuni, 'g-s', 'LineWidth', 1.5); hold on;
plot(p_values, theoretical, 'r--', 'LineWidth', 1.5);
xlabel('BSC交叉概率 p');
ylabel('码字错误率');
title('m=3，Hamming码性能 (信息比特非均匀分布)');
legend('仿真值', '理论值', 'Location', 'northwest');
grid on;

