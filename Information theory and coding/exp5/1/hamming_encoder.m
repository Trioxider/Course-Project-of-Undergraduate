function C = hamming_encoder(U, m)
    % 计算参数
    k = 2^m - 1 - m;  % 信息位数
    
    % 填充信息序列
    remainder = mod(length(U), k);
    if remainder ~= 0
        padding_length = k - remainder;
        U = [U, zeros(1, padding_length)];  % 填充0
    end
    
    % 生成并调整G矩阵
    [H, G] = hammgen(m);
    % 调整H矩阵
    H = [H(:, m+1:end) H(:, 1:m)];
    % 调整G矩阵
    G = [G(:, end-k+1:end) G(:, 1:end-k)];
    
    % 分块编码
    num_blocks = length(U) / k;
    C = [];
    
    for i = 1:num_blocks
        % 提取当前信息块
        start_idx = (i-1)*k + 1;
        end_idx = i*k;
        u = U(start_idx:end_idx);
        
        % 编码：c = u * G (模2乘法)
        c = mod(u * G, 2);
        
        % 添加到输出
        C = [C, c];
    end
end

