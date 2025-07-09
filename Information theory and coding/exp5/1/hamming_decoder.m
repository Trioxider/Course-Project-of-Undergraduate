function U_decoded = hamming_decoder(R, m)
    % 参数计算
    k = 2^m - 1 - m;  % 信息位数
    n = 2^m - 1;      % 码字长度
    
    % 生成校验矩阵H
    [H, ~] = hammgen(m);
    H = [H(:, m+1:end) H(:, 1:m)];  % 调整格式
    
    % 确保接收序列长度是n的整数倍
    if mod(length(R), n) ~= 0
        error('接收序列长度必须是%d的整数倍', n);
    end
    
    % 分块处理
    num_blocks = length(R) / n;
    U_decoded = [];
    
    for i = 1:num_blocks
        % 提取当前码字块
        start_idx = (i-1)*n + 1;
        end_idx = i*n;
        r = R(start_idx:end_idx);
        
        % 计算伴随式
        s = mod(r * H', 2);

        % 使用ismember查找错误位置
        [index, ~] = ismember(H', s, 'rows');
        
        % 生成错误模式向量E
        E = zeros(1, n);
        if any(index)
            error_pos = find(index, 1);
            E(error_pos) = 1;
        end
        
        % 纠正错误: c = r + E (模2)
        c = mod(r + E, 2);
        
        % 提取信息位(前k位)
        u = c(1:k);
        
        % 添加到输出
        U_decoded = [U_decoded, u];
    end
end
