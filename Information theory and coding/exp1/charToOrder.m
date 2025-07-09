%
% Purpose: Convert a character array (string) to an array of integers 0 through 27 (the "order")
%          Order of 1 is a space or other character, Order 1-26 are the letters "a" through "z".
%
% Author: Neal Patwari
% 
% 主程序
text = fileread('English text data.txt');
order = cTO(text);

    % 计算H1
    prob = histcounts(order, 1:28) / length(order);
    H1 = -sum(prob .* log2(prob), 'omitnan');
    
    % 计算H2和条件熵
    joint_counts = zeros(27,27);
    for i = 1:length(order)-1
        current = order(i);
        next = order(i+1);
        joint_counts(current, next) = joint_counts(current, next) + 1;
    end
    joint_prob = joint_counts / sum(joint_counts(:));
    H_joint2 = -sum(joint_prob(:) .* log2(joint_prob(:)), 'omitnan');
    H2 = H_joint2 / 2;
    
    % 条件熵H(X2|X1)
    H_cond = 0;
    total_pairs = sum(joint_counts(:));
    for i = 1:27
        for j = 1:27
            if joint_counts(i,j) > 0
                p_cond = joint_counts(i,j) / sum(joint_counts(i,:));
                H_cond = H_cond - (joint_counts(i,j)/total_pairs) * log2(p_cond);
            end
        end
    end

% 输出结果
fprintf('H1 = %.4f bits/symbol\n', H1);
fprintf('H2 = %.4f bits/symbol\n', H2);
fprintf('1/2*H(X2|X1) = %.4f bits/symbol\n', H_cond/2);


function [y] = cTO(s)
    % 修正后的字符到序号的映射函数
    y = zeros(size(s));
    for i = 1:length(s)
        c = lower(s(i));
        if c == ' '
            y(i) = 27;
        elseif c >= 'a' && c <= 'z'
            y(i) = c - 'a' + 1;
        else
            y(i) = 0; % 非字母和空格标记为0
        end
    end
    y = y(y ~= 0); % 过滤无效字符
end



