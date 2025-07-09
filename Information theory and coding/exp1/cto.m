% Read and preprocess data
fid = fopen('English text data.txt', 'r');
textData = fscanf(fid, '%c')';
fclose(fid);

% Convert characters to sequential order
order = charToOrder(textData);

% Calculate H(X1)
prob = histcounts(order, 1:28) / length(order);
H1 = -sum(prob .* log2(prob), 'omitnan');

% Calculate H(X1, X2) and conditional entropy H(X2|X1)
joint_counts = zeros(27, 27);
for i = 1:length(order)-1
    current = order(i);
    next = order(i+1);
    joint_counts(current, next) = joint_counts(current, next) + 1;
end
joint_prob = joint_counts / sum(joint_counts(:));
H_joint2 = -sum(joint_prob(:) .* log2(joint_prob(:)), 'omitnan');

% Conditional entropy H(X2|X1)
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

% Calculate H(X1, X2, X3)
joint_counts3 = zeros(27, 27, 27);
for i = 1:length(order)-2
    current1 = order(i);
    current2 = order(i+1);
    current3 = order(i+2);
    joint_counts3(current1, current2, current3) = joint_counts3(current1, current2, current3) + 1;
end
joint_prob3 = joint_counts3 / sum(joint_counts3(:));
H_joint3 = -sum(joint_prob3(:) .* log2(joint_prob3(:)), 'omitnan');

% Calculate H(X1, X2, X3, X4)
joint_counts4 = zeros(27, 27, 27, 27);
for i = 1:length(order)-3
    current1 = order(i);
    current2 = order(i+1);
    current3 = order(i+2);
    current4 = order(i+3);
    joint_counts4(current1, current2, current3, current4) = joint_counts4(current1, current2, current3, current4) + 1;
end
joint_prob4 = joint_counts4 / sum(joint_counts4(:));
H_joint4 = -sum(joint_prob4(:) .* log2(joint_prob4(:)), 'omitnan');

% Output results
fprintf('H(X1) = %.4f bit/symbol\n', H1);
fprintf('1/2*H(X1, X2) = %.4f bit/symbol\n', H_joint2/2);
fprintf('H(X2|X1) = %.4f bit/symbol\n', H_cond);
fprintf('1/3*H(X1, X2, X3) = %.4f bit/symbol\n', H_joint3/3);
fprintf('1/4*H(X1, X2, X3, X4) = %.4f bit/symbol\n', H_joint4/4);

function [y] = charToOrder(s)
    % Function to convert characters to sequential order
    y = zeros(size(s));
    for i = 1:length(s)
        c = lower(s(i));
        if c == ' '
            y(i) = 27;
        elseif c >= 'a' && c <= 'z'
            y(i) = c - 'a' + 1;
        else
            y(i) = 0; % Mark non-alphabetic and non-space characters as 0
        end
    end
    y = y(y ~= 0); % Filter out invalid characters
end