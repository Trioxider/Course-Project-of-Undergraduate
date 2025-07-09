clear;

% 3.1 Calculate the channel capacity for a binary symmetric channel (BSC)
Q = [0.8 0.2; 0.3 0.7]; % Transition probability matrix of the BSC
p_values = 0.001:0.001:0.999; % Range of probabilities to test for optimal p
max_I = 0; % Initialize maximum mutual information
optimal_p = 0; % Initialize optimal probability

% Pre-calculate conditional entropies given X1 and X2
H_Y_given_X1 = -sum(Q(1,:) .* log2(Q(1,:)));
H_Y_given_X2 = -sum(Q(2,:) .* log2(Q(2,:)));

% Initialize an array to store mutual information values for each p
mutual_information = zeros(size(p_values));

for idx = 1:length(p_values)
    p = p_values(idx);
    P_x = [p, 1-p]; % Probability distribution over input symbols
    
    % Compute output distribution
    P_y1 = P_x(1)*Q(1,1) + P_x(2)*Q(2,1);
    P_y2 = P_x(1)*Q(1,2) + P_x(2)*Q(2,2);
    P_y = [P_y1, P_y2];
    
    % Entropy of Y
    H_Y = -sum(P_y .* log2(P_y + eps)); % Add eps to avoid log(0)
    
    % Conditional entropy of Y given X
    H_Y_given_X = P_x(1)*H_Y_given_X1 + P_x(2)*H_Y_given_X2;
    
    % Average mutual information
    I = H_Y - H_Y_given_X;
    
    if I > max_I
        max_I = I;
        optimal_p = p;
    end
    
    % Store the calculated mutual information for plotting
    mutual_information(idx) = I;
end

disp(['Optimal p: ', num2str(optimal_p)]);
disp(['Channel Capacity: ', num2str(max_I)]);

% Plot the average mutual information against the input probability x1
figure;
plot(p_values, mutual_information, 'b-', 'LineWidth', 2); % Plot the data with blue line
xlabel('Input Probability p'); % Label for x-axis
ylabel('Average Mutual Information'); % Label for y-axis
title('Average Mutual Information vs Input Probability'); % Title of the plot
grid on; % Turn on grid for better readability

clear;

%% 3.2 Calculate the channel capacity for a ternary symmetric channel
Q = [0.8 0.15 0.05; 0.15 0.15 0.7; 0.6 0.3 0.1]; % Transition probability matrix
max_I = 0; % Initialize maximum mutual information
optimal_dist = [0, 0, 0]; % Initialize optimal input distribution

% Pre-calculate conditional entropies for each row in Q
H_Y_given_X = zeros(3, 1);
for i = 1:3
    H_Y_given_X(i) = -sum(Q(i,:) .* log2(Q(i,:) + eps));
end

% Define range of probabilities to test for optimal p
x1_values = 0.001:0.001:0.999;
x2_values = 0.001:0.001:0.999;

% Initialize a matrix to store mutual information values for plotting
mutual_information = zeros(length(x1_values), length(x2_values));

% Iterate over all possible combinations of x1 and x2
for idx1 = 1:length(x1_values)
    for idx2 = 1:length(x2_values)
        x1 = x1_values(idx1);
        x2 = x2_values(idx2);
        if (x1 + x2 < 1) % Ensure that x1 + x2 + x3 = 1
            x3 = 1 - x1 - x2;
            P_x = [x1, x2, x3];
            
            % Compute output distribution
            P_y = P_x * Q;
            
            % Entropy of Y
            H_Y = -sum(P_y .* log2(P_y + eps));
            
            % Conditional entropy of Y given X
            H_YgX = sum(P_x .* H_Y_given_X');
            
            % Average mutual information
            I = H_Y - H_YgX;
            
            if I > max_I
                max_I = I;
                optimal_dist = [x1, x2, x3];
            end
            
            % Store the calculated mutual information for plotting
            mutual_information(idx1, idx2) = I;
        else
            mutual_information(idx1, idx2) = NaN; % No valid distribution
        end
    end
end

disp('Optimal Input Distribution:');
disp(optimal_dist);
disp(['Channel Capacity: ', num2str(max_I)]);

% Create a meshgrid for plotting
[X1, X2] = meshgrid(x1_values, x2_values);

% Plot the average mutual information against the input probabilities x1 and x2
figure;
surf(X1, X2, mutual_information, 'EdgeColor', 'none'); % Surface plot without edges
xlabel('Input Probability x1'); % Label for x-axis
ylabel('Input Probability x2'); % Label for y-axis
zlabel('Average Mutual Information'); % Label for z-axis
title('Average Mutual Information vs Input Probabilities x1 and x2'); % Title of the plot
colorbar; % Add color bar to show the scale of AMI
view(3); % Set the view angle to 3D
grid on; % Turn on grid for better readability

%% 3.3 Iterative algorithm for computing channel capacity with arbitrary transition matrix
% Assume Q is loaded as a 5x5 matrix from 'data3.3.mat'
data = load('data3.3.mat', 'p');
Q1 = data.p; % Force conversion to numeric matrix
K = size(Q1, 1); % Number of input symbols
P = ones(1, K) / K; % Initial uniform distribution
epsilon = 1e-6; % Convergence threshold
I_L = 0; % Lower bound initialization
I_U = 1; % Upper bound initialization
iter = 0; % Iteration counter

while (I_U - I_L) >= epsilon
    % Compute beta_i for each input symbol
    beta = zeros(1, K);
    for i = 1:K
        sum_term = 0;
        for j = 1:size(Q1, 2)
            denominator = sum(P .* Q1(:, j)');
            if denominator == 0
                continue;
            end
            term = Q1(i, j) * log2(Q1(i, j) / denominator);
            sum_term = sum_term + term;
        end
        beta(i) = exp(sum_term);
    end
    
    % Update lower and upper bounds
    I_L = log(sum(P .* beta));
    I_U = log(max(beta));
    
    % Update input distribution
    P = P .* beta / sum(P .* beta);
    
    iter = iter + 1;
end

disp(['Channel Capacity: ', num2str(I_L / log(2))]); % Convert to bits
disp('Optimal Input Distribution:');
disp(P);

%% 3.4 Similar iterative algorithm but for a larger 100x100 transition matrix
clear;
% Load the 100x100 channel transition matrix from 'data3.4.mat'
data = load('data3.4.mat', 'p');
Q = data.p;             % Ensure Q is a numeric matrix
[M, N] = size(Q);       % M: input symbols, N: output symbols
epsilon = 1e-6;         % Convergence threshold
P = ones(M, 1) / M;     % Initial uniform distribution (column vector)
Qy = Q * P;             % Initial output distribution
I_U = 1;                % Upper bound of mutual information
I_L = 0;                % Lower bound of mutual information
iter = 0;               % Iteration counter

while (I_U - I_L) > epsilon
    % Step 1: Compute F(j) = exp(sum(Q(:,j) .* log(Q(:,j) ./ Qy))
    F = zeros(M, 1);
    for j = 1:M
        term = 0;
        for k = 1:N
            if Q(k, j) > 0
                term = term + Q(k, j) * log2(Q(k, j) / Qy(k));
            end
        end
        F(j) = exp(term);
    end
    
    % Step 2: Update bounds
    x = F' * P;
    I_L_new = log(x);
    I_U_new = log(max(F));
    
    % Step 3: Check convergence
    if (I_U_new - I_L_new) < epsilon
        I_L = I_L_new;
        break;
    else
        % Update input distribution
        P = (F .* P) / x;   % Normalization
        Qy = Q * P;         % Recompute output distribution
        I_L = I_L_new;
        I_U = I_U_new;
        iter = iter + 1;
    end
end

% Convert to bits and calculate mutual information
C = I_L / log(2);
I = 0;
for j = 1:M
    for k = 1:N
        if Q(k, j) > 0 && Qy(k) > 0
            I = I + P(j) * Q(k, j) * log2(Q(k, j) / Qy(k));
        end
    end
end

% Output results
disp(['Channel Capacity: ', num2str(C)]);
disp(['Mutual Information: ', num2str(I)]);
disp(['Iterations: ', num2str(iter)]);
disp('Optimal Input Distribution (Top 5 Values):');
disp(P(1:5)');

%% 3.5 Calculate channel capacity using linear equations for a 5x5 matrix
clear;
data = load('data3.5.mat', 'p');
Q3 = data.p; % Force conversion to numeric matrix
A = sum(Q3 .* log2(Q3 + eps), 2); % Left-hand side of the equation
beta = Q3 \ A; % Solve linear equation
C = log2(sum(2.^beta));
disp(['Channel Capacity (Linear): ', num2str(C)]);

%% 3.5_1 Further refine the computation using matrix operations for the same 5x5 matrix
clear
% Load the 5x5 channel transition matrix from 'data3.5.mat'
data = load('data3.5.mat', 'p');
Q3 = data.p; 

% Construct the linear equations A = Q * beta
% Compute A_i = sum_j Q(i,j) * log2(Q(i,j)) for each row i
A = sum(Q3 .* log2(Q3 + eps), 2); % Add eps to avoid log(0)

% Solve beta = Q^{-1} * A using matrix inversion
beta = Q3 \ A;

% Compute channel capacity C = log2(sum(2.^beta))
C = log2(sum(2.^beta));

% Compute optimal output distribution P_b
P_b = 2.^(beta - C);

% Solve Q^T * P_a = P_b for optimal input distribution P_a
% P_a = (Q^T)^{-1} * P_b
P_a = (Q3' \ P_b);

% Ensure P_a is a valid probability distribution (sums to 1, non-negative)
P_a = P_a / sum(P_a); % Normalize
P_a = max(P_a, 0);    % Enforce non-negativity

% Display results
fprintf('Channel Capacity C = %.4f bits\n', C);
disp('Optimal Input Distribution P(a_i):');
disp(P_a');
disp('Optimal Output Distribution P(b_j):');
disp(P_b');