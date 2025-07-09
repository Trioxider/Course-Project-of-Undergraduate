load('MNIST_subset.mat');
% Load MNIST dataset, using only the first 1000 samples
X = tr_feats(1:1000, :); % Input feature matrix, each row is a 784-dimensional image vector
y = tr_label(1:1000); % Output label vector, each element is a digit from 0 to 9
y(y ~= 6) = -1; % Set non-6 digit labels to -1 (negative class)
y(y == 6) = 1; % Set digit 6 labels to 1 (positive class)

% Use the MNIST training and test sets directly without splitting
X_train = X; % Training feature matrix
y_train = y; % Training label vector
X_test = te_feats; % Test feature matrix
y_test = te_label; % Test label vector
y_test(y_test ~= 6) = -1; % Set non-6 digit labels to -1 (negative class)
y_test(y_test == 6) = 1; % Set digit 6 labels to 1 (positive class)

% Initialize parameters, generate weight vector and bias randomly
init_method = 'xavier'; % Options: 'normal', 'xavier', 'he', etc.

% Initialize parameters based on selected initialization method
% if strcmp(init_method, 'normal')
%     w = randn(784, 1); % Initialize with normal distribution
% elseif  strcmp(init_method, 'xavier')
%     w = xavier_init(784, 1); % Initialize with Xavier method
% elseif strcmp(init_method, 'he')
%     w = he_init(784, 1); % Initialize with He method
% end

w=1; % Set initial weight vector

% disp(w);
b = randn; % Bias term, a random number
eta = 0.0001; % Learning rate, a small positive number

goal = 0; % Target error, training stops when error is below this value
max_epochs = 100; % Maximum number of epochs, training stops when this is reached

err = 1;
epoch = 0;
epoch_count = 0;

train_errors = zeros(max_epochs,1);
test_errors = zeros(max_epochs,1);

% Train single-layer perceptron
while err > goal && epoch < max_epochs % Continue training while error is above target and epoch limit not reached
    err = 0; % Reset error
    epoch = epoch + 1; % Increment epoch count
    gradient_w = 0; % Compute gradient
    for i = 1:size(X_train, 1) % Loop over each training sample
        x = X_train(i, :)'; % Extract the i-th sample feature vector
        y = y_train(i); % Extract the i-th sample label
        y_pred = sign(w' * x + b); % Compute perceptron output using sign function as activation
        if y_pred ~= y % If prediction does not match label, it's a misclassification
            err = err + 1; % Increment error
        end
        gradient_w = -(y-y_pred).*x + gradient_w;
        w = w + eta * y * gradient_w; % Update weight vector
        b = b + eta * y; % Update bias term
    end
    err = err / size(X_train, 1); % Compute training error rate
    train_errors(epoch) = err;
    fprintf('Epoch %d, error rate: %.4f\n', epoch, err); % Print epoch and error rate

    % Test single-layer perceptron
    Error = 0; % Reset error
    y_true = []; % Vector for true labels
    y_pred = []; % Vector for predicted labels
    for i = 1:size(X_test, 1) % Loop over each test sample
        x = X_test(i, :)'; % Extract the i-th sample feature vector
        y = y_test(i); % Extract the i-th sample label
        y_hat = sign(w' * x + b); % Compute perceptron output using sign function
        if y_hat ~= y % If prediction does not match label, it's a misclassification
            Error = Error + 1; % Increment error
        end
        y_true = [y_true; y]; % Append true label
        y_pred = [y_pred; y_hat]; % Append predicted label     
    end
    Error = Error / size(X_test, 1); % Compute test error rate
    test_errors(epoch) = Error; % Calculate error rate of test subset
    fprintf('Test error rate: %.4f\n', Error); % Print error rate
end

% Compute additional metrics like accuracy, recall, F1 score, etc.
accuracy = sum(y_true == y_pred) / size(X_test, 1); % Accuracy: proportion of correct classifications
recall = sum(y_true == 1 & y_pred == 1) / sum(y_true == 1); % Recall: proportion of positive class correctly classified
precision = sum(y_true == 1 & y_pred == 1) / sum(y_pred == 1); % Precision: proportion of predicted positive class that is correct

fprintf('Accuracy: %.4f, Recall: %.4f, Precision: %.4f\n', accuracy, recall, precision); % Print evaluation metrics

figure;
plot(1:max_epochs, train_errors, 'b', 'DisplayName', 'Train error'); % Plot training error using blue line
hold on; % Hold current figure
plot(1:max_epochs, test_errors, 'r', 'DisplayName', 'Test error'); % Plot test error using red line
hold off; % Release hold
legend; % Show legend
title('Error rate over epoch'); % Add title
xlabel('Epoch'); % X-axis label
ylabel('Error rate'); % Y-axis label
