clear;
clc;
warning off;
% the above two commands clear all the previous record in the Memory
% counterparty action: 0 trust, others betray
% your action: 0 trust, others call police

N_trades = 100;                                       % trade N_trade times
N_games = 500;                                        % number of games
Return_total_all = zeros(N_games, 1);                 % total return
counterparty_betray_prob_total = zeros(N_games, 1);   % estimated probability of betrayal by the counterparty in all games

% evaluation of your strategy
Your_Strategy_Return_total_all = zeros(N_games, 1); % total return of your strategy in all games    

for n_game = 1:N_games
    % initialize each game
    Return_total = 0;
    % initialize the probability of betrayal of the counterparty following a uniform distribution in (0, 1)
    counterparty_betray_prob = unifrnd(0, 1);
    
    % initialize the counterparty's historical actions
    counterparty_previous_action_list = rand(10, 1);
    counterparty_previous_action = double(counterparty_betray_prob > counterparty_previous_action_list);
    counterparty_action_total = counterparty_previous_action;
    
    for n_trade = 1:N_trades
        % initialize the new action of counterparty 
        if n_trade > 10
            counterparty_action = double(counterparty_betray_prob > rand(1));
        else
            counterparty_action = counterparty_previous_action(n_trade);
        end
        
        Your_Strategy = Your_Strategies(counterparty_previous_action);
        % this time, you can pass anything you want into the 'Your_Strategies',
        % except the 'counterparty_betray_prob'
        % you can change the whole system as you wish
        if Your_Strategy == 0
            if counterparty_action == 0
                Return_current = 10;        % both trust, add 10 points
            else
                Return_current = -5;       % self trust, counterparty betray, -10 points
            end
        else
                Return_current = 0;       % self reject, counterparty trust or trust, 0 points
        end
        
        % update the return
        Return_total = Return_total + Return_current;
        
        % update the counterparty's historical actions
        % no need to update the historical action table for the first 10 trades
        if n_trade > 10
            counterparty_action_total(end + 1) = counterparty_action;
        end
        
        % update estimated probability of betrayal by the counterparty
        counterparty_betray_prob_mean = mean(counterparty_action_total);
    end
    
    % record game results
    Return_total_all(n_game) = Return_total;
    counterparty_betray_prob_total(n_game) = counterparty_betray_prob_mean;
    Your_Strategy_Return_total_all(n_game) = Return_total;
    disp(['Game ', num2str(n_game), ' Estimated Total Return: ', num2str(Return_total)]);
    disp(['Game ', num2str(n_game), ' Estimated Average Counterparty Betrayal Probability: ', num2str(counterparty_betray_prob_mean)]);
end

% calculate average payoff and estimated probability of betrayal by the counterparty
Average_Return = mean(Return_total_all);
Average_counterparty_betray_prob = mean(counterparty_betray_prob_total);

% display results
disp(['Average Payoff: ', num2str(Average_Return)]);
disp(['Average Estimated Counterparty Betrayal Probability: ', num2str(Average_counterparty_betray_prob)]);

% plot payoff distribution
figure;
histogram(Your_Strategy_Return_total_all);
xlabel('Payoff');
ylabel('Number of Games');
title('Payoff Distribution of Your Strategy');
