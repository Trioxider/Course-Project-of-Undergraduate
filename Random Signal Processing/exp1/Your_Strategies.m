% your_strategy returns your strategy of the trade this time
% your_strategy = 0 means that you want to trust the counterparty this time
% your_strategy not equal to 0 means that you want to betray the
% counterparty this time 
%%
% counterparty_id is the ID of the counterparty you are going to trade with
% this time
function Your_Strategy = Your_Strategies(counterparty_previous_action)
    % Record the counterparty's current action
    counterparty_preact_now = counterparty_previous_action;
    % Implement Tit-for-Tat strategy:
    % Cooperate if the counterparty cooperates, and betray if the counterparty betrays.
    if counterparty_preact_now == 0
       Your_Strategy = 0;
    else
       Your_Strategy = 1;
    end
end