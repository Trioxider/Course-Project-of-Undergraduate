%
% Purpose: Convert a character array (string) to an array of integers 0 through 27 (the "order")
%          Order of 1 is a space or other character, Order 1-26 are the letters "a" through "z".
%
% Author: Neal Patwari
% 

function [y] = charToOrder(s)

y = s - 95;
y(find(y < 0))  = 1;
y(find(y > 27)) = 1;
