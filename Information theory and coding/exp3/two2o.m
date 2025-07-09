function out = two2o(bf)
% TWO2O  Convert the decimal-coded “1/2” stream from fano into a bit‑string + ‘–’
%   bf  – a positive integer whose decimal digits are all 1’s or 2’s
%         (1 means bit ‘0’, 2 means bit ‘1’)
%
% Returns a character vector like '0101-' ready for strsplit.

    str = num2str(bf);            % e.g. '21212'
    % map '1'→'0', '2'→'1'
    str(str=='1') = '0';
    str(str=='2') = '1';
    out = [str '-' ];             % append dash delimiter
end
