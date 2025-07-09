function decoded_info = hamming_decoder(R, m)
    k = 2^m - 1 - m;
    n = 2^m - 1;
    % Pad zeros if necessary
    len_R = length(R);
    pad_len = mod(-len_R, n);
    R = [R, zeros(1, pad_len)];
    % Generate adjusted H
    [H, ~] = hammgen(m);
    H_adjusted = [H(:, m+1:end), H(:, 1:m)];
    Ht = H_adjusted';
    % Decode each block
    num_blocks = length(R) / n;
    decoded_info = [];
    for i = 1:num_blocks
        r = R((i-1)*n + 1 : i*n);
        s = mod(r * Ht, 2);
        % Find error position
        e = zeros(1, n);
        found = false;
        for col = 1:size(Ht, 1)
            if isequal(s, Ht(col, :))
                e(col) = 1;
                found = true;
                break;
            end
        end
        c = mod(r + e, 2);
        decoded_info = [decoded_info, c(1:k)];
    end
end
