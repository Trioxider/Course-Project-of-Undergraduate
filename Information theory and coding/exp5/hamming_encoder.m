function C = hamming_encoder(U, m)
    k = 2^m - 1 - m;
    n = 2^m - 1;
    % Pad zeros if necessary
    len_U = length(U);
    pad_len = mod(-len_U, k);
    U = [U, zeros(1, pad_len)];
    % Generate H and G
    [H, G] = hammgen(m);
    % Adjust column order
    H_adjusted = [H(:, m+1:end), H(:, 1:m)];
    G_adjusted = [G(:, m+1:end), G(:, 1:m)];
    % Encode each block
    num_blocks = length(U) / k;
    C = [];
    for i = 1:num_blocks
        u = U((i-1)*k + 1 : i*k);
        c = mod(u * G_adjusted, 2);
        C = [C, c];
    end
end