close all; clc; clear;

%% Step 1: OTFS Parameters
M = 32;                      
N = 64;                     
Q = 4;                      
bits_per_sym = log2(Q);
P = 4;
l = randi([0 M-1], 1, P);
k = randi([0 N-1], 1, P);                         
L_zp = max(l);              
M_prime = M + L_zp;         

%% Simulation Parameters
SNR_dB_range = 0:2:30;      % SNR range for the plot
max_frames = 100;           % Number of frames to average per SNR point
max_iter = 10;               % MRC iterations

BER_results = zeros(length(SNR_dB_range), 1);

%% Outer Loop: SNR Range
for snr_idx = 1:length(SNR_dB_range)
    current_snr = SNR_dB_range(snr_idx);
    sigma_w_2 = 10^(-current_snr/10);
    total_bit_errors = 0;
    
    fprintf('Simulating SNR = %d dB... ', current_snr);
    
    %% Inner Loop: Monte Carlo Frames
    for frame = 1:max_frames
        % 1. Generate Bits and Symbols
        bits = randi([0 1], N * M * bits_per_sym, 1);
        symbols = qammod(bits, Q, 'InputType', 'bit', 'UnitAveragePower', true);
        X_tx_vec = zeros(N * M_prime, 1);
        X_tx_vec(1:N*M) = symbols;                                                % ZP

        l = randi([0 M-1], 1, P);
        k = randi([0 N-1], 1, P);

        % 2. Channel Generation (Rayleigh)
        h = (randn(1, P) + 1j*randn(1, P)) / sqrt(2);
        %h = h / norm(h);

        % 3. Build K matrices
        K = cell(M_prime, P);                                                      % N x N submatrix
        for m_idx = 1:M_prime
            for p = 1:P
                nu_ml = zeros(N, 1);                                               % doppler response vector
                nu_ml(mod(k(p), N) + 1) = h(p) * exp(1j * 2 * pi * k(p) * (m_idx - 1 - l(p)) / (M * N)); 
                K{m_idx, p} = gallery('circul', nu_ml).';                          % circulant for m>l
            end
        end

        % 4. Pre-calculate Dm
        Dm = cell(M, 1);
        for m = 1:M
            sum_K_sq = zeros(N, N);                                                % Dm: N x N matrix
            for p = 1:P
                target_m = m + l(p);
                if target_m <= M_prime
                    sum_K_sq = sum_K_sq + K{target_m, p}' * K{target_m, p};
                end
            end
            Dm{m} = sum_K_sq;
        end

        % 5. Transmission
        Y_vec = zeros(N * M_prime, 1);                                             % NM'x1 Rx signal vector
        for target_m = 1:M_prime
            for p = 1:P
                source_m = target_m - l(p);
                if source_m >= 1 && source_m <= M
                    idx_x = (source_m-1)*N + (1:N);                                 % Doppler vector at delay = source_m
                    idx_y = (target_m-1)*N + (1:N);                                 % Doppler vector at delay = target_m
                    Y_vec(idx_y) = Y_vec(idx_y) + K{target_m, p} * X_tx_vec(idx_x); 
                end
            end
        end
        noise = sqrt(sigma_w_2/2) * (randn(size(Y_vec)) + 1j*randn(size(Y_vec)));
        Y_vec = Y_vec + noise;

        % 6. Iterative MRC Detection
        X_hat_vec = zeros(N * M_prime, 1);
        for iter = 1:max_iter
            for m = 1:M 
                gm = zeros(N, 1);
                for p = 1:P
                    target_m = m + l(p);
                    if target_m > M_prime, continue; end
                    idx_y = (target_m-1)*N + (1:N);
                    
                    interference = zeros(N, 1);
                    for p_prime = 1:P
                        source_m = target_m - l(p_prime);
                        if (source_m == m && p_prime == p), continue; end
                        if source_m >= 1 && source_m <= M
                            idx_curr_x = (source_m-1)*N + (1:N);
                            interference = interference + K{target_m, p_prime} * X_hat_vec(idx_curr_x);
                        end
                    end
                    blm = Y_vec(idx_y) - interference;
                    gm = gm + K{target_m, p}' * blm;
                end
                cm = Dm{m} \ gm; 
                idx_m = (m-1)*N + (1:N);
                X_hat_vec(idx_m) = qammod(qamdemod(cm, Q, 'UnitAveragePower', true), ...
                                         Q, 'UnitAveragePower', true);
            end
        end

        % 7. Error Counting
        estimated_symbols = X_hat_vec(1:N*M);
        bits_hat = qamdemod(estimated_symbols, Q, 'UnitAveragePower', true, 'OutputType', 'bit');
        total_bit_errors = total_bit_errors + sum(bits ~= bits_hat);
    end
    
    BER_results(snr_idx) = total_bit_errors / (max_frames * N * M * bits_per_sym);
    fprintf('BER: %e\n', BER_results(snr_idx));
end

%% Step 8: Plotting
figure;
semilogy(SNR_dB_range, BER_results, 'b-o', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('BER Performance of ZP-OTFS with Iterative MRC Detection');
legend(['Iterations = ', num2str(max_iter)]);