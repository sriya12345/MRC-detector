close all; clc; clear;

%% Step 1: OTFS Parameters & Grid Design
M = 4;                      % Number of subcarriers
N = 64;                     % Number of time slots
Q = 4;                      % 4-QAM
bits_per_sym = log2(Q);

%% Channel Parameters
v = 500*(1000/3600);        % Speed (m/s)
fc = 4e9;                   % Carrier frequency
c = 3e8;                    % Speed of light
fd = (v/c)*fc;              % Max Doppler frequency

% Hardcoded delay and doppler taps (as per your requirement)
l = [0, 1, 2, 3];           % Delay taps
k = [0, 1, 2, 3];           % Doppler taps
P = length(l);              % Number of paths
L_zp = max(l);              % Zero-padding length
M_prime = M + L_zp;         % Extended grid dimension for ZP-OTFS

%% Step 2: Signal Generation & Modulation
num_sym = N * M;
bits = randi([0 1], num_sym * bits_per_sym, 1);

% QAM Modulation
symbols = qammod(bits, Q, 'InputType', 'bit', 'UnitAveragePower', true);

% DD transmit matrix (First M blocks contain data, last L_zp blocks are zero)
X_dd_data = reshape(symbols, M, N); 
X_tx_vec = zeros(N * M_prime, 1);
X_tx_vec(1:N*M) = symbols;  % Map to vectorized format for processing

%% Step 3: Channel Generation (Matrix Formulation)
% Channel gains (Rayleigh fading)
h = (randn(1, P) + 1j*randn(1, P)) / sqrt(2);
h = h / norm(h); % Normalize power

% Build extended channel subblocks K_{m,l}
% K is a cell array where K{m, p} represents the N x N block 
% for symbol block m and path p
K = cell(M_prime, P); 
for m_idx = 1:M_prime
    for p = 1:P
        % Define Doppler vector for the circulant matrix
        % nu_{m,l}[k] = g_i * exp(j2pi*k*(m-l)/MN)
        nu_ml = zeros(N, 1);
        % Position k(p)+1 handles the Doppler shift index
        nu_ml(mod(k(p), N) + 1) = h(p) * exp(1j * 2 * pi * k(p) * (m_idx - 1 - l(p)) / (M * N)); 
        
        % Create the N x N circulant block matrix
        K{m_idx, p} = gallery('circul', nu_ml).'; 
    end
end

%% Step 4: Channel Transmission with Noise
SNR_dB = 20; 
sigma_w_2 = 10^(-SNR_dB/10);

% Generate Received Signal Y_vec = H * X_vec
Y_vec = zeros(N * M_prime, 1);
for target_m = 1:M_prime
    for p = 1:P
        source_m = target_m - l(p);
        if source_m >= 1 && source_m <= M
            idx_x = (source_m-1)*N + (1:N);
            idx_y = (target_m-1)*N + (1:N);
            Y_vec(idx_y) = Y_vec(idx_y) + K{target_m, p} * X_tx_vec(idx_x);
        end
    end
end

% Add AWGN
noise = sqrt(sigma_w_2/2) * (randn(size(Y_vec)) + 1j*randn(size(Y_vec)));
Y_vec = Y_vec + noise;

%% Step 5: MRC Pre-calculations (Calculated ONCE)
% Calculate Dm for each symbol block m = 1 to M
Dm = cell(M, 1);
for m = 1:M
    sum_K_sq = zeros(N, N);
    for p = 1:P
        target_m = m + l(p);
        if target_m <= M_prime
            % Sum of K' * K for all delay branches
            sum_K_sq = sum_K_sq + K{target_m, p}' * K{target_m, p};
        end
    end
    Dm{m} = sum_K_sq;
end

%% Step 6: Iterative MRC Detection (Algorithm 2)
max_iter = 5; 
X_hat_vec = zeros(N * M_prime, 1); % Initial estimates = 0

for iter = 1:max_iter
    for m = 1:M 
        gm = zeros(N, 1);
        
        % 1. Extract and combine components from all delay branches
        for p = 1:P
            target_m = m + l(p);
            if target_m > M_prime, continue; end
            
            idx_y = (target_m-1)*N + (1:N);
            
            % 2. Interference Cancellation (Decision Feedback)
            % Subtract interference from other paths/symbols
            interference = zeros(N, 1);
            for p_prime = 1:P
                source_m = target_m - l(p_prime);
                % If source_m == m and p_prime == p, this is our signal, don't subtract
                if (source_m == m && p_prime == p), continue; end
                
                if source_m >= 1 && source_m <= M
                    idx_curr_x = (source_m-1)*N + (1:N);
                    interference = interference + K{target_m, p_prime} * X_hat_vec(idx_curr_x);
                end
            end
            
            % Cleaned branch signal
            blm = Y_vec(idx_y) - interference;
            % Maximal Ratio Combining
            gm = gm + K{target_m, p}' * blm;
        end
        
        % 3. Update block estimate using pre-calculated Dm
        cm = Dm{m} \ gm; 
        
        % 4. Hard Decision (MLD mapping)
        idx_m = (m-1)*N + (1:N);
        X_hat_vec(idx_m) = qammod(qamdemod(cm, Q, 'UnitAveragePower', true), ...
                                 Q, 'UnitAveragePower', true);
    end
end

%% Step 7: BER Computation
% Extract the data blocks (1 to M)
estimated_symbols = X_hat_vec(1:N*M);

% Demodulate to bits
bits_hat = qamdemod(estimated_symbols, Q, 'UnitAveragePower', true, 'OutputType', 'bit');

% Calculate Bit Error Rate
bit_errors = sum(bits ~= bits_hat);
BER = bit_errors / length(bits);

%% Display Results
fprintf('\n=================================\n');
fprintf('   ZP-OTFS MRC DETECTOR RESULTS  \n');
fprintf('=================================\n');
fprintf('SNR              : %d dB\n', SNR_dB);
fprintf('Total Bits       : %d\n', length(bits));
fprintf('Bit Errors       : %d\n', bit_errors);
fprintf('Measured BER     : %e\n', BER);
fprintf('Iterations       : %d\n', max_iter);
fprintf('=================================\n');