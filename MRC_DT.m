clc; close all; clear;

%% Step 1: OTFS Parameters + Grid design
M = 4;                      % Number of subcarriers
N = 64;                     % Number of time slots
Q = 4;                      % 4-QAM
bits_per_sym = log2(Q);

%% Channel Parameters
l = [0, 1, 2, 3];           % Delay taps
k = [0, 1, 2, 3];           % Doppler taps
P = length(l);              % No. of paths
L_zp = max(l);              % ZP length
M_prime = M + L_zp;     

%% Monte Carlo Setup
SNR_dB_range = 0:5:25;      % SNR range to simulate
num_MC = 1000;              % No. of Monte Carlo iterations per SNR
iter_max = 10;              % No. of MRC iterations

BER_results = zeros(length(SNR_dB_range), 1);
num_sym = N * M;
total_bits_per_run = num_sym * bits_per_sym;

% Pre-calculate phase rotation term (Static for all runs)
z = exp(1i*2*pi / (N*M_prime));

fprintf('Starting Monte Carlo Simulation...\n');

%% Main Simulation Loop
for snr_idx = 1:length(SNR_dB_range)
    SNR_dB = SNR_dB_range(snr_idx);
    SNR_linear = 10^(SNR_dB/10);
    
    total_errors = 0;
    total_bits = 0;
    
    for mc_idx = 1:num_MC
        
        %% Step 2: Signal Generation & Modulation
        bits = randi([0 1], total_bits_per_run, 1);
        symbols = qammod(bits, Q, 'InputType', 'bit', 'UnitAveragePower', true);
        
        X_dd_data = reshape(symbols, M, N);                 % DD domain
        X_dt_data = ifft(X_dd_data, N, 2) * sqrt(N);        % DT domain
        X_dt_zp = [X_dt_data; zeros(L_zp, N)];              % ZP insertion
        s = X_dt_zp(:);                                     % Time domain (NM' x 1) 
        
        %% Step 3: Channel Generation 
        h = (randn(1, P) + 1j*randn(1, P)) / sqrt(2);       % Rayleigh fading
        h = h / norm(h);                                    % Normalize power
        
        gs = zeros(L_zp + 1, N * M_prime);
        for q = 0:(N*M_prime)-1
            for i = 1:P
                gs(l(i)+1, q+1) = gs(l(i)+1, q+1) + h(i) * z^(k(i) * (q - l(i)));
            end
        end
        
        % Build Time Domain Channel Matrix
        G = zeros(N*M_prime, N*M_prime);                    
        for q = 0:(N*M_prime)-1
            for ell = 0:L_zp
                if (q >= ell)
                    G(q+1, q-ell+1) = gs(ell+1, q+1);
                end
            end
        end
        
        %% Step 4: Channel Transmission
        signal_power = mean(abs(s).^2);
        noise_var = signal_power / SNR_linear;
        noise = sqrt(noise_var/2) * (randn(N*M_prime, 1) + 1i*randn(N*M_prime, 1));
        
        r = G * s + noise;                                  % Received signal
        
        %% Step 5: Receiver Setup (DT Domain)
        Y_dt_zp = reshape(r, M_prime, N);                   
        
        % Extract N x 1 Delay-Time Channel Vectors
        nu_dt = cell(M_prime, L_zp + 1);                    
        for l_idx = 0:L_zp 
            for m = 1:M_prime
                q_indices = (0:N-1) * M_prime + (m - 1); 
                nu_dt{m, l_idx + 1} = gs(l_idx + 1, q_indices + 1).'; 
            end
        end
        
        %% Step 6: MRC Algorithm Implementation 
        d_m = zeros(M, N);                  
        for m = 1:M
            d_temp = zeros(N, 1);
            for l_idx = 0:L_zp
                nu = nu_dt{m + l_idx, l_idx + 1}; 
                d_temp = d_temp + abs(nu).^2;
            end
            d_m(m, :) = (d_temp + noise_var).'; 
        end
        
        X_dt_est = zeros(M, N);            
        y_residual = Y_dt_zp;               
        
        for iter = 1:iter_max
            for m = 1:M
                x_prev = X_dt_est(m, :).';   
                d_m_vec = d_m(m, :).';       
                
                g_m = zeros(N, 1);           
                for l_idx = 0:L_zp
                    nu = nu_dt{m + l_idx, l_idx + 1}; 
                    y_ml = y_residual(m + l_idx, :).'; 
                    g_m = g_m + conj(nu) .* y_ml;
                end
                
                c_m = x_prev + (g_m ./ d_m_vec);  
                x_m_new = c_m; 
                
                delta_x = x_m_new - x_prev;
                X_dt_est(m, :) = x_m_new.';
                
                for l_idx = 0:L_zp
                    nu = nu_dt{m + l_idx, l_idx + 1};
                    y_residual(m + l_idx, :) = y_residual(m + l_idx, :) - (nu .* delta_x).';
                end
            end
        end
        
        %% Step 7: Demodulation & BER
        X_dd_est = fft(X_dt_est, N, 2) / sqrt(N);
        symbols_est = X_dd_est(:);
        bits_est = qamdemod(symbols_est, Q, 'OutputType', 'bit', 'UnitAveragePower', true);
        
        [num_err, ~] = biterr(bits, bits_est);
        total_errors = total_errors + num_err;
        total_bits = total_bits + total_bits_per_run;
    end
    
    % Calculate and store average BER for this SNR
    BER_results(snr_idx) = total_errors / total_bits;
    fprintf('SNR: %2d dB | Average BER: %e \n', SNR_dB, BER_results(snr_idx));
end

%% Plot Results
figure;
semilogy(SNR_dB_range, BER_results, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('ZP-OTFS Delay-Time SISO MRC Performance');