clc; close all; clear;

%% Step 1: OTFS Parameters + Grid design
M = 4;                      % Number of subcarriers
N = 64;                     % Number of time slots
Q = 4;                      % 4-QAM
bits_per_sym = log2(Q);

%% Channel Parameters
v = 500*(1000/3600);        % Speed (m/s)
fc = 4e9;                   % Carrier frequency
c = 3e8;                    % Speed of light
fd = (v/c)*fc;              % Max Doppler frequency

l = [0, 1, 2, 3];           % Delay taps
k = [0, 1, 2, 3];           % Doppler taps
P = length(l);              % No. of paths
L_zp = max(l);              % ZP length
M_prime = M + L_zp;     

%% Step 2: Signal Generation & Modulation
num_sym = N * M;
bits = randi([0 1], num_sym * bits_per_sym, 1);

% QAM Modulation
symbols = qammod(bits, Q, 'InputType', 'bit', 'UnitAveragePower', true);

X_dd_data = reshape(symbols, M, N);                 % DD domain
X_dt_data = ifft(X_dd_data, N, 2) * sqrt(N);        % DT domain
X_dt_zp = [X_dt_data; zeros(L_zp, N)];              % ZP insertion
s = X_dt_zp(:);                                     % Time domain (NM' x 1) 

%% Step 3: Channel Generation (Matrix Formulation)
h = (randn(1, P) + 1j*randn(1, P)) / sqrt(2);       % Rayleigh fading
h = h / norm(h);

z = exp(1i*2*pi / (N*M_prime));                     %% DOUBT
gs = zeros(L_zp + 1, N * M_prime);

for q = 0:(N*M_prime)-1
    for i = 1:P
        gs(l(i)+1, q+1) = gs(l(i)+1, q+1) + h(i) * z^(k(i) * (q - l(i)));
    end
end

G = zeros(N*M_prime, N*M_prime);                    % Time domain channel matrix
for q = 0:(N*M_prime)-1
    for ell = 0:L_zp
        if (q >= ell)
            G(q+1, q-ell+1) = gs(ell+1, q+1);
        end
    end
end

%% Step 4: Channel Transmission
SNR_dB = 5; 
SNR_linear = 10^(SNR_dB/10);
signal_power = mean(abs(s).^2);
noise_var = signal_power / SNR_linear;
noise = sqrt(noise_var/2) * (randn(N*M_prime, 1) + 1i*randn(N*M_prime, 1));

r = G * s + noise;                                  % Time domain Rx signal

%% Step 5: Receiver Setup for Algorithm 3 (DT Domain)
Y_dt_zp = reshape(r, M_prime, N);                   % DT domain

% Extract N x 1 Delay-Time Channel Vectors
nu_dt = cell(M_prime, L_zp + 1);                    % DT domain channel vectors
for l_idx = 0:L_zp 
    for m = 1:M_prime
        % Calculate absolute time indices 'q' across the N blocks
        q_indices = (0:N-1) * M_prime + (m - 1); 
        nu_dt{m, l_idx + 1} = gs(l_idx + 1, q_indices + 1).'; 
    end
end

%% Step 6: MRC Algorithm 3 Implementation (Corrected)
d_m = zeros(M, N);                  % denominator of c_m
for m = 1:M
    d_temp = zeros(N, 1);
    for l_idx = 0:L_zp
        nu = nu_dt{m + l_idx, l_idx + 1}; 
        d_temp = d_temp + abs(nu).^2;
    end
    % Store as 1 x N row in the d_m matrix, adding noise variance
    d_m(m, :) = (d_temp + noise_var).'; 
end

iter_max = 10;                     % No. of iterations
X_dt_est = zeros(M, N);            % Estimated DT symbols 
y_residual = Y_dt_zp;              % Residual signal 

for iter = 1:iter_max
    for m = 1:M
        % Retrieve previous symbol estimate \tilde{x}_m^{(i-1)} and \tilde{d}_m
        x_prev = X_dt_est(m, :).';   % N x 1 vector
        d_m_vec = d_m(m, :).';       % N x 1 vector
        
        g_m = zeros(N, 1);           % MRC step
        for l_idx = 0:L_zp
            nu = nu_dt{m + l_idx, l_idx + 1}; 
            y_ml = y_residual(m + l_idx, :).'; 
            
            g_m = g_m + conj(nu) .* y_ml;
        end
        
        c_m = x_prev + (g_m ./ d_m_vec);  % soft estimate
        
        x_m_new = c_m; 
        
        % Calculate the difference (\Delta) for interference cancellation
        delta_x = x_m_new - x_prev;
        
        % Update the stored estimate matrix
        X_dt_est(m, :) = x_m_new.';
        
        % iv) Update \tilde{y}_{m+l} (Cancel interference of the updated symbol)
        for l_idx = 0:L_zp
            nu = nu_dt{m + l_idx, l_idx + 1};
            
            % Subtract the delta from the residuals
            y_residual(m + l_idx, :) = y_residual(m + l_idx, :) - (nu .* delta_x).';
        end
    end
end

%% Step 7: Demodulation & BER Calculation
% Convert estimated DT symbols back to DD domain via N-point FFT
X_dd_est = fft(X_dt_est, N, 2) / sqrt(N);

% Detect symbols and calculate BER
symbols_est = X_dd_est(:);
bits_est = qamdemod(symbols_est, Q, 'OutputType', 'bit', 'UnitAveragePower', true);

% Calculate BER
[num_errors, ber] = biterr(bits, bits_est);
fprintf('SNR: %d dB | BER: %e | Errors: %d\n', SNR_dB, ber, num_errors);
