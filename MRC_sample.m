close all; clc; clear;
%% Step1: Grid Design + QAM Modulation
%% OTFS Parameters
M = 4;
N = 64;
del_f = 15e3;
T = 1/del_f;
T_f = N*T;

%% Channel Parameters
v = 500*(1000/3600);      % speed (m/s)
fc = 4e9;                 % carrier frequency
c = 3e8;                  % speed of light

fd = (v/c)*fc;            % max Doppler frequency

% Hardcoded delay taps
l_hard = [0,1,2,3];

% Hardcoded doppler taps
k_hard = [0,1,2,3];

%% Modulation
Q = 4;                    % 4-QAM
bits_per_sym = log2(Q);

num_sym = N*M;
bits = randi([0 1], num_sym*bits_per_sym ,1);
symbols = qammod(bi2de(reshape(bits,bits_per_sym,[]).'), ...
                 Q, 'UnitAveragePower', true);

% delay-Doppler transmit matrix
D = reshape(symbols,M,N);   % x(k,l)
d = D(:);

%% Step 2: DD to time domain + ZP insertion
X_tf = fft(ifft(D, [], 1), [], 2);      % Time freq domain (ISFFT)
x_time = ifft(X_tf, M, 1);              % time domain (Heisenberg tranform)

% ZP insertion
L_zp = max(l_hard);                     % max delay

x_zp = [x_time; zeros(L_zp, N)];
s = x_zp(:);                        

%% Step 3: Channel design
P = 4;                                        % No. of paths
h = (randn(1,P) + 1j*randn(1,P)) / sqrt(2);   % Rayleigh fading

l = l_hard;
k = k_hard;

channel.h = h;
channel.l = l;
channel.k = k;
channel.P = P;

y = 0;
for i = 1:P
    y = y + h(i) * circshift(s, [l, k]);
end

