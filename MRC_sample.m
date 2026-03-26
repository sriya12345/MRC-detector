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
