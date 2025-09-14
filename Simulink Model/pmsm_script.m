%% PMSM parameters (from the paper, Table 1)
Rs   = 4.765;          % Ohm
Lq   = 0.014;          % H
Ld   = 0.014;          % H
P    = 2;              % poles
B    = 4.047e-5;       % N*m/(rad/s)
J    = 0.0001051;      % kg*m^2
flux = 0.1848;         % Wb (lambda_f)

%% Test supply (three-phase) – you can tweak these later
Vm   = 100;            % phase peak volts (pick a reasonable value)
fe   = 50;             % electrical frequency (Hz)
we   = 2*pi*fe;        % electrical ang. freq (rad/s)

%% Load torque profile
TL0      = 0;          % N*m before 0.5 s
TL_step  = 1.7;        % N*m after 0.5 s (as in the paper)
t_step   = 0.5;        % s

%% Initial conditions
id0   = 0; iq0 = 0;
wm0   = 0;             % mechanical speed (rad/s)
wr0   = (P/2)*wm0;     % electrical speed (rad/s)
theta0= 0;             % electrical angle (rad)

%% Handy gains
Kt = 3*P/4;            % torque gain = (3/2)*(P/2)
