clc
clear all
close all
load noise.mat   
fig=0;                      % Figure ID
[sig_ori,FS]=audioread('test_audio.wav');
sig_ori = sig_ori';
Lsig = length(sig_ori);     % detect the length of the signal

dt=1/FS;    % the dt is the samping interval, which is, for dt second, there will be one sample of the speech
t=0:dt:Lsig/FS;
t=t(1:Lsig);  % how many seconds

dB_list = [-10,10,30]; % set dB needed in the experiment
Is_add_special_noise = 0;  % if 0, add random noise

% this code is for ploting the FFT of the signal ------------------------
% we can see that for a speech, The energy is mainly concentrated in low frequency
f=linspace(0,FS/2,Lsig/2);
f=f';
Y=fft(sig_ori,Lsig);
inputy=Y(1:Lsig/2);
yabs=abs(Y); 
yy=yabs(1:Lsig/2);

fig=fig+1;
figure(fig)
plot(f,yy);
title('FFT')

% this code place the microphones and source
M=4;        % 2 microphones
c=340;     % The speed at which sound travels through the air is 340m/s

% We now place two microphones in four different places
Loc(1,:)=[0, 20, 0, 20]; 
Loc_M_x=Loc(1,:);
Loc(2,:)=[0, 20, 10, 10]; 
Loc_M_y=Loc(2,:);

% source located in (1, 1)m
xs=1;
ys=1;

% distance between microphones and source
Rsm=[];
 for q=1:M
     rsm=sqrt((xs-Loc_M_x(q))^2+(ys-Loc_M_y(q))^2);
     Rsm=[Rsm rsm];  % distance between microphones and source
 end

TD=Rsm/c; 
L_TD=TD/dt;
L_TD=fix(L_TD);
signal_power = sig_ori*sig_ori'/Lsig;       % calculate signal power

for idx = 1:length(dB_list)
    SNR_dB = dB_list(idx);  % 将当前元素赋值给变量 dB
    Signal_Received=[];
    noise_power = signal_power/(10^(SNR_dB/10));   % noise

    for p=1:M    
        % adding noise
        noise = sqrt(noise_power)*randn(1, Lsig);    % assume noize is zero, or says, no noise
        sig_noise =  sig_ori + noise;
        Signal_with_noise=[sqrt(noise_power)*randn(1, L_TD(p)), sig_noise, sqrt(noise_power)*randn(1, max(L_TD)-L_TD(p))]; % add the time delay with noise 
        Signal_Received=[Signal_Received; Signal_with_noise];
    end
    
    % Signal_Re_Sum = sum(Signal_Received,1);       % directly sum the signal from the M microphones
    Signal_Re_1 = Signal_Received(1,:);                 % the signal received in the first microphone
    Signal_Received_size = size(Signal_Received);
    plot_time=0:dt:(Signal_Received_size(2)-1)/FS;

    if idx == 3
    % plot and output the signal received in microphone 1
        fig=fig+1;
        figure(fig)
        plot(plot_time,Signal_Re_1)
        title('Signal-First')
        Signal_Re_1=Signal_Re_1./max(Signal_Re_1);
        audiowrite('Signal-First.wav',Signal_Re_1,FS);
        
        % % plot and output the Sum of the signal received from all microphones directly
        % fig=fig+1;
        % figure(fig)
        % plot(plot_time,Signal_Re_Sum)
        % title('Signal-Direct-Sum')
        % Signal_Re_Sum=Signal_Re_Sum./max(Signal_Re_Sum);
        % audiowrite('Signal-Direct-Sum.wav',Signal_Re_Sum,FS);
    end

    fig=fig+1;
    figure(fig)
    plot(plot_time, Signal_Received','DisplayName','Signal_Received')
    title('All Signal')
    
    % now we use xcorr, the cross-correlation, to detect the difference between
    % the microphones, and thus can add the signal correctly. 
    x1 = Signal_Received(1,:);  % microphone 1
    x2 = Signal_Received(2,:);  % microphone 2
    x3 = Signal_Received(3,:);  % microphone 3
    x4 = Signal_Received(4,:);  % microphone 4
    Max_lag = 8000; 	               % we assume that the maximum distance between any two microphones is less than 170m, which is 0.5s, which is 8000 samples
    % note that in this case, the 0 lag, which is, the lag corresponding to 0
    % time difference, is Max_lag+1 = 8001
    R_12 = xcorr(x1, x2, Max_lag, 'coeff'); 
    R_13 = xcorr(x1, x3, Max_lag, 'coeff'); 
    R_14 = xcorr(x1, x4, Max_lag, 'coeff'); 

    % plot the Cross-Correlation
    lag_list = -Max_lag:Max_lag;
    fig=fig+1;
    figure(fig)
    subplot(3,1,1)
    plot(lag_list, R_12) 
    title('the Cross-Correlation between microphone 12 in the dB of',num2str(SNR_dB))
    [Lag_12_value, Lag_12_index] = max(R_12); 
    Lag_12_estimate = Lag_12_index-(Max_lag+1);     % the lag between microphone 1 and 2

    subplot(3,1,2)
    plot(lag_list, R_13) 
    title('the Cross-Correlation between microphone 13 in the dB ofin the dB of',num2str(SNR_dB))
    [Lag_13_value, Lag_13_index] = max(R_13); 
    Lag_13_estimate = Lag_13_index-(Max_lag+1);     % the lag between microphone 1 and 3

    subplot(3,1,3)
    plot(lag_list, R_14) 
    title('the Cross-Correlation between microphone 14 in the dB ofin the dB of',num2str(SNR_dB))
    [Lag_14_value, Lag_14_index] = max(R_14); 
    Lag_14_estimate = Lag_14_index-(Max_lag+1);     % the lag between microphone 1 and 4

    % just to see the real lag, cannot use Real_lag = L_TD(1)-L_TD(2) in your
    % code when add the signal from different microphones
    % visualize the real lag of between microphone and the others
    error=[];
    Real_lag=[];

    Real_lag(1) = L_TD(1)-L_TD(2);    
    Real_lag(2) = L_TD(1)-L_TD(3);
    Real_lag(3) = L_TD(1)-L_TD(4);
    error(1) = Lag_12_estimate - Real_lag(1);
    error(2) = Lag_13_estimate - Real_lag(2);
    error(3) = Lag_14_estimate - Real_lag(3);
    for i = 1:3 
        fprintf("Error of lags' number between 1 %d with the SNR of %d is %d\n", (i+1), SNR_dB, error(i))
    end
    
    x2_pad = x2(round(-Lag_12_estimate):end);
    x3_pad = x3(round(-Lag_13_estimate):end);
    x4_pad = x4(round(-Lag_14_estimate):end);

    % Align the FOUR signals with correct lag
    x1_with_lag = x1;
    x2_with_lag = [x2_pad, zeros(1, round(-Lag_12_estimate)-1)];
    x3_with_lag = [x3_pad, zeros(1, round(-Lag_13_estimate)-1)];
    x4_with_lag = [x4_pad, zeros(1, round(-Lag_14_estimate)-1)];

    Correct_Sum_with_lag =x1_with_lag + x2_with_lag + x3_with_lag +x4_with_lag;
    plot_time2=0:dt:(length(Correct_Sum_with_lag)-1)/FS;

    fig=fig+1;
    figure(fig)
    plot(plot_time2,Correct_Sum_with_lag)
    title('Signal-Correct-Sum-1234 with dB of',num2str(SNR_dB))
    Correct_Sum_with_lag=Correct_Sum_with_lag./max(Correct_Sum_with_lag);

    if idx ==1
        audiowrite('Signal-Correct-Sum-1234-dB-10.wav',Correct_Sum_with_lag,FS);
    elseif idx ==2
        audiowrite('Signal-Correct-Sum-1234-dB10.wav',Correct_Sum_with_lag,FS);
    else
        audiowrite('Signal-Correct-Sum-1234-dB30.wav',Correct_Sum_with_lag,FS);
    end
end

