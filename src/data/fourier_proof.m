%%
band_buffer=[];
PSD_buffer=[];
PSD_buffer2=[];
band_powers2=[];
%%
eegdata=sine_data11.data;
Y = fft(eegdata);
%Y=reshape(Y,[],250);
L=length(Y);

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
Fs=200;
f = Fs*(0:(L/2))/L;
% plot(f,P1) 
% title('Single-Sided Amplitude Spectrum of S(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
BUFFER_LENGTH = 5;
EPOCH_LENGTH = 1;
OVERLAP_LENGTH = 0.85;
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH;

w=hamming(L);
dataWinCentered =  eegdata- mean(eegdata);
dataWinCenteredHam=dataWinCentered.*w;
NFFT=256;
Y=fft(dataWinCenteredHam,NFFT)/L;
PSD=2*abs(Y(1:(NFFT/2)));
f=Fs/2*linspace(0,1,NFFT/2);
ind_delta=f<4;
ind_theta=(f>=4 & f<=8); 
ind_alpha=(f>=8 & f<=12);
ind_beta=(f>=12 & f<=30);
ind_all=f<=30;
meanDelta=mean(PSD(ind_delta));
meanTheta=mean(PSD(ind_theta));
meanAlpha=mean(PSD(ind_alpha));
meanBeta=mean(PSD(ind_beta));
sumDelta=sum(PSD(ind_delta)*(f(2)-f(1)));
sumTheta=sum(PSD(ind_theta)*(f(2)-f(1)));
sumAlpha=sum(PSD(ind_alpha)*(f(2)-f(1)));
sumBeta=sum(PSD(ind_beta)*(f(2)-f(1)));
band_powers=[meanDelta,meanTheta,meanAlpha,meanBeta];
band_powers2=[sumDelta,sumTheta,sumAlpha,sumBeta];
PSD_power=mean(PSD);
band_buffer=[band_buffer;band_powers];
PSD_buffer=[PSD_buffer;PSD_power];
PSD_buffer2=[PSD_buffer2;sum(PSD(ind_all)*(f(2)-f(1)))];
%%
w2=hamming(length(band_powers2));
dataWinCentered2 =  band_powers2- mean(band_powers2);
dataWinCenteredHam2=dataWinCentered2.*w2;
NFFT2=16;
Fs2=1;
Y2=fft(dataWinCenteredHam2,NFFT2)/length(band_powers2);
PSD2=2*abs(Y2(1:(NFFT2/2),:));
f2=Fs2/2*linspace(0,1,NFFT2/2);
