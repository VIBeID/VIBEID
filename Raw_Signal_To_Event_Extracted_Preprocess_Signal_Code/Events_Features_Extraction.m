function [ feat ] = Events_Features_Extraction( Fs,sig )
% Function for event and feature extraction

i=1;


feat(i) = std(sig); i = i+1;
feat(i) = kurtosis(sig); i= i+1; % Kurtosis on signal

L=length(sig);
NFFT = 8*2^nextpow2(L); % Next power of 2 from length of y
fft_sig = fft(sig,NFFT)/L; 
fft_sig=2*abs(fft_sig(1:NFFT/2+1)); % fft of signal
f = Fs/2*linspace(0,1,NFFT/2+1);

step = 40 ;

 for j = 40:step:120
 
      feat(i)=(norm( fft_sig(  sum(f <= j)+1 : sum(f <= j+step) )  )^2);i=i+1;
 
 end

end

