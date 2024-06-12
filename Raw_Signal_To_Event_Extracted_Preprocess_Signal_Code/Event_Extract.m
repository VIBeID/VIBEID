function [ Event_loc, Features] = Event_Extract( Evnt_Prdctd, signal, sigma, prsn )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

fs=8000; % frequency of the signal
tm = (0:length(signal)-1)/fs; 

new_sig = zeros(size(signal,1),1);
Event_loc = zeros(size(signal,1),1);

Detctd_Evnt_idx = find(Evnt_Prdctd(:,3) == 1);
j=1;
iter = 1;

footfall_len = 1500;  % footstep length 
Sig = zeros(1,footfall_len);
k= 1:footfall_len;

while j < length(Detctd_Evnt_idx)
    
    fprintf('\n================= Person %d and Event %d =================\n',prsn, iter )

    stp = 0;
    while (Detctd_Evnt_idx(j+1) == Detctd_Evnt_idx(j)+1)
        j = j+ 1;
        stp = stp+1;
        if j == length(Detctd_Evnt_idx)
            break;
        end
    end
    
    Evnt_Start = Evnt_Prdctd(Detctd_Evnt_idx(j-stp),1);
    Evnt_Stop =  Evnt_Prdctd(Detctd_Evnt_idx(j),2);  
    
    [~ , mn] = max(abs(signal(Evnt_Start:Evnt_Stop)));
    mn = mn + Evnt_Start;

    wndw = tukeywin(footfall_len,0.5); % turkey window

    strt = mn - 400;
    stop = strt + footfall_len-1;
    if strt < 1
        strt = 1;
        wndw = wndw(1:stop-strt+1);
    end
    
    if stop >= length(signal)
        stop = length(signal);
        wndw = wndw(1:stop-strt+1);
    end
    
    w_diag = diag(wndw);
    sig = (signal(strt:stop));
    Sig(:,1:length(sig)) = w_diag*sig;

     evnt_len(iter,:) = sum(Sig ~=0);  

    Features(iter,:) = Sig; % zero padding in case the signals is smaller
    Event_loc(mn,:) = 0.5;

    %% --- For the visualization, Uncomment this section ------------------------------
%     
%    ovrlp_sig = [zeros(strt-1,1); signal(strt:stop) ; zeros((length(signal)-stop),1)];            
%     
%     if(strt<80000)
%            min_lim = strt;
%            max_lim = stop+80000;
%            
%            if max_lim > length(signal)
%                max_lim = length(signal);
%            end
%            N = min_lim:max_lim;
%            idx = linspace(min_lim,max_lim,length(N));
%            
%        else
%            min_lim = strt-40000;
%            max_lim = stop +40000;
%            
%            if max_lim > length(signal)
%                max_lim = length(signal);
%            end
%            N = min_lim:max_lim;
%            
%            idx = linspace(min_lim,max_lim,length(N));
%            
%        end
%        
%        %
%        
%        subplot(211)
%        plot(tm(idx),signal(idx),'k')
%        hold on
%        plot(tm(idx),ovrlp_sig(idx),'r')
%        grid on
%        xlim([tm(min_lim) tm(max_lim)])
%        xlabel('Samples')
%        ylabel('Amplitude')
%        title('Complete Original Signal')
%        legend('Original Signal', 'Windowed Signal')
%        subplot(223)
%        plot(signal(strt:stop))
%        hold on
%        plot(0.4*wndw)
%        grid on
%        xlabel('Samples')
%        ylabel('Amplitude')
%        title('Chunk of the Original Signal')
%        subplot(224)
%        plot(Sig)
%        grid on
%        xlabel('Samples')
%        ylabel('Amplitude')
%        title('Windowed Signal convoluted with Gaussian Window')
%        pause(0.081)
%        clf
%% ---------------------------------------------------------------------------------------       
%        tm = (0:length(wndw)-1)/fs;
%        N2 = 2^nextpow2(length(Sig));
%        Sig_fft = fft(Sig,N2);
%        freq  = fs/2*(linspace(0,1,N2/2+1));
%        
%        subplot(121)
%        plot(tm,Sig)
%        grid MINOR
%        subplot(122)
%        plot(freq,abs(Sig_fft(1:N2/2+1))/max(abs(Sig_fft(1:N2/2+1))),'black');
%        xlim([0 260])
%        grid MINOR
%     subplot(211)
%     plot(signal)
%     subplot(223)
%     plot(sig)
%     grid on
%     hold on
%     plot(wndw)
%     grid on
%     hold off
%     subplot(224)
%     plot(Sig)
%     grid on
%     pause(0.1)
%     clf


j = j + 1
iter = iter + 1

end

end

