% This script is based on the following paper:

% Qi, Chenhao, et al. "Hierarchical Codebook-Based Multiuser Beam Training
% for Millimeter Wave Massive MIMO." IEEE Transactions on Wireless
% Communications 19.12 (2020): 8142-8152.

% The script closely follows Section III of the paper and implements the
% alternative minimization method with a closed-form expression (AMCF)

% Author: Jianhua Mo
% Created on 2/28/2021

close all;
clearvars;
N = 64;

% Q should be larger than N, based on Prof. Qi's paper. I set it as 1024
% sampled directions to evaluate the beam pattern
Q = 2^10;
q = 1:1:Q;
Omega_q = -1 + (2*q-1)./Q;

% array responses in a matrix
A_phase = (0:1:N-1)'* Omega_q;
A = exp(1j*pi*A_phase);

num_beams = 2;
for k = 1:1:num_beams
    
    % omega (spatial angle) is between -1 and 1.
    min_omega = -1 + (k-1)* 2/num_beams;
    max_omega = min_omega + 2/num_beams;
    
    % physical angles between -90 to 90
    min_theta = asind(min_omega);
    max_theta = asind(max_omega);
    
    % beamwidth in spatial angle
    B = max_omega - min_omega;
    
    mainlobe_index = find(Omega_q>=min_omega & Omega_q <= max_omega);
    sidelobe_index = find(Omega_q < min_omega | Omega_q > max_omega);
    
    % ideal beam amplitude pattern
    g = zeros(Q, 1);
    g(mainlobe_index) = sqrt(2/B);
    
    % g_eps = g in the mainlobe and = eps in the sidelobe. It is defined
    % for pattern plotting in dB scale since log 0 is NaN.
    g_eps = g;
    g_eps(sidelobe_index) = eps;
    
    figure, plot(Omega_q, g);
    xlabel('sin \theta')
    ylabel('Ideal amplitude pattern')
    
    if mod(N, 2) == 0
        % N is even
        % Qi's paper, it is not symmetric with respect to 0
        v0_phase = B*(0:1:N-1).^2/2/N + (0:1:N-1)*min_omega;
    else
        % N is odd
        v0_phase = B*(0:1:N-1).*(1:1:N)/2/N + (0:1:N-1)*(min_omega);
    end
    
    % JMo: I revise it to make it symmetric with the center direction
    % Basically, calculate the frequency at n=0 and n=N-1, and they should
    % be symmetric.
    v0_phase = B*(0:1:N-1).*(1:1:N)/2/N + (0:1:N-1)*(min_omega);
    
    
    v0 = 1/sqrt(N)*exp(1j*pi*v0_phase');
    v = v0;
    
    %% plot the intial codeword
    figure,
    subplot(2,1,1), plot(Omega_q, abs(A'*v0) )
    hold on; plot(Omega_q, g);
    grid on;
    xlabel('sin \theta')
    ylabel('Amplitude pattern')
    title('Initial chirp beam')
    
    subplot(2,1,2), plot(Omega_q, 20*log10(abs(A'*v0)))
    hold on; plot(Omega_q, 20*log10(g_eps));
    grid on;
    xlabel('sin \theta')
    ylabel('Power pattern (dB)')
    title('Initial chirp beam')
    ylim([-20, +inf])
    
    figure(11)
    subplot(1,2,1)
    plot(asind(Omega_q), 20*log10(abs(A'*v0)))
    hold on; plot(asind(Omega_q), 20*log10(g_eps));
    grid on;
    xlabel('\theta')
    ylabel('Power pattern (dB)')
    title('Initial chirp beam')
    legend('Chirp', 'Ideal','location', 'best')
    ylim([-10, +inf])
    
    %% loop
    ite = 1;
    while 1
        MSE(ite) = mean((abs(A'*v) - g).^2); % Eq. (11a)
        if ite >=10 && abs(MSE(ite)-MSE(ite-1)) < 0.01*MSE(ite)
            break;
        else
            ite = ite + 1;
        end
        Theta = angle(A'*v);
        r = g .* exp(1j*Theta);
        v = 1/sqrt(N)*exp(1j* angle(A*r));
    end
    V(:,k) = v;
    
    %% plot results
    figure, plot(MSE)
    grid on;
    xlabel('Iteration')
    ylabel('MSE')
    
    figure,
    subplot(2,1,1), plot(Omega_q, abs(A'*v) )
    hold on; plot(Omega_q, g);
    grid on;
    xlabel('sin \theta')
    ylabel('Amplitude pattern')
    title('Final beam after convergence')
    
    subplot(2,1,2), plot(Omega_q, 20*log10(abs(A'*v)))
    hold on; plot(Omega_q, 20*log10(g_eps));
    grid on;
    xlabel('sin \theta')
    ylabel('Power pattern (dB)')
    title('Final beam after convergence')
    ylim([-20, +inf])
    
    figure(11)
    subplot(1,2,2)
    plot(asind(Omega_q), 20*log10(abs(A'*v)))
    hold on; plot(asind(Omega_q), 20*log10(g_eps));
    grid on;
    xlabel('\theta')
    ylabel('Power pattern (dB)')
    legend('AMCF', 'Ideal', 'location', 'best')
    title('Final beam')
    ylim([0, +inf])
end

figure(101)
hold on;
for k = 1:1:num_beams
    plot(asind(Omega_q), 20*log10(abs(A'*V(:,k))))
end
grid on;
xlabel('\theta')
ylabel('Power pattern (dB)')
box on;
ylim([0, +inf])
hold off;

for k = 1:1:num_beams
    polarplot(asin(Omega_q), max(0, 10*log10((abs(A'*V(:,k))).^2)))
    rlim([0 +inf])
    thetalim([-90 90])
    hold on;
end
title(['WB pattern (dB), K=', num2str(num_beams)])
hold off;

for k = 1:1:num_beams
    figure()
    polarplot(asin(Omega_q), max(0, 10*log10((abs(A'*V(:,k))).^2)))
    rlim([0 +inf])
    thetalim([-90 90])
    title(['WB pattern (dB), K=', num2str(num_beams),' k=', num2str(k)])
end


% for k = 1:1:num_beams
%     polarplot(asin(Omega_q), max(0, (abs(A'*V(:,k))).^2)/N)
%     rlim([0 +inf])
%     thetalim([-90 90])
%     hold on;
% end
% title(['WB pattern, K=', num2str(num_beams)])

% save(sprintf('%d_beam_AMCF_codebook.mat',num_beams),'V')
