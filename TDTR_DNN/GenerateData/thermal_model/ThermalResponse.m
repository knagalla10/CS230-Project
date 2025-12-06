function r = ThermalResponse(sys_props, tp)

% Hankel components
h = linspace(1, 4/sqrt(sys_props(12, 1)^2 + sys_props(13, 1)^2), 250);

% Fourier components
F_pump = 50e6; % pump frequency (MHz)
M = ceil(1/(F_pump*sys_props(15, 1)));
f = (-M:M)*F_pump;

% pump beam temporal profile
pulse = exp(-pi^2/(4*log(2)) * (sys_props(15, 1)*f).^2);
 
% pump/probe beam spatial profiles
Pr_pump = exp(-(sys_props(12, 1).*h).^2 / 8);
Pr_probe = exp(-(sys_props(13, 1).*h).^2 / 8);

% temperature profile (T)
T = zeros(length(h(:)), length(f(:)));

parfor i = 1:length(h(:))
    [Mt, qt] = MatrixQuadrupole(sys_props(2, 1), sys_props(3, 1), sys_props(4, 1), sys_props(5, 1), h(i), f, 1);
    [Ml1, ~] = MatrixQuadrupole(sys_props(6, 1), sys_props(7, 1), sys_props(8, 1), sys_props(9, 1), h(i), f, 1);
    [~, qs] = MatrixQuadrupole(sys_props(10, 1), sys_props(11, 1), 0, 0, h(i), f, 0);
    M0 = [ones(1,1,length(qt(1,1,:))), zeros(1,1,length(qt(1,1,:))); ...
        sys_props(3, 1)*sys_props(1, 1).*qt(1,1,:).^2, ones(1,1,length(qt(1,1,:)))];
    
    M = pagemtimes(M0(1:2,1:2,:), pagemtimes(Mt(1:2,1:2,:), Ml1(1:2,1:2,:)));
    G = (M(1,1,:) + sys_props(11, 1).*qs(1,1,:).*M(1,2,:)) ./ (M(2,1,:) + sys_props(11, 1).*qs(1,1,:).*M(2,2,:));
    
    T(i, :) = (Pr_pump(i)*Pr_probe(i)*h(i)) * (pulse.*G(:)');
end

% normalized response (r)
r_real = sum(real((ones(length(tp(:)), 1) * mean(T)) .* exp(-2i*pi*(tp * f))), 2);
r = (r_real - min(r_real(:))) / (max(r_real(:)) - min(r_real(:)));

end