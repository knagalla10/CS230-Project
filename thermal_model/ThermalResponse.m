function r = ThermalResponse(sys_props, tp)

% pump frequency (MHz)
F_pump = 50e6;

% Hankel components
h_index = 500;
[h, weight] = lgwt(h_index, 0, 2/sqrt(sys_props(12, 1)^2 + sys_props(13, 1)^2));

% Fourier components
M = ceil(5/(F_pump*sys_props(15, 1)));
f = (-M:M)*F_pump;

% Gaussian pulse
pulse = exp(-pi*(f*sys_props(15, 1)).^2) .* exp(-2i*pi*f*sys_props(14, 1));
% pulse = exp(-pi*(f*sys_props(15, 1)).^2);

% beam profiles
Pr_pump = exp(-(pi*sys_props(12, 1).*h).^2 / 2);
Pr_probe = exp(-(pi*sys_props(13, 1).*h).^2 / 2);

% temperature profile (T)
T = zeros(h_index, length(f(:)));

parfor i = 1:h_index
    [Mt, qt] = MatrixQuadrupole(sys_props(2, 1), sys_props(3, 1), sys_props(4, 1), sys_props(5, 1), h(i), f, 1);
    [Ml1, ~] = MatrixQuadrupole(sys_props(6, 1), sys_props(7, 1), sys_props(8, 1), sys_props(9, 1), h(i), f, 1);
    [~, qs] = MatrixQuadrupole(sys_props(10, 1), sys_props(11, 1), 0, 0, h(i), f, 0);
    M0 = [ones(1,1,length(qt(1,1,:))), zeros(1,1,length(qt(1,1,:))); ...
        sys_props(3, 1)*sys_props(1, 1).*qt(1,1,:).^2, ones(1,1,length(qt(1,1,:)))];
    
    M = pagemtimes(M0(1:2,1:2,:), pagemtimes(Mt(1:2,1:2,:), Ml1(1:2,1:2,:)));
    G = (M(1,1,:) + sys_props(11, 1).*qs(1,1,:).*M(1,2,:)) ./ (M(2,1,:) + sys_props(11, 1).*qs(1,1,:).*M(2,2,:));
    
    T(i, :) = (weight(i)*Pr_pump(i)*Pr_probe(i)*h(i)) * G(:)';
end

T = sum(T);

% normalized response (r)
r_real = sum(real((ones(length(tp(:)), 1) * pulse.*T) .* exp(-2i*pi*(tp * f))), 2);
r = (r_real - min(r_real(:))) / (max(r_real(:)) - min(r_real(:)));

end
