function [M, q] = MatrixQuadrupole(Cv, k, t, Z, h, f, matrix_flag)

q(1, 1, :) = sqrt(h^2 + 2i*pi*f*Cv/k);

if matrix_flag == 1
    M(1:2, 1:2, :) = pagemtimes([cosh(q.*t), sinh(q.*t)./(k.*q); sinh(q.*t).*k.*q, cosh(q.*t)], [1, Z; 0, 1]);
else
    M = zeros(2, 2, length(q(1, 1, :)));
end

end
