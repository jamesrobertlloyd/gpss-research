function simple_gp_sample

	% Choose a set of x locations.
	N = 100;
	x = linspace( -2, 2, N);   

	% Specify covariance between all f(x)
	for j = 1:N
		for k = 1:N
			sigma(j,k) = covariance( x(j), x(k) );
		end
	end

    % Specify that prior mean of f is zero.
    mu = zeros(N, 1);

	% Sample from a multivariate Gaussian.
	f = mvnrnd( mu, sigma );

	plot(x, f); 
end

% Squared-exp covariance function.
function k = covariance(x, y)
    k = exp( -0.5*( x - y )^2 );
end
