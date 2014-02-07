function simple_gp_sample

	% Choose a set of x locations.
	N = 100;
	x = linspace( -2, 2, N);   

	% Specify the covariance between function
    % values, depending on their location.
	for j = 1:N
		for k = 1:N
		    sigma(j,k) = covariance( x(j), x(k) );
		end
	end

    % Specify that the prior mean of f is zero.
    mu = zeros(N, 1);

	% Sample from a multivariate Gaussian.
	f = mvnrnd( mu, sigma );

	plot(x, f); 
end

% Periodic covariance function.
function c = covariance(x, y)
    c = exp( -0.5*( sin(( x - y )*1.5).^2 ));
end
