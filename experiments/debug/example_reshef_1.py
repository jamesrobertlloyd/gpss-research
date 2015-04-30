dict(description='For demonstrating new functionality',
     data_dir='../data/debug/',
     n_rand=4,
     skip_complete=False,
     results_dir='../results/debug/reshef_1',
     iter_and_subset_schedule=[(50, 100), (25, 250), (10, 500), (5, inf)],
     base_kernels='SE,Noise',
     additive_form=True,
     mean='ff.MeanZero()',      # Starting mean
     kernel='ff.NoiseKernel()', # Starting kernel
     lik='ff.LikGauss(sf=-np.Inf)', # Starting likelihood
     verbose=False,
     search_operators=[('A', ('+', 'A', 'B'), {'A': 'kernel', 'B': 'base'}),
                       ('A', ('*', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
                       ('A', 'B', {'A': 'kernel', 'B': 'base'}),
                       ('A', ('None',), {'A': 'kernel'})])
