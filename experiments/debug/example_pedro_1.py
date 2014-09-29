dict(description='For debugging',
     data_dir='../data/debug/',
     n_rand=4,
     skip_complete=False,
     results_dir='../results/debug/pedro_1',
     iters=100,
     base_kernels='SE,Noise',
     additive_form=True,
     mean='ff.MeanZero()',      # Starting mean
     kernel='ff.NoiseKernel()', # Starting kernel
     lik='ff.LikGauss(sf=-np.Inf)', # Starting likelihood
     starting_subset=75,
     verbose=False,
     search_operators=[('A', ('+', 'A', 'B'), {'A': 'kernel', 'B': 'base'}),
                       ('A', ('*', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
                       ('A', 'B', {'A': 'kernel', 'B': 'base'}),
                       ('A', ('None',), {'A': 'kernel'})])
