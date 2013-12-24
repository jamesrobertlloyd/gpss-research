Experiment(description='For debugging changewindows',
           data_dir='../data/debug/',
           max_depth=3, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=True, 
           n_rand=1,
           sd=2, 
           jitter_sd=0.1,
           max_jobs=300, 
           verbose=False,
           make_predictions=True,
           skip_complete=False,
           results_dir='../results/debug-changewindow/',
           iters=250,
           base_kernels='SE',
           random_seed=1,
           period_heuristic=5,
           period_heuristic_type='min',
           subset=True,
           subset_size=250,
           full_iters=10,
           bundle_size=5,
           additive_form=False,
           mean='ff.MeanZero()',      # Starting mean
           kernel='ff.NoiseKernel()', # Starting kernel
           lik='ff.LikGauss(sf=-np.Inf)', # Starting likelihood 
           score='pl2',
           search_operators=[('A', ('+', 'A', 'B'), {'A': 'kernel', 'B': 'base'}),\
                             ('A', 'B', {'A': 'kernel', 'B': 'base'}),\
                             ('A', ('None',), {'A': 'kernel'}),\
                             ('A', ('CW', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'})])
