Experiment(description='For debugging all',
           data_dir='../data/debug/',
           max_depth=4, 
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
           results_dir='../results/debug-all/',
           iters=250,
           base_kernels='SE,Per,Lin,Const,Noise',
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
           # search_operators=[('A', ('+', 'A', 'B'), {'A': 'kernel', 'B': 'base'}),
           #                   ('A', ('*', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}), # Might be generalised via excluded types?
           #                   ('A', ('*-const', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
           #                   ('A', 'B', {'A': 'kernel', 'B': 'base'}),
           #                   ('A', ('CP', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
           #                   ('A', ('CW', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
           #                   ('A', ('B', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
           #                   ('A', ('BL', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
           #                   ('A', ('None',), {'A': 'kernel'})])
           search_operators=[('A', ('+', 'A', 'B'), {'A': 'kernel', 'B': 'base'}),
                             ('A', ('*', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
                             ('A', ('*-const', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
                             ('A', 'B', {'A': 'kernel', 'B': 'base'}),
                             ('A', ('CP', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                             ('A', ('CW', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                             ('A', ('B', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                             ('A', ('BL', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                             ('A', ('None',), {'A': 'kernel'})])