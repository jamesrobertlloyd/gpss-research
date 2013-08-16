Experiment(description='Trying out the integrated Brownian motion',
           data_dir='../data/tsdl/',
           max_depth=8, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=9,
           sd=4, 
           max_jobs=600, 
           verbose=False,
           make_predictions=False,
           skip_complete=True,
           results_dir='../results/2013-08-16-IBM/',
           iters=500,
           base_kernels='SE,Lin,IBMLin,Const,Per',
           zero_mean=True,
           random_seed=0,
           period_heuristic=5)
           

           
