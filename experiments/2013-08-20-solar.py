Experiment(description='Trying to get solar to work',
           data_dir='../data/solar/',
           max_depth=8, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=9,
           sd=4, 
           max_jobs=500, 
           verbose=False,
           make_predictions=False,
           skip_complete=True,
           results_dir='../results/2013-08-20-solar/',
           iters=500,
           base_kernels='Step,BurstSE,Per,Cos,Lin,SE,Const',
           zero_mean=True,
           random_seed=0,
           period_heuristic=5)
           

           
