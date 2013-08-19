Experiment(description='More kernels and operators',
           data_dir='../data/tsdl/',
           max_depth=8, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=9,
           sd=4, 
           max_jobs=400, 
           verbose=False,
           make_predictions=False,
           skip_complete=True,
           results_dir='../results/2013-08-19-time-series/',
           iters=500,
           base_kernels='IBM,IBMLin,Step,BurstSE,Per,Cos,Lin,PP2,PP3,MT3,MT5,SE,Const',
           zero_mean=True,
           random_seed=0,
           period_heuristic=5)
           

           
