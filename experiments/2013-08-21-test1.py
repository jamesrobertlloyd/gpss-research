Experiment(description='Seeing if CP, step and burst ever used',
           data_dir='../data/temp/',
           max_depth=6, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=4,
           sd=4, 
           max_jobs=400, 
           verbose=False,
           make_predictions=False,
           skip_complete=True,
           results_dir='../results/2013-08-21-test1/',
           iters=500,
           base_kernels='SE,Const,Lin,Cos,Step,BurstSE,IBMLin,Per,PP3,MT5',
           zero_mean=True,
           random_seed=0)
           

           
