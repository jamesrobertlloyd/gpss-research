Experiment(description='Testing the subsetting',
           data_dir='../data/temp/',
           max_depth=6, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=4,
           sd=4, 
           max_jobs=600, 
           verbose=False,
           make_predictions=False,
           skip_complete=True,
           results_dir='../results/2013-09-02-test1/',
           iters=100,
           base_kernels='SE,Const,Lin,Cos,Step,BurstSE,IBMLin,Per,PP3,MT5',
           zero_mean=True,
           random_seed=0,
           subset=True,
           subset_size=50)
           

           
