Experiment(description='Trying to recreate old results using latest code',
           data_dir='../data/internet/',
           max_depth=4, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=9,
           sd=2, 
           jitter_sd=0.1,
           max_jobs=200, 
           verbose=False,
           make_predictions=False,
           skip_complete=False,
           results_dir='../results/2014-01-09-internet/',
           iters=250,
           base_kernels='SE,Per,Lin,Const,Noise',
           random_seed=1,
           period_heuristic=3,
           period_heuristic_type='min',
           max_period_heuristic=4,
           subset=True,
           subset_size=250,
           full_iters=10,
           bundle_size=5,
           additive_form=True,
           mean='ff.MeanZero()',      # Starting mean
           kernel='ff.SumKernel(operands=[ff.SqExpKernel(dimension=0, lengthscale=-3.39883272338, sf=8.33060225149), ff.PeriodicKernel(dimension=0, lengthscale=0.842997779897, period=-3.97317017218, sf=8.65711337467), ff.ProductKernel(operands=[ff.PeriodicKernel(dimension=0, lengthscale=-0.24765463152, period=-5.90022125708, sf=0.293632002427), ff.PeriodicKernel(dimension=0, lengthscale=0.842997779897, period=-3.97317017218, sf=7.83175865214)]), ff.ChangeWindowKernel(dimension=0, location=2004.9961733, steepness=4.35732876322, width=-3.92962236011, operands=[ ff.SumKernel(operands=[ff.NoiseKernel(sf=7.39362696689), ff.ConstKernel(sf=10.3189945124), ff.SqExpKernel(dimension=0, lengthscale=-6.34290946346, sf=8.85861605252), ff.PeriodicKernel(dimension=0, lengthscale=-0.24765463152, period=-5.90022125708, sf=9.78727179228), ff.ProductKernel(operands=[ff.SqExpKernel(dimension=0, lengthscale=-6.34290946346, sf=8.03326132999), ff.PeriodicKernel(dimension=0, lengthscale=-0.24765463152, period=-5.90022125708, sf=0.293632002427)])]), ff.SumKernel(operands=[ff.SqExpKernel(dimension=0, lengthscale=-3.31006226322, sf=9.648401374), ff.ProductKernel(operands=[ff.SqExpKernel(dimension=0, lengthscale=-3.31006226322, sf=8.82304665147), ff.PeriodicKernel(dimension=0, lengthscale=-0.24765463152, period=-5.90022125708, sf=0.293632002427)])]) ])])', # Starting kernel
           lik='ff.LikGauss(sf=-np.Inf)', # Starting likelihood 
           score='bic',
           search_operators=[('A', 'B', {'A': 'kernel', 'B': 'base'}),
                             ('A', ('None',), {'A': 'kernel'})])



