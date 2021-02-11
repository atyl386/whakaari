import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe, save_dataframe, datetimeify
from datetime import timedelta, datetime

# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

def forecast_dec2019():
    ''' forecast model for Dec 2019 eruption
    '''
    # constants
    threshold = 0.85
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    td = TremorData()
    ti = '2011-01-01'
    tf = '2013-01-01'
    # construct model object
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(ti=ti, tf=tf, window=2., overlap=0.75,
        look_forward=2., data_streams=data_streams)
    
    # columns to manually drop from feature matrix because they are highly correlated to other 
    # linear regressors
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 8
    # train the model, excluding 2019 eruption
    # note: building the feature matrix may take several hours, but only has to be done once
    # and will intermittantly save progress in ../features/
    # trained scikit-learn models will be saved to ../models/*root*/
    fm.train(ti=ti, tf=tf, drop_features=drop_features, retrain=True, n_jobs=n_jobs)

    # run forecast from 2011 to 2020
    # model predictions will be saved to ../predictions/*root*/ 
    ys = fm.forecast(ti=ti, tf=tf, recalculate=True, n_jobs=n_jobs)
    #fm.plot_accuracy(ys, save='{:s}\\accuracy_'.format(fm.plotdir)+'.png')
    # plot forecast and quality metrics
    # plots will be saved to ../plots/*root*/
    fm.plot_forecast(ys, threshold=threshold, save='{:s}\\forecast_'.format(fm.plotdir)+'.png',xlim=['2011-01-01',tf])
    # construct a high resolution forecast (10 min updates) around the Dec 2019 eruption
    # note: building the feature matrix might take a while
    #fm.hires_forecast(ti=datetimeify('2020-09-01'), tf=datetimeify('2020-10-01'), recalculate=False, 
        #save='{:s}\\forecast_hires_'.format(fm.plotdir)+'.png', n_jobs=n_jobs, threshold=threshold, root='2020-09')
        
if __name__ == "__main__":
    forecast_dec2019()
    #forecast_test()
    #forecast_now()