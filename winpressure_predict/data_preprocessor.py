#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import concat
import matplotlib.pyplot as plt
from pandas import Series
from winpressure_predict import core
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")  # Ignore some annoying warnings
ANGLE='ANGLE 90_ave'

PATH, FIG_PATH, LOG_PATH= core.check_path('./')
#print all
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 10000)

# Decompose
def decom(series=None, trials=100,decom_mode='ceemdan',re_decom=False,re_imf=0,vmd_params=None, draw=False,FORECAST_LENGTH=1,SENSOR='', **kwargs):

    if series is None: raise ValueError('Please input pd.Series, or pd.DataFrame(1D), nd.array(1D).')
    try:
        series = pd.Series(series)
    except:
        raise ValueError(
            'Sorry! %s is not supported to decompose, please input pd.Series, or pd.DataFrame(1D), nd.array(1D).' % type(
                series))
    decom_mode = decom_mode.lower()
    best_params = 'None'

    if decom_mode == 'vmd':
        df_decom, imfs_hat, omega  = decom_vmd(series,SENSOR=SENSOR, **kwargs)

    elif decom_mode == 'ovmd':
        print("============================ovmd start=====================")
        df_decom, best_params = decom_ovmd(series,SENSOR=SENSOR, vmd_params=vmd_params, **kwargs)

    elif decom_mode == 'svmd':
        df_decom = decom_svmd(series, vmd_params=vmd_params, FORECAST_LENGTH=FORECAST_LENGTH, **kwargs)
    else:
        raise ValueError('%s is not a supported decomposition method!' % (decom_mode))

        if draw:
            print("==================Draw decome====================")
            series_index = range(len(series))
            fig = plt.figure(figsize=(16, 2 * imfs_num))
            plt.subplot(1 + imfs_num, 1, 1)
            plt.plot(series_index, series, color='#0070C0')  # F27F19 orange #0070C0 blue
            plt.ylabel('Original data')

            # Plot IMFs
            for i in range(imfs_num):
                plt.subplot(1 + imfs_num, 1, 2 + i)
                plt.plot(series_index, imfs_emd[i, :], color='#F27F19')
                plt.ylabel(str.upper(decom_mode) + '-VMF' + str(i))
            # Save figure
            fig.align_labels()
            plt.tight_layout()
            plt.show
            if re_decom == False:
                plt.savefig(FIG_PATH + ANGLE + ' ' + str.upper(decom_mode) + ' Result.png', bbox_inches='tight')
            else:
                plt.savefig(FIG_PATH + ANGLE + ' ' + 're_decom-IMF' + str(re_imf) + ' ' + str.upper(
                    decom_mode) + ' Re-decomposition Result.png',
                            bbox_inches='tight')
            print("==============Draw decome result sucessfully===========")

        else:
            print("==================not Draw result decome====================")
    if isinstance(series, pd.Series):
        if 'vmd' in decom_mode and len(series) % 2:
            df_decom.index = series[1:].index  # change index
        else:
            df_decom.index = series.index  # change index

    df_decom_target = df_decom    #result with target
    print("df_decom:----------------------------------\n",df_decom)
    df_decom.name = decom_mode.lower() + '_' + str(best_params)

    return df_decom,df_decom_target

# VMD
def decom_vmd(series=None, alpha=2000, tau=0, K=4, DC=0, init=1, tol=1e-7,SENSOR='',**kwargs):  # VMD Decomposition
    try:
        import vmdpy
    except ImportError:
        raise ImportError('Cannot import vmdpy, run: pip install vmdpy!')
    if series is None:
        raise ValueError('Please input pd.Series, or pd.DataFrame(1D), nd.array(1D).')
    if len(series) % 2:
        print('Warning! The vmdpy module will delete the last one data point of series before decomposition')

    #VMD
    imfs_vmd, imfs_hat, imfs_omega = vmdpy.VMD(series, alpha, tau, K, DC, init, tol, **kwargs)
    IMF_all=[(SENSOR + '-VMF' + str(i)) for i in range(K)]

    df_vmd = pd.DataFrame(imfs_vmd.T, columns=IMF_all)

    return df_vmd, imfs_hat, imfs_omega

# Optimized VMD (OVMD)
def decom_ovmd(series=None, SENSOR='',vmd_params=None, trials=100):  # VMD Decomposition

    if series is None: raise ValueError('Please input pd.Series, or pd.DataFrame(1D), nd.array(1D).')
    try:
        series = pd.Series(series)
    except:
        raise ValueError(
            'Sorry! %s is not supported to decompose, please input pd.Series, or pd.DataFrame(1D), nd.array(1D)' % type( series))
    if vmd_params is None:
        try:
            import optuna
        except:
            raise ImportError('Cannot import optuna, run: pip install optuna!')

        def objective(trial):
            K = trial.suggest_int('K', 4, 4)  # set hyperparameter range
            alpha = trial.suggest_int('alpha', 1, 10000)
            tau = trial.suggest_float('tau', 0, 1)
            df_vmd, imfs_hat, imfs_omega = decom_vmd(series,SENSOR=SENSOR, K=K, alpha=alpha, tau=tau)
            return abs((df_vmd.sum(axis=1).values - series.values).sum())  # residual of decomposed and original series

        study = optuna.create_study(study_name='OVMD Method', direction='minimize')  # TPESampler is used

        optuna.logging.set_verbosity(optuna.logging.WARNING)  # not to print
        print("test")
        study.optimize(objective, n_trials=trials, n_jobs=-1, gc_after_trial=True)  # number of iterations
        print("OVMD test")
        vmd_params = study.best_params
    df_vmd, imfs_hat, imfs_omega = decom_vmd(series,SENSOR=SENSOR, K=vmd_params['K'], alpha=vmd_params['alpha'],
                                             tau=vmd_params['tau'])
    print("(------------------------)")
    return df_vmd, vmd_params


# Separated VMD (SVMD)
def decom_svmd(series=None, FORECAST_LENGTH=None, optimize=True, vmd_params=None, trials=100):  # VMD Decomposition

    if series is None: raise ValueError('Please input pd.Series, or pd.DataFrame(1D), nd.array(1D).')
    try:
        series = pd.Series(series)
    except:
        raise ValueError(
            'Sorry! %s is not supported to decompose, please input pd.Series, or pd.DataFrame(1D), nd.array(1D)' % type(
                series))
    if len(series) % 2:
        print('Warning! The vmdpy module will delete the last one data point of series before decomposition')
        series = series[1:]
    if FORECAST_LENGTH is None: raise ValueError('Please input FORECAST_LENGTH.')
    if vmd_params is None and optimize == False: vmd_params = {'K': 5, 'tau': 0, 'alpha': 2000}

    series_train = series[:-FORECAST_LENGTH]
    series_test = series[-FORECAST_LENGTH:]
    if vmd_params is None and optimize:
        try:
            import optuna
        except:
            raise ImportError('Cannot import optuna, run: pip install optuna!')

        def objective(trial):
            K = trial.suggest_int('K', 1, 10)  # set hyperparameter range
            alpha = trial.suggest_int('alpha', 1, 10000)
            tau = trial.suggest_float('tau', 0, 1)
            df_vmd, imfs_hat, imfs_omega = decom_vmd(series_train, K=K, alpha=alpha, tau=tau)
            return abs((df_vmd.sum(axis=1).values - series_train.values).sum())  # residual of decomposed and original training(h5) set series

        study = optuna.create_study(study_name='SVMD Method for training(h5) set',
                                    direction='minimize')  # TPESampler is used
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # not to print
        study.optimize(objective, n_trials=trials, n_jobs=-1, gc_after_trial=True)  # number of iterations
        vmd_params = study.best_params
    df_vmd_train, imfs_hat, imfs_omega = decom_vmd(series_train, K=vmd_params['K'], alpha=vmd_params['alpha'],
                                                   tau=vmd_params['tau'])
    df_vmd_test, imfs_hat, imfs_omega = decom_vmd(series_test, K=vmd_params['K'], alpha=vmd_params['alpha'],
                                                  tau=vmd_params['tau'])
    df_vmd = pd.concat((df_vmd_train, df_vmd_test))
    df_vmd.index = series.index
    return df_vmd

# Evaluate R2, MSE, MAE, MAPE
def eval_result(y_real, y_pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error  # R2, MSE, MAE, MAPE
    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()
    scale = np.max(y_real) - np.min(y_real)  # scale is important for RMSE and MAE

    r2 = r2_score(y_real, y_pred)
    rmse = mean_squared_error(y_real, y_pred, squared=False)  # RMSE and MAE are various on different scales
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred)  # Note that dataset cannot have any 0 value.
    df_eval = pd.DataFrame({'Scale': scale, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}, index=[0])
    print("df_eval:\n",df_eval)
    return df_eval

# 4.Normalize
def normalize_dataset(data=None, FORECAST_LENGTH=None, NOR_METHOD='MinMax'):

    try:
        from sklearn.preprocessing import MinMaxScaler, StandardScaler ,RobustScaler
    except ImportError:
        raise ImportError('Cannot import sklearn, run: pip install scikit-learn!')
    try:
        data = pd.DataFrame(data)
    except:
        raise ValueError('Invalid input!')

    # Split
    if len(data.columns) == 1:  # Initialize Series
        dataY = data.values.reshape(-1, 1)
        dataX = dataY
    else:  # Initialize DataFrame training set and test set
        dataY = data['target'].values.reshape(-1, 1)
        dataX = data.drop('target', axis=1, inplace=False)

    # Setting normalizing method
    if NOR_METHOD is None: NOR_METHOD = ''
    if NOR_METHOD.lower() == 'minmax':
        scalarX = MinMaxScaler(feature_range=(0, 1))
        scalarY = MinMaxScaler(feature_range=(0, 1))
    elif NOR_METHOD.lower() == 'std':
        scalarX = StandardScaler()
        scalarY = StandardScaler()
    elif NOR_METHOD.lower() == 'robustScaler':
        scalarX = RobustScaler()
        scalarY = RobustScaler()
    else:
        scalarY = None
        print("Warning! Data is not normalized, please set nor_method = eg.'minmax', 'std'")

    # Normalize by sklearn
    if scalarY is not None:
        if FORECAST_LENGTH is None:
            dataX = scalarX.fit(dataX)  # Normalize X
        else:
            scalarX.fit(dataX[:-FORECAST_LENGTH])  # Avoid using training(h5) set for normalization
        dataX = scalarX.transform(dataX)
        # if fit_method is not None: fit_method = scalarX.transform(fitting_set)

        if FORECAST_LENGTH is None:
            dataY = scalarY.fit(dataY)
        else:
            scalarY.fit(dataY[:-FORECAST_LENGTH])  # Avoid using training(h5) set for normalization
        dataY = scalarY.transform(dataY)

    return np.array(dataX), np.array(dataY), scalarY

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def inverse_difference(history, yhat, interval=1):  # 历史数据，预测数据，差分间隔
    return yhat + history[-interval]

'''------------------------reshape---------------------------------'''
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def create_train_test_set(data=None, FORECAST_LENGTH=None, FORECAST_HORIZONS=None,target_lable='', val_split=0.8):
    # Normalize
    if FORECAST_LENGTH is None: raise ValueError('Please input a FORECAST_LENGTH!')
    if FORECAST_HORIZONS is None: raise ValueError('Please input a FORECAST_HORIZONS!')
    # dataX, dataY, scalarY = normalize_dataset(data, FORECAST_LENGTH, NOR_METHOD)  # data X Y is np.array here
    values = data.values
    # FOD
    # diff_values = difference(values, 1)
    # print("diff_values:-----------------------\n", diff_values)
    values = values.astype('float32')
    # normalize features
    print("values:-----------------------\n", values)
    # 转成有监督数据
    reframed = series_to_supervised(values,FORECAST_HORIZONS,FORECAST_LENGTH)
    print(reframed.head())
    reframed_target = reframed['var'+target_lable+'(t)']
    print(reframed_target.head())

    # 划分训练数据和测试数据
    print("len(data) :",len(data))
    n_train=int(val_split*len(data))

    #target列
    train = reframed.values[:n_train, :]
    train_target =reframed_target.values[:n_train]

    # target列
    test = reframed.values[n_train:, :]
    test_target = reframed_target.values[n_train:]

    print("len(train)", len(train))
    # print("train", train)
    print("len(test)", len(test))
    # print("test", test)

    # 拆分输入输出 split into input and outputs
    train_X, train_y = train[:,:-5*FORECAST_LENGTH], train_target[:,]
    test_X,  test_y = test[:,:- 5*FORECAST_LENGTH], test_target[:,]

    scalar_X = MinMaxScaler(feature_range=(0, 1))    #RobustScaler()
    scalar_Y = MinMaxScaler(feature_range=(0, 1))

    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    # 在训练集上拟合并转换
    train_X = scalar_X.fit_transform(train_X)
    train_y = scalar_Y.fit_transform(train_y)

    # 在测试集上仅进行转换，使用训练集的统计信息
    test_X = scalar_X.fit_transform(test_X)
    test_y = scalar_Y.fit_transform(test_y)

    # reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, test_X, train_y, test_y,scalar_X,scalar_Y

def check_series(series):
    try:
        series = pd.Series(series)
    except:
        raise ValueError(
            'Sorry! %s is not supported for the Hybrid Method, please input pd.DataFrame, pd.Series, nd.array(<=2D)' % type(data))
    return series

