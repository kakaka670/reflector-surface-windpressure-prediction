#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ANGLE='ANGLE 90_ave'
# Show Tensorflow running device
def show_devices():
    try:
        try:
            import tensorflow as tf # for tensorflow and keras
            print("Tensorflow version:", tf.__version__)
            print("Available devices:") 
            print(tf.config.list_physical_devices())
        except: 
            import torch # if install Pytorch
            print("Pytorch version:", torch.__version__)
            print("CUDA:", torch.cuda.is_available(), "version:", torch.version.cuda, ) # 查看CUDA的版本号
            print("GPU:", torch.cuda.get_device_name())
    except: raise ImportError('Please install Tensorflow or Pytorch firstly!')

# Load the example data set
def load_dataset(dataset_name):
    dataset_location = os.path.dirname(os.path.realpath(__file__)) + '/windpressure_data/'
    df_windpressure = pd.read_csv(dataset_location+dataset_name, header=0)
    P_ANGLE=df_windpressure
    print(P_ANGLE)

    return P_ANGLE

# Check dataset
def check_dataset(data, show_data=False, decom_mode=None, redecom_list=None):

    # 原始
    if data is None:
        raise ValueError('Please input data!')
    #检查数据
    try: check_data = pd.DataFrame(data)
    except: raise ValueError('Invalid input of dataset %s!'%type(data))

    check_data = pd.DataFrame(data)
    if decom_mode is None: decom_mode = ''
    if redecom_list is None: redecom_list = {}
    if pd.isnull(check_data.values).any(): raise ValueError('Please check inputs! There is NaN!')
    if not check_data.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all(): 
        raise ValueError('Please check inputs! Cannot convert it to number.')


    for x in redecom_list.values(): 
        if 'vmd' in x.lower() and len(check_data)%2: check_data = check_data.sort_index()[1:] 
    if 'vmd' in decom_mode.lower() and len(check_data)%2: check_data = check_data.sort_index()[1:] 

    # Set target
    if 'target' not in check_data.columns:
        if len(check_data.columns) == 1: check_data.columns=['target']
        elif decom_mode.lower() in ['emd', 'eemd', 'ceemdan', 'vmd', 'ovmd', 'svmd']: 
            check_data['target'] = check_data.sum(axis=1).values
            print('Warning! The sum of all column has been set as the target column.') 
        else:
            check_data.columns = check_data.columns[1:].insert(0, 'target') # set first columns as target 
            print('Warning! The first column has been set as the target column.')  
            print('Or, you can set DECOM_MODE as "emd" to let the sum of all column be the target column.')  

    # Show the inputting data
    if show_data:
        print('Data type is %s.'%type(check_data))
        print('Part of inputting dataset:')
    return check_data

# Check PATH
def check_path(PATH):
    if PATH is not None:
        if type(PATH) != str:
            raise TypeError('PATH should be strings such as D:/winpressure_predict/.../.')
        else:
            if PATH[-1] != '/': PATH = PATH + '/'
            PATH = './predict_result/'
            FIG_PATH = PATH+'figures/'
            LOG_PATH = PATH+'log/'
            print('Saving path of figures and logs: %s'%(PATH))
            for p in [PATH, FIG_PATH, LOG_PATH]: # Create folders for saving
                if not os.path.exists(p): os.makedirs(p)
    else: PATH, FIG_PATH, LOG_PATH = None, None, None
    return PATH, FIG_PATH, LOG_PATH

# Name the predictor
def name_predictor(now, name, module, model, decom_mode=None, redecom_list=None, next_pred=False):
    """
    Name the predictor for convenient saving.
    """
    redecom_mode = ''
    # if redecom_list is not None: # Check redecom_list and get redecom_mode
        # try: redecom_list = pd.DataFrame(redecom_list, index=[0])
        # except: raise ValueError("Invalid input for redecom_list! Please input eg. None, '{'windpressure-imf0':'vmd', 'windpressure-imf1':'emd'}'.")
        # for i in (redecom_list+redecom_list.columns.str[-1]).values.ravel(): redecom_mode = redecom_mode+i+'-'

    if type(model) == str and '.h5' not in str(model):
        if 'Single' not in name:
            if decom_mode is not None: 
                name = name + ' ' + decom_mode.upper() + '-' 
                name = name + redecom_mode.upper()
        else: name = name + ' '
        name = name + model.upper()  # predicting model
    else: name = name+' Custom Model'
    if next_pred: name = name+' Next-day' # Next-day predicting or not
    name = name + ' ' + module+' predicting'
    print('==============================================================================')
    print(str(now.strftime('%Y-%m-%d %H:%M:%S'))+' '+ name +' is running...')
    print('==============================================================================')
    return name

def output_result(df_result, name, time, imf='',i='', run=None):
    # Output Result and add Runtime
    imf_name, run_name = '', ''
    if run is not None: run_name = 'Run'+str(run)+'-'
    if imf != '' and imf != 'Final': 
        imf_name = ' of '+imf
        print('\n----------'+ANGLE+'_'+name+imf_name+' Finished----------')
    else: print('\n================'+ANGLE+'_'+name+' Finished================')
    
    def finish_evaluation(final_pred, df_eval=None):
        from winpressure_predict.data_preprocessor import eval_result

        if df_eval is None: #
            final_eval= eval_result(final_pred['P'+i],final_pred['Predict-P'+i])
            final_eval= pd.DataFrame(final_eval,columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE'])  #
            final_eval['Runtime'], final_eval['VMF'] = time, imf
            print("final_eval:\n",final_eval)

        elif len(df_eval)==1: 
            final_eval = df_eval
            final_eval['Runtime'], final_eval['VMF'] = time, imf

        elif 'Final' in df_eval['VMF'].values:
            final_eval = df_eval
        else:
            final_eval = eval_result(final_pred['Actual'], final_pred['Predict'])
            final_eval['Runtime'], final_eval['VMF'] = time, imf
            final_eval = pd.concat((final_eval, df_result[1]))
        final_eval.name = name+' Evaluation'+imf_name

        return final_eval

    if isinstance(df_result, tuple) and len(df_result)==3: # input (df_result, df_eval, df_loss)
            final_pred, final_pred.name = df_result[0], name+' Result'+imf_name
            df_result[2].name = name+' Loss'+imf_name
            # print("final_pred: \n", final_pred)
            final_eval = finish_evaluation(final_pred,df_eval=df_result[1])   # evaluation for final
            final_pred.columns = run_name+final_pred.columns
            if 'of' in imf_name: # not Final
                final_pred.columns = run_name+imf+'-'+final_pred.columns 
                df_result[2].columns = run_name+imf+'-'+df_result[2].columns 
            final_eval['IMF'] = run_name + final_eval['IMF']
            df_result = (final_pred, final_eval, df_result[2]) # return (final_pred, final_eval)

    elif isinstance(df_result, tuple) and len(df_result)==2: # input (df_pred, df_eval)
            final_pred, final_pred.name = df_result[0], name+' Result'+imf_name
            final_eval = finish_evaluation(final_pred,df_eval=df_result[1]) # evaluation for final
            final_pred.columns = run_name+final_pred.columns
            if 'of' in imf_name: final_pred.columns = run_name+imf+'-'+final_pred.columns # not Final
            final_eval['IMF'] = run_name + final_eval['IMF']
            df_result = (final_pred, final_eval) # return (final_pred, final_eval)

    elif isinstance(df_result, pd.DataFrame): # input df_pred
            final_pred, final_pred.name = df_result,ANGLE+'_'+name+' Result'
            final_eval = finish_evaluation(final_pred)
            df_result = (final_pred,final_eval) # return (final_pred, final_eval)
    else: raise ValueError('Unknown Error.')

    return df_result ,final_eval

# Plot and save data
def plot_save_result(data, name=None, plot=True, save_plot=True, save_log=True,path=None, type=None):

    PATH, FIG_PATH, LOG_PATH = check_path(path)
    if PATH is None:
        save_plot,save_log = False,False

    def default_output(df, file_name): 
        if 'Evaluation' not in df.name and 'Next' not in df.name and plot:
            if df.columns.size<3:
                df.plot(figsize=(10,4))
            else:
                df.plot(figsize=(12,6))
            plt.title(df.name, fontsize=12, y=1)
            if save_plot:
                plt.savefig(FIG_PATH+ ANGLE +'_predict result_'+file_name+'.jpg', dpi=300, bbox_inches='tight') # Save figure
            plt.show()
        if save_log:
            pd.DataFrame.to_csv(df, LOG_PATH+ANGLE+'_'+file_name+'.csv', encoding='utf-8',index_label=0) # Save log

    # Ouput
    if isinstance(data, tuple):
        print("--------------------------data is tuple-------------------")
        for df in data:
            try: file_name = name + df.name.replace('-','_').replace(' ','_') # Check df Name
            except: df.name, file_name = 'output', name+'output'
            default_output(df, file_name)

    elif isinstance(data, pd.DataFrame):
        print("--------------------------data is DataFrame-------------------")
        try: file_name = ANGLE +'_'+name + data.name.replace('-','_').replace(' ','_') # Check data Name
        except: data.name, file_name = '-output', name+'output'
        # if 'decom' in data.name:
        if plot:
            data.plot(figsize=(12,6))  # subplots=True
            plt.title(name)
            if save_plot:
                plt.savefig(FIG_PATH +ANGLE +'_'+ name+'.jpg', dpi=300, bbox_inches='tight') # Save figure 保存图片
            plt.show()
        if save_log: pd.DataFrame.to_csv(data, LOG_PATH + ANGLE +'_'+name +'.csv', encoding='utf-8',index=0) # Save log 保存日志

    else:
        try: series = pd.Series(data)
        except: raise ValueError('Sorry! %s is not supported to plot and save, please input pd.DataFrame, pd.Series, nd.array(<=2D)'%type(data))
        default_output(series, file_name=name)
    if PATH is not None and save_plot: print('The figures and  of predicting results have been saved ')


