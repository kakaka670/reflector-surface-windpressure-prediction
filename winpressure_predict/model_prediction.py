#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time
import pandas as pd
import warnings
from datetime import datetime
warnings.filterwarnings("ignore") # Ignore some annoying warnings
# winpressure_predict
from winpressure_predict.core import check_dataset, check_path, plot_save_result, name_predictor, output_result

try: from tensorflow import constant 
except: raise ImportError('Cannot import tensorflow, install or check your tensorflow verison!')
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM ,CuDNNGRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

ANGLE='ANGLE 90_ave'
n_feature=1
class keras_predictor:
    def __init__(self, PATH='./', FORECAST_HORIZONS=1, FORECAST_LENGTH=1, KERAS_MODEL='LSTM', DECOM_MODE='CEEMDAN', INTE_LIST='auto',EPOCH=100,EARLY_STOP=15,
                  REDECOM_LIST='', NEXT_DAY=False, DAY_AHEAD=1, NOR_METHOD='minmax', FIT_METHOD='add', USE_TPU=False , **kwargs):  #  REDECOM_LIST={'windpressure-imf0':'vmd'}

        self.PATH = PATH
        self.FORECAST_HORIZONS = int(FORECAST_HORIZONS)
        self.FORECAST_LENGTH = int(FORECAST_LENGTH)
        self.KERAS_MODEL = KERAS_MODEL
        self.DECOM_MODE = str(DECOM_MODE)
        self.INTE_LIST = INTE_LIST
        self.REDECOM_LIST = REDECOM_LIST
        self.NEXT_DAY = bool(NEXT_DAY)
        self.DAY_AHEAD = int(DAY_AHEAD)
        self.NOR_METHOD = str(NOR_METHOD)
        self.FIT_METHOD = str(FIT_METHOD)
        self.USE_TPU = bool(USE_TPU)

        self.TARGET = None
        self.VMD_PARAMS = None

        self.epochs = int(kwargs.get('epochs', EPOCH))
        self.dropout = float(kwargs.get('dropout', 0.2))
        self.units = int(kwargs.get('units',16))
        self.activation = str(kwargs.get('activation','tanh'))
        self.batch_size = int(kwargs.get('batch_size',32))
        self.shuffle = bool(kwargs.get('shuffle',False))
        self.verbose = int(kwargs.get('verbose', 1))
        self.valid_split = float(kwargs.get('valid_split', 0.1))
        self.opt = str(kwargs.get('opt', 'adam'))
        self.opt_lr = float(kwargs.get('opt_lr', 0.001))
        self.opt_loss = str(kwargs.get('opt_loss', 'mse'))
        self.opt_patience = int(kwargs.get('opt_patience', 10))  #10
        self.stop_patience = int(kwargs.get('stop_patience',EARLY_STOP))  #10
        self.callbacks_monitor = str(kwargs.get('callbacks_monitor', 'val_loss'))

        # check
        self.PATH, self.FIG_PATH, self.LOG_PATH  = check_path(PATH) # Check PATH

        if self.FORECAST_HORIZONS <= 0: raise ValueError("Invalid input for FORECAST_HORIZONS! Please input a positive integer >0.")
        if self.FORECAST_LENGTH <= 0: raise ValueError("Invalid input for FORECAST_LENGTH! Please input a positive integer >0.")
        if self.DAY_AHEAD < 0: raise ValueError("Invalid input for DAY_AHEAD! Please input a integer >=0.")
        if self.epochs < 0: raise ValueError("Invalid input for epochs! Please input a positive integer >0.")
        if self.units <= 0: raise ValueError("Invalid input for units! Please input a positive integer >0.")
        if self.verbose not in [0, 1, 2] <= 0: raise ValueError("Invalid input for verbose! Please input 0 - not displayed, 1 - detailed, 2 - rough.")
        if self.opt_patience <= 0: raise ValueError("Invalid input for opt_patience! Please input a positive integer >0.")
        if self.stop_patience <= 0: raise ValueError("Invalid input for stop_patience! Please input a positive integer >0.")
        if self.dropout < 0 or self.dropout > 1: raise ValueError("Invalid input for dropout! Please input a number between 0 and 1.")
        if self.opt_lr < 0 or self.opt_lr > 1: raise ValueError("Invalid input for opt_lr! Please input a number between 0 and 1.")
        
        if not isinstance(KERAS_MODEL, Sequential): # Check KERAS_MODEL
            if type(KERAS_MODEL) == str: 
                if '.h5' not in str(self.KERAS_MODEL): self.KERAS_MODEL = KERAS_MODEL.upper()
                else:
                    if self.PATH is None: raise ValueError("Please set a PATH to load keras model in .h5 file.")
                    if not os.path.exists(self.PATH+self.KERAS_MODEL): raise ValueError("File does not exist:", self.PATH+self.KERAS_MODEL)
            else:
                if self.PATH is None: raise ValueError("Please set a PATH to load keras model in .h5 file.")
                try: KERAS_MODEL = pd.DataFrame(KERAS_MODEL, index=[0])
                except: raise ValueError("Invalid input for KERAS_MODEL! Please input eg. 'GRU', 'LSTM', or model = Sequential(), h5 file, dict, pd.DataFrame.")
                for file_name in KERAS_MODEL.values.ravel():
                    if '.h5' not in str(file_name): raise ValueError("Invalid input for KERAS_MODEL values! Please input eg. 'GRU', 'LSTM', or model = Sequential(), h5 file, dict, pd.DataFrame.")
                    if not os.path.exists(self.PATH+file_name): raise ValueError("File does not exist:", self.PATH+file_name)

        if type(DECOM_MODE) == str: self.DECOM_MODE = str(DECOM_MODE).upper() # Check
        if self.opt_lr != 0.001 and self.opt == 'adam': self.opt = Adam(learning_rate=self.opt_lr) # Check optimizer
        if self.opt_patience > self.epochs: self.opt_patience = self.epochs // 10 # adjust opt_patience
        if self.stop_patience > self.epochs and self.stop_patience > 10: self.stop_patience = self.epochs // 2 # adjust stop_patience


    class HiddenPrints:
        """
        A class used to hide the print
        """
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    # Build model
    def build_model(self, trainset_shape, model_name='Keras model', model_file=None):
        """
        Build Keras model, eg. 'GRU', 'LSTM', 'CUDNNLSTM', 'CUDNNGRU', model = Sequential(), or load_model.
        """
        if model_file is not None and os.path.exists(str(model_file)):
            print('Load Keras model:', model_file)
            return load_model(model_file) # load user's saving custom model
        elif isinstance(self.KERAS_MODEL, Sequential): # if not load a model
            return self.KERAS_MODEL

        elif self.KERAS_MODEL =='LSTM':
            model = Sequential(name=model_name)
            model.add(CuDNNLSTM(self.units *2, input_shape=(trainset_shape[1], trainset_shape[2]), return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(CuDNNLSTM(self.units *2, return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(CuDNNLSTM(self.units , return_sequences=False))
            model.add(Dropout(self.dropout))
            model.add(Dense(n_feature*self.FORECAST_LENGTH, activation=self.activation))
            self.opt_lr=0.001
            self.opt = Adam(learning_rate=self.opt_lr)
            model.compile(loss=self.opt_loss, optimizer=self.opt)
            return model

        elif self.KERAS_MODEL == 'GRU':
            model = Sequential(name=model_name)
            model.add(CuDNNGRU(self.units *2, input_shape=(trainset_shape[1], trainset_shape[2]), return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(CuDNNGRU(self.units *2, return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(CuDNNGRU(self.units , return_sequences=False))
            model.add(Dropout(self.dropout))
            model.add(Dense(n_feature*self.FORECAST_LENGTH, activation=self.activation))
            self.opt_lr = 0.001
            self.opt = Adam(learning_rate=self.opt_lr)
            model.compile(loss=self.opt_loss, optimizer=self.opt)
            return model

            # BiLSTM
        elif self.KERAS_MODEL == 'BILSTM':
            model = Sequential(name=model_name)
            model.add(Bidirectional(CuDNNLSTM(self.units * 2, return_sequences=True),input_shape=(trainset_shape[1], trainset_shape[2])))
            model.add(Dropout(self.dropout))
            model.add(Bidirectional(CuDNNLSTM(self.units * 2, return_sequences=True)))
            model.add(Dropout(self.dropout))
            model.add(Bidirectional(CuDNNLSTM(self.units, return_sequences=False)))
            model.add(Dropout(self.dropout))
            model.add(Dense(n_feature*self.FORECAST_LENGTH, activation=self.activation))
            self.opt_lr = 0.001
            self.opt = Adam(learning_rate=self.opt_lr)
            model.compile(loss=self.opt_loss, optimizer=self.opt)
            return model
        else: raise ValueError("%s is an invalid input for KERAS_MODEL! eg. 'GRU', 'LSTM', or model = Sequential()"%self.KERAS_MODEL)

    def keras_predict(self,target='',imf_name='',data=None,DECOM_MODE='OVMD', show_model=False, **kwargs):
        from winpressure_predict.data_preprocessor import create_train_test_set
        x_train, x_test, y_train, y_test, scalar_X, scalar_Y = create_train_test_set(data, self.FORECAST_LENGTH, self.FORECAST_HORIZONS,target_lable=target)
        print(" y_train:------------", y_train)
        try:
            data.name = data.name.replace('-','_').replace(' ','_')
            if '.h5' not in str(data.name):
                data.name = data.name+'.h5'
        except: data.name = self.KERAS_MODEL+'_'+imf_name+'Keras_model.h5'
        
        # Load and save model
        model_file = None
        Reduce = ReduceLROnPlateau(monitor=self.callbacks_monitor, patience=self.opt_patience, verbose=self.verbose,
                                       mode='auto')  # Adaptive learning rate
        EarlyStop = EarlyStopping(monitor=self.callbacks_monitor, patience=self.stop_patience, verbose=self.verbose,
                                      mode='auto')  # Early stop at small learning rate
        callbacks_list = [Reduce, EarlyStop]

        if self.PATH is not None:
            # Load model if get input of self.KERAS_MODEL
            if isinstance(self.KERAS_MODEL, dict): self.KERAS_MODEL = pd.DataFrame(self.KERAS_MODEL, index=[0])
            if isinstance(self.KERAS_MODEL, pd.DataFrame):
                for x in self.KERAS_MODEL.columns:
                    if (x).replace('-','_').replace(' ','_') in data.name: model_file = x # change to be key value
                if model_file is not None: model_file = self.PATH + self.KERAS_MODEL[model_file][0]
                else: raise KeyError("Cannot match an appropriate model file by the column name of pd.DataFrame. Please check KERAS_MODEL.")
            if isinstance(self.KERAS_MODEL, str) and '.h5' in str(self.KERAS_MODEL): model_file = self.PATH + self.KERAS_MODEL

            # Save model by CheckPoint with model name = data.name = df_redecom.name
            CheckPoint = ModelCheckpoint(self.PATH+'/model/'+ANGLE+'_'+DECOM_MODE+'_'+data.name, monitor=self.callbacks_monitor, save_best_only=True, verbose=self.verbose, mode='auto') # Save the model to self.PATH after each epoch
            callbacks_list.append(CheckPoint)  # save Keras model in .h5 file

        # Build or load the model
        from winpressure_predict.data_preprocessor import eval_result
        model = self.build_model(x_train.shape, data.name, model_file)

        if show_model:
            print('\nInput Shape: (%d,%d)\n'%(x_train.shape[1], x_train.shape[2]))
            model.summary()
        print('**********{}************'.format(data.name))

        # from tensorflow.keras.utils import plot_model
        # plot_model(model,to_file=self.FIG_PATH+self.KERAS_MODEL+'/'+self.KERAS_MODEL+'-model.png',show_shapes=True)

        # Training
        if self.epochs != 0:
            print("=======================training===============================")
            history = model.fit(x_train, y_train, # Train the model
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                validation_split=self.valid_split,
                                verbose=self.verbose,
                                shuffle=self.shuffle,
                                callbacks=callbacks_list,
                                **kwargs)
        df_loss = pd.DataFrame({'loss': history.history['loss'], 'val_loss': history.history['val_loss']},
                                   index=range(len(history.history['val_loss'])))

        print("origin x_test.shape:",x_test.shape)
        y_predict = model.predict(x_test) # Predict
        print("y_predict: ",y_predict)

        x_test_v= x_test.reshape((x_test.shape[0], x_test.shape[2]))
        y_test_v= y_test.reshape((y_test.shape[0], y_test.shape[1]))

        print("y_predict.shape:",y_predict.shape)
        print("y_test.shape:",y_test_v.shape)

        if scalar_Y is not None:
            #inverse
            x_test_v=scalar_X.inverse_transform(x_test_v)
            y_predict_v = scalar_Y.inverse_transform(y_predict)
            print("y_predict_v :\n", y_predict_v)
            y_test_v = scalar_Y.inverse_transform(y_test_v)
            print("y_test_v :\n", y_test_v)

            if self.TARGET is not None:
                result_index = self.TARGET.index[-self.FORECAST_LENGTH:] # predicting result idnex
            else:
                result_index = range(y_test.shape[0])
        print("result_index: ",result_index)
        df_eval = eval_result(y_test_v, y_predict_v)

        if imf_name=='':
            df_result = pd.DataFrame({
            # every predict result
            'P' + target : y_test_v[:, 0],
            'Predict-P' + target : y_predict_v[:, 0]
            }, index=result_index)
        else:
            df_result = pd.DataFrame({
            #every predict result
            'P'+target+'_'+imf_name: y_test_v[:,0],
            'Predict-P'+target+'_'+imf_name: y_predict_v[:,0]
            }, index=result_index)
        print("------------------finish  prediction------------------")
        return df_result, df_eval ,df_loss


    #Single prediction
    def single_keras_predict(self, data=None, show=False, plot=False, **kwargs):

        now = datetime.now()
        predictor_name = name_predictor(now, 'Single', 'Keras', self.KERAS_MODEL, None, None, self.NEXT_DAY)
        data = check_dataset(data, show, self.DECOM_MODE, None)
        data.name = (ANGLE+'_'+predictor_name+' model.h5')
        data.drop(data.columns[5],axis=1,inplace=True)

        df_empty = pd.DataFrame()
        data_final_eval = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE', 'Runtime'])
        # prediction
        for i in ['1','2','3','4','5']:
          start = time.time()
          df_result, df_eval ,df_loss = self.keras_predict(data=data,imf_name='',target=i, show_model=show,**kwargs)
          end = time.time()

          df_all_pre_result = pd.concat([df_empty, df_result], axis=1)
          df_empty = df_all_pre_result

          df_result ,final_eval = output_result(df_result, predictor_name+'-P'+i, end - start,i=i, imf=ANGLE+'-P'+i)
          data_final_eval = pd.concat((data_final_eval, final_eval), axis=0)

          # save fig
          name = 'single_keras_predict result'
          plot_save_result(df_result, name=name, plot=plot, save_plot=False, save_log=True, path=self.PATH)

        # save Final result
        df_all_pre_result.to_csv('./predict_result/log/' + ANGLE + '_' + predictor_name + '_ALL_PREDICT_RESULT' + '.csv', index=0)
        # save evl result
        data_final_eval.to_csv('./predict_result/log/' + ANGLE + '_' + predictor_name + ' Final Evaluation.csv')
        # Output
        print("===========successfully save single predict plot================")
        return df_result


    #Decom prediction
    def Decom_keras_predict(self, data=None, keras_model='',DECOM_MODE='VMD', show=True, plot=True, **kwargs):

        #   reconstruct
        def reconstruct(data, df_all_pre_result,time):
            data_final = pd.DataFrame()
            data_final_eval = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE','Runtime','VMF'])

            df_all_pre_resultl=df_all_pre_result
            data_test = data.iloc[-len(df_all_pre_resultl):, :]
            data_test.reset_index(drop=True, inplace=True)
            data_final = pd.concat((data_final, data_test), axis=1)

            for SENSOR in ['1','2','3','4','5']:
                data_predict_result = df_all_pre_result.filter(like='Predict-P'+SENSOR)
                data_final['Predict-P'+SENSOR] = data_predict_result.sum(axis=1)
                df_evl_result,final_eval= output_result(data_final,predictor_name+'P'+SENSOR,time=time,i=SENSOR,imf='P'+SENSOR+' Final')
                data_final_eval =pd.concat((data_final_eval,final_eval), axis=0)
            data_final_eval.to_csv('./predict_result/log/'+ANGLE+'_'+predictor_name+' Final Evaluation.csv')
            print("df_evl_result :\n", df_evl_result)
            return data_final

        # Name
        self.DECOM_MODE=DECOM_MODE
        now = datetime.now()
        predictor_name = name_predictor(now,'Decom', 'Keras', self.KERAS_MODEL, self.DECOM_MODE, self.REDECOM_LIST, self.NEXT_DAY)
        # Decompose
        start = time.time()
        from winpressure_predict.data_preprocessor import decom
        read_decom_result=True  # read_decom_result.csv
        if read_decom_result==True:
            df_all_decom=pd.read_csv('./predict_result/DECOM/' + ANGLE +'/'+DECOM_MODE+'_'+ANGLE+ '_DECOM_RESULT.csv',header=0)
            print("successfully read")
        else:
          print("----------------start {} decom----------------------".format(self.DECOM_MODE))
          df_ini_decom = pd.DataFrame()
          print("df_ini_decom:--------------------\n", df_ini_decom)
          for i in ['P1', 'P2', 'P3', 'P4','P5']:
                df_decom,df_decom_target= decom(series=data[i], decom_mode=self.DECOM_MODE,draw=True,SENSOR=i)
                df_all_decom= pd.concat([df_ini_decom, df_decom], axis=1)
                df_ini_decom=df_all_decom
                print("successfully decompose:----------{}----------=".format(i))
          df_all_decom.to_csv('./predict_result/DECOM/' + ANGLE +'/'+DECOM_MODE+'_'+ANGLE+ '_DECOM_RESULT.csv', index=False)
          print("df_all_decom:--------------------------------\n", df_all_decom)

        # Predict and ouput each wind pressure-VMF
        print(df_all_decom.columns)

         # df for storing evaluation result
        df_empty = pd.DataFrame()
        df_result, df_eval, df_loss = [], [], []
        df_all_pre_result = []

        for VMF in ['VMF0','VMF1','VMF2','VMF3']:
            df_new_decom=df_all_decom.filter(like=VMF)
            print("df_new_decom:--------------------------------\n", df_new_decom)
            df_eval_result = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE', 'Runtime', 'IMF'])
            for SENSOR in ['1', '2', '3', '4', '5']:
                # predict
                start = time.time()
                df_result, df_eval, df_loss = self.keras_predict(data=df_new_decom,DECOM_MODE=DECOM_MODE,imf_name=VMF,target=SENSOR, show_model=show, **kwargs)
                end = time.time()

                df_all_pre_result = pd.concat([df_empty, df_result], axis=1)
                df_empty = df_all_pre_result
                plot_predictor_name=predictor_name + VMF + '-P' + SENSOR
                print("df_result: \n", df_result)

                # save figure
                name = 'decom_keras_predict result'
                plot_save_result(df_result, name=plot_predictor_name, plot=plot, save_plot=False,save_log=False, path=self.PATH)

                #plt  every  sensor
                df_eval=pd.DataFrame(df_eval)
                df_eval_result = pd.concat((df_eval_result, df_eval),axis=0)


        # Final result
        df_all_pre_result.to_csv('./predict_result/log/'+keras_model+'_'+ DECOM_MODE +ANGLE + '_' +  '_ALL_PREDICT_RESULT' + '.csv', index=0)

        # Read result
        # df_all_pre_result=pd.read_csv('./predict_result/log/'+keras_model+'_'+ DECOM_MODE +ANGLE + '_' +  '_ALL_PREDICT_RESULT' + '.csv',index_col=False)

        end = time.time()
        data_final = reconstruct(data, df_all_pre_result, time=end - start)

        plot_save_result(data_final, name=predictor_name, plot=plot, save_plot=True,save_log=True, path=self.LOG_PATH + ANGLE +'_'+DECOM_MODE+'/')

        end = time.time()
        print("===========successfully save FIGURE================")
        return df_all_pre_result





