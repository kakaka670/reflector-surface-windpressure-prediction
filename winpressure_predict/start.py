import winpressure_predict as wp
import pandas as pd
import time

if __name__ == '__main__':
    t0 = time.time()
    ANGLE='ANGLE 90_ave'

    dataset ='PITCH '+ ANGLE +'/'+ANGLE+'.csv'
    data = wp.load_dataset(dataset)
    FORECAST_HORIZONS=1
    FORECAST_LENGTH=1
    DECOM_MODE = 'OVMD'
    keras_model='GRU'

    kp = wp.keras_predictor(FORECAST_HORIZONS=FORECAST_HORIZONS, FORECAST_LENGTH=FORECAST_LENGTH,DECOM_MODE=DECOM_MODE,KERAS_MODEL=keras_model, EPOCH=1000,EARLY_STOP=50)
    df_result = kp.single_keras_predict(data,show=False, plot=False)   # single
    # df_result = kp.Decom_keras_predict(data,keras_model=keras_model,DECOM_MODE='OVMD', show=True, plot=True)  # decom
    t1 = time.time()

    print("running time: {}s".format(t1-t0))