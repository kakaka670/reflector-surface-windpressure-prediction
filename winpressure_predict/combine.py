import os
import pandas as pd
import numpy as np
import csv
ANGLE='ANGLE 45_ave'
path = './windpressure_data/PITCH '+ ANGLE +'/'

def ave():
    '''
    Calculation of average wind pressure
    1min -->50 points
    '''
    # read csv
    input_file = './windpressure_data/PITCH '+ ANGLE +'/'+ANGLE+'.csv'
    df = pd.read_csv(input_file)

    # average
    average_values = df.groupby(df.index //50).mean()

    # save to csv
    output_file = './windpressure_data/PITCH ' + ANGLE +'/'+ANGLE+'.csv'
    average_values.to_csv(output_file, index=False)

def combine():
  '''
  Integration of raw wind pressure data, 10 minutes each, 6 files in total
  '''

  for files in os.listdir(path):
    datas = []
    for fname in ['030.csv','031.csv', '032.csv', '033.csv', '034.csv','035.csv']:
        if 'csv' in fname:
            fname_path = path + fname
            with open(fname_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                reader = list(reader)[1:]
                for line in reader:
                    datas.append(line)
  excel_name = './windpressure_data/PITCH ' + ANGLE +'/'+ANGLE+'.csv'
  csv_head = [
    'P1',
    'P2',
    'P3',
    'P4',
    'P5'
]

  with open(excel_name, 'w') as csvfile2:
    writer = csv.writer(csvfile2)
    # 写表头
    writer.writerow(csv_head)
    writer.writerows(datas)

  print ('finish~')

combine()
ave()