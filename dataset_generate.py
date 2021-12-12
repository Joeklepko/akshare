import os
import pandas as pd
from datetime import datetime
import pickle

from get_stop_date import get_stop_all_info

pre_stop1 = 50
pre_stop2 = 10
af_stop = 10
stop_ruse_day = 20

name_list = os.listdir('./data/stock_all_daily')

stop_info = get_stop_all_info()

def get_dataset(i):
    max_price1 = max(stock_data.iloc[i - pre_stop1:i, 4])
    max_date1 = pre_stop1 - list(stock_data.iloc[i - pre_stop1:i, 4]).index(max_price1)
    max_price2 = max(stock_data.iloc[i - pre_stop2:i, 4])
    max_date2 = pre_stop2 - list(stock_data.iloc[i - pre_stop2:i, 4]).index(max_price2)

    max_volume1 = max(stock_data.iloc[i - pre_stop1:i, 7])
    max_volume2 = max(stock_data.iloc[i - pre_stop2:i, 7])
    current_volume = stock_data.iloc[i, 7]

    current_price = stock_data.iloc[i, 3]
    end_price = stock_data.iloc[i + af_stop, 3]
    current_up_down = stock_data.iloc[i, 9]
    if current_up_down > 9 or current_up_down < -9:
        return []
    try:
        dataset = [max_price2 / max_price1, current_price / max_price1, max_date1, max_date2,
                   max_volume2 / max_volume1, current_volume / max_volume1, (end_price / current_price - 1)*100]
    except:
        return []
    return dataset


dataset_all = []
count = 1
for name in name_list[:20]:
    print(name, count, datetime.now())
    count += 1
    stock_data = pd.read_csv(f'./data/stock_all_daily/{name}')
    stock_code = name[:-4]
    stock_on_day = len(stock_data)
    if pre_stop1 >= stock_on_day - af_stop:
        continue
    if stock_code not in stop_info:
        for i in range(pre_stop1, stock_on_day-af_stop):
            dataset = get_dataset(i)
            if dataset:
                dataset_all.append(dataset)

    else:
        for i in range(pre_stop1, stock_on_day - af_stop):
            index = i - pre_stop1
            start_date = stock_data.iloc[index+20, 1]
            end_date = stock_data.iloc[index+pre_stop1 + af_stop, 1]
            flag = True
            for stop_day in stop_info[stock_code]:
                if (stop_day[0] >= start_date and stop_day[0] <= end_date) or \
                        (stop_day[1] >= start_date and stop_day[1] <= end_date):
                    flag = False
            if flag:
                dataset = get_dataset(i)
                if dataset:
                    dataset_all.append(dataset)

with open('./data/dataset/dataset_tmp.pkl', 'wb') as f:
    pickle.dump(dataset_all, f)
