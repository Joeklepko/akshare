import os
import pandas as pd
from datetime import datetime
import pickle

from get_stop_date import get_stop_all_info

pre_stop1 = 10
af_stop = 10
stop_ruse_day = 20

name_list = os.listdir('./data/stock_all_daily')

stop_info = get_stop_all_info()

def get_dataset(i):
    date = stock_data.iloc[i, 1]
    if date < '2012-01-01':
        return []
    current_up_down = stock_data.iloc[i, 9]
    if current_up_down > 9 or current_up_down < -9:
        return []
    current_price = stock_data.iloc[i, 3] + 0.0001
    end_price = max(stock_data.iloc[i:i + af_stop, 3])
    up_down = (end_price / current_price - 1)*100
    if up_down < 20 and up_down > -20:
        return []
    price_hist = list(stock_data.iloc[i - pre_stop1:i, 4])
    price_ratio = [x/(price_hist[0] + 0.001) for x in price_hist]
    volume_hist = list(stock_data.iloc[i - pre_stop1:i, 7])
    volume_ratio = [x/(volume_hist[0] + 0.001) for x in volume_hist]

    print('get one')
    dataset = []
    dataset.extend(price_ratio[1:])
    dataset.extend(volume_ratio[1:])
    dataset.extend([up_down])
    return dataset


dataset_all = []
count = 1
for name in name_list:
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
            start_date = stock_data.iloc[index+af_stop, 1]
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

with open('./data/dataset/dataset_all_stg2.pkl', 'wb') as f:
    pickle.dump(dataset_all, f)

print('end')
