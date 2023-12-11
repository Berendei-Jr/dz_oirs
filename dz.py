import csv
import random
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots

LINES_TO_PROCESS = 12000

def dispersion(data: list, expected_value: float) -> float:
    sum = 0
    for el in data:
        sum += (el - expected_value)**2
    return sum/len(data)

with open('EDA.csv') as csv_file:
    csv_reader = csv.reader(csv_file)

    data = []
    minutes_array = [0]
    frequency = None

    line = 0
    for row in csv_reader:
        if line > LINES_TO_PROCESS:
            break
        elif line == 1:
            line += 1
            frequency = float(row[0])
        elif line <= 3:
            line += 1
        else:
            data.append(float(row[0]))
            minutes_array.append(minutes_array[-1] + 1/frequency/60)
            line += 1
    minutes_of_measurment = LINES_TO_PROCESS/frequency/60
    minutes_array.pop(-1)
    plt.plot(minutes_array, data)
    plt.xlabel('Минуты')
    plt.ylabel('Электродермальная активность (мкСм)')
    plt.title('Зависимость данных датчика электродермальной активности от времени')
    plt.grid()
    #plt.show()
    plt.clf()

    expected_value = sum(data)/len(data)
    print(f'Математическое ожидание: {expected_value}')
    basic_dispersion = dispersion(data, expected_value)
    print(f'Дисперсия: {basic_dispersion}')

    data_with_noise = []
    min_noise = -2*data[0]
    max_noise = 2*data[0]
    for el in data:
        data_with_noise.append(el + random.uniform(min_noise, max_noise))
    noised_dispersion = dispersion(data_with_noise, expected_value)
    print(f'Дисперсия зашумленного сигнала: {noised_dispersion}')

    ds_minimum = min(data)
    ds_maximum = max(data)
    ds_range = ds_maximum - ds_minimum
    print(f'\nМинимальное значение: {ds_minimum}')
    print(f'Максимальное значение: {ds_maximum}')
    print(f'Размах: {ds_range}')

    #plot autocorrelation function
    #fig = tsaplots.plot_acf(data, lags=1000)
    acf = sm.tsa.acf(data)
    plt.figure(figsize=(16,12), dpi=150)
    #pd.plotting.autocorrelation_plot(grouped_data['total']).plot()
    plt.xlabel('Лаг')
    plt.ylabel('Автокорреляция')
    plt.xticks(rotation=90)
    plt.show()
    plt.savefig('2.jpg', format ='jpg', dpi=300)
    #plt.show()
