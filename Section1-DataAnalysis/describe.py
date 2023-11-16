import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import skew, kurtosis

class describe():
    def __init__(self):
        self.table = []

    def fill_in_the_data(self, _file):
        data_raw = pd.read_csv(_file)
        data = data_raw.dropna()

        self.table.append(["", "count", "mean", "std", "skew", "kurtosis", "variance", "min", "25%", "50%", "75%", "max"])

        for idx, (column_name, column_data) in enumerate(data.items()):
            if np.issubdtype(column_data.dtype, np.number) and idx != 0:
                name = column_name[:5]
                count = column_data.count()
                mean = round(column_data.mean(), 4)
                std = round(column_data.std(), 4)
                skewness = round(skew(column_data), 4)
                kurt = round(kurtosis(column_data), 4)
                variance = round(column_data.var(), 4)
                min_val = round(column_data.min(), 4)
                percentile_25 = round(np.percentile(column_data, 25), 4)
                percentile_50 = round(np.percentile(column_data, 50), 4)
                percentile_75 = round(np.percentile(column_data, 75), 4)
                max_val = round(column_data.max(), 4)

                self.table.append([name, count, mean, std, variance, skewness, kurt, min_val, percentile_25, percentile_50, percentile_75, max_val])

    def save_table_to_file(self, file_name):
        arr = np.array(self.table, dtype=object)    
        table_str = tabulate(np.transpose(arr), headers='firstrow', numalign="center")
        with open(file_name, "w") as file:
            file.write(table_str)

def main(): 
    desc = describe()
    desc.fill_in_the_data(sys.argv[1])
    desc.save_table_to_file("table")

if __name__ == "__main__":
	main()