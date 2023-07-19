import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from scipy.stats import skew, kurtosis
import seaborn as sns
from log_reg import MyLogisticRegression

def main():
	#read the data
    data_raw = pd.read_csv("dataset_train.csv")

    #remove missing data points
    data = data_raw.dropna()

    # #create a new table with the statistics
    # table = []
    # table.append(["", "count", "mean", "std", "skew", "kurtosis", "variance", "min", "25%", "50%", "75%", "max"])
    # min_std_cat = "not found"
    # min_std = np.inf

    # for idx, (column_name, column_data) in enumerate(data.iteritems()):
    #     if np.issubdtype(column_data.dtype, np.number) and idx != 0:
    #         name = column_name[:5]
    #         count = column_data.count()
    #         mean = round(column_data.mean(), 4)
    #         std = round(column_data.std(), 4)
    #         skewness = round(skew(column_data), 4)
    #         kurt = round(kurtosis(column_data), 4)
    #         variance = round(column_data.var(), 4)
    #         min_val = round(column_data.min(), 4)
    #         percentile_25 = round(np.percentile(column_data, 25), 4)
    #         percentile_50 = round(np.percentile(column_data, 50), 4)
    #         percentile_75 = round(np.percentile(column_data, 75), 4)
    #         max_val = round(column_data.max(), 4)

    #         # Append the column to the new table
    #         table.append([name, count, mean, std, variance, skewness, kurt, min_val, percentile_25, percentile_50, percentile_75, max_val])

    #         #check for minimum
    #         if (std < min_std):
    #             min_std_cat = column_name
    #             min_std = std

    # #print the table
    # arr = np.array(table, dtype=object)    
    # print(tabulate(np.transpose(arr), headers='firstrow', numalign="center"))

    # #section 1: print the category with lowest std
    # print("minimum std belongs to: ", min_std_cat)

    # # plot the histogram of the category
    # categories = data['Hogwarts House'].unique()
    # colors = ['red', 'green', 'blue', 'yellow']

    # i = 0
    # for category in categories:
    #     category_data = data[data['Hogwarts House'] == category]
    #     plt.hist(category_data[min_std_cat], bins=10, alpha=0.5, color=colors[i], label=category)
    #     i += 1
    
    # plt.title("Histogram of " + min_std_cat + " by Hogwarts House")
    # plt.xlabel(min_std_cat)
    # plt.ylabel("Frequency")
    # plt.legend(title="Hogwarts House")

    # # section 2: calculating the correlation between every two feature
    # correlation_matrix = [['', 'Arith', 'Astro', 'Herbo', 'Defen', 'Divin', 'Muggl', 'Ancie', 'Histo', 'Trans', 'Potio', 'Care ', 'Charm', 'Flyin']]  # First row with feature names
    
    # columns_to_drop = ["Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]
    # data_wo = data.drop(columns=columns_to_drop)
    # Feature_names = data_wo.columns
    
    # # Initialize the remaining cells with zeros
    # for i in range(1, len(correlation_matrix[0])):
    #     correlation_matrix.append([correlation_matrix[0][i], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # for i in range(1, len(correlation_matrix[0])):
    #     for j in range(1, len(correlation_matrix[0])):
    #         feature1 = Feature_names[i]
    #         feature2 = Feature_names[j]
    #         correlation_matrix[i][j] = data[feature1].corr(data[feature2])

    # print(tabulate(correlation_matrix, headers='firstrow', numalign="center"))

    # #manually looked into the correlation of 1 or -1: make this automatic later
    # plt.figure()
    # plt.scatter(data["Defense Against the Dark Arts"], data["Astronomy"])
    # plt.title("Defense Against the Dark Arts vs Astronomy")

    # #section 3: pair plot
    # sns.set(font_scale=0.3)  # You can adjust the font_scale value to change the font size
    # col_todrop = ["Index", "First Name", "Last Name", "Birthday", "Best Hand"]
    # data_pp = data.drop(columns=col_todrop)
    # plot = sns.pairplot(data_pp, hue='Hogwarts House', diag_kind='hist', height=0.5, plot_kws={'s': 2})
    # sns.despine()
    # plt.show()

    #training
    X = data[["Herbology", "Astronomy", "Ancient Runes", "Defense Against the Dark Arts"]]
    y = data["Hogwarts House"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Instantiate the MyLogisticRegression class
    logistic_regression = MyLogisticRegression(alpha=0.001, max_iter=1000)

    # Step 3: Fit the model to your training data
    logistic_regression.fit(X_train, y_train)

    # Step 4: Predict the class labels for your test data
    y_pred = logistic_regression.predict(X_test)

    print(y_pred)

if __name__ == "__main__":
	main()
