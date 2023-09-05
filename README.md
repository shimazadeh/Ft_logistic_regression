# DataScience x Logistic Regression | School-42 project
Implement one-vs-all logistic regression that will solve classification problem. Recreated Poudlard's Sorting Hat by implementing 
logistic regression from scratch. 

## Requirements:
- Python 3
- NumPy
- Pandas
- Matplotlib
- Sklearn
- Seaborn
- Tabulate
- Scipy

## Set-up
<pre><code>
  git clone https://github.com/shimazadeh/Ft_logistic_regression.git DSLR
  cd DSLR
  pip3 install -r requirements.txt</code></pre>

## Data Analysis
describe.py is implementation of pandas.DataFrame.describe. This program takes a dataset as a parameter and it displays all the statistical 
parameters of all numerical features. See the data analysis folder for the code implementation. Here is the output of the dataset used in this project:

|          | Arithmancy | Astronomy | Herbology | Defense Against the Dark Arts | Divination | Muggle Studies | Ancient Runes | History of Magi | Transfiguration | Potions | Care of Magical Creatures | Charms | Flying |
| -------- | ----------- | -------- | -------- | -------- | ------- | -------- | ------- | ------- | ------- | ------- | ------- | -------- | ------- |
| count | 1251 | 1251 | 1251 | 1251 | 1251 | 1251 | 1251 | 1251 | 1251 | 1251 | 1251 | 1251 | 1251 |
| mean     | 49453.1     | 46.4764  | 1.1895   | -0.4648  | 3.2138  | -222.904 | 496.252 | 2.9786  | 1029.86 | 5.9613  | -0.0643 | -243.326 | 23.109  |
| std      | 16701.6     | 520.946  | 5.2231   | 5.2095   | 4.111   | 484.986  | 106.711 | 4.457   | 43.9829 | 3.1029  | 0.9726  | 8.7904   | 97.755  |
| skew     | 2.78942e+08 | 271385   | 27.2812  | 27.1385  | 16.9003 | 235211   | 11387.2 | 19.8645 | 1934.49 | 9.6281  | 0.946   | 77.2712  | 9556.04 |
| kurtosis | -0.0525     | -0.1174  | -0.4316  | 0.1174   | -1.4067 | 0.8039   | 0.0318  | -1.0414 | -1.2183 | 0.0033  | -0.0202 | 0.3781   | 0.859   |
| variance | 0.2119      | -1.693   | -1.3692  | -1.693   | 0.6879  | -0.7592  | -1.5902 | -0.1    | 0.1994  | -0.5513 | 0.0342  | -1.088   | -0.1605 |
| min      | -24370      | -966.74  | -10.2957 | -10.1621 | -8.727  | -1043.96 | 283.87  | -8.4311 | 906.627 | -3.6208 | -3.3137 | -261.049 | -181.47 |
| 25%      | 38180       | -485.323 | -4.2523  | -5.2835  | 3.1205  | -573.969 | 396.41  | 2.2309  | 1025.64 | 3.6842  | -0.6944 | -250.586 | -40.085 |
| 50%      | 48793       | 272.072  | 3.5264   | -2.7207  | 4.621   | -419.164 | 464.328 | 4.4026  | 1045.48 | 5.8685  | -0.0651 | -244.789 | -1.92   |
| 75%      | 60794.5     | 528.346  | 5.4637   | 4.8532   | 5.727   | 264.144  | 597.517 | 5.8939  | 1058.33 | 8.2067  | 0.5756  | -232.528 | 52.625  |
| max      | 104956      | 1016.21  | 10.2968  | 9.6674   | 10.032  | 1092.39  | 745.396 | 11.8897 | 1094.46 | 13.5368 | 3.0565  | -225.428 | 279.07  |

## Data Visualization
Here I developed some data visualization tools to make insights and develop an intuition of what the data looks like:
- Histogram.py: generates the histogram of the features to see which Hogwarts course has a homogeneous score distribution between all four houses:
  
  ![histograms](https://github.com/shimazadeh/Ft_logistic_regression/assets/67879533/c7950c13-d595-4a22-ae7b-fc3c71415ec2)

- scatter_plot.py: displays a scatter plot of the features in the data that are similar and one can be eliminated:
  
  ![Figure_2](https://github.com/shimazadeh/Ft_logistic_regression/assets/67879533/9748a445-b3bf-4dd4-b258-43ba4a052e17)

- pair_plot.py: displays a pair plot matrix of the data to see what features can be eliminated and  what features can be used for the logistic regression model:
  
  ![Figure_3](https://github.com/shimazadeh/Ft_logistic_regression/assets/67879533/3582dfc1-66db-44cb-9b20-0be9c55203eb)

## Training and Evaluation


