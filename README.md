# DataScience | Logistic Regression | 42Paris
Implement one-vs-all logistic regression that will solve classification problem: 
- Implementation of pandas.DataFrame.describe from scratch
- Implementation of data visulazionation tools from scratch to make insights and develop an intuition of what the data looks like
- Recreated Poudlard's Sorting Hat by implementing logistic regression from scratch.

## Requirements:
- Python 3
- NumPy
- Pandas
- Matplotlib
- Sklearn
- Tabulate
- Scipy

## How to Run:
<pre><code>
  git clone https://github.com/shimazadeh/Ft_logistic_regression.git DSLR
  cd DSLR
  pip3 install -r requirements.txt</code></pre>
  python main.py config.yaml: config.yaml file must include necessary information for training and testing purposes

## Implementation
The following sections indicates the method and results for each part of the program, note all the methods are developed from scratch:

### Data Analysis
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
Three programs that implementation of histogram, scatter plot and pair-plot library in python:

| Histogram.py                                  | scatter_plot.py                               |
|-----------------------------------------------|-----------------------------------------------|
| Generates the histogram of the features to see the homogeneous score distribution between all four houses. | Displays a scatter plot of similar features to identify those that can be eliminated. |
| ![Histogram Screenshot](<Screen Shot 2023-11-15 at 6.40.41 PM.png>) | ![Scatter Plot Screenshot](<Section2-DataVisualization/_Astronomy_vs_Defense Against the Dark Arts.png>) |

| pair_plot.py                                                                                       |
|----------------------------------------------------------------------------------------------------|
| Displays a pair plot matrix of the data to identify features for the logistic regression model.  |
| ![Pair Plot Screenshot](https://github.com/shimazadeh/Ft_logistic_regression/assets/67879533/216e4d59-4d86-4aa2-87a3-cdbe3c3e80a7) |



## Training and Evaluation
The program is modular and can be run with different settings. Adjust the config.yml file with your speicfic parameters and feeatures. The program can be run in two different mode: training and testing:
- Training: you must provide models parameters, the dataset and features to do the trainings in the yml file
- Testing: this mode of the program uses the model.joblib file generated from the training phase and outputs the result in a json file. 

During training the loss of each category is printed in the terminal for each iteration. At the end of the training a confusion matrix with performance of each category is also generated in the terminal.

![Alt text](<Screen Shot 2023-11-15 at 6.56.11 PM.png>)



| Stochastic GD                                                     | Mini-Batch GD                                                     | GD                                                                |
|-------------------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|
|![Alt text](<Section3-Model and Training/Loss_training_SGD.png>)|![Alt text](<Section3-Model and Training/Loss_training_MiniGD.png>)|![Alt text](<Section3-Model and Training/Loss_training_GD.png>)|

