import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PairPlot:
    def __init__(self, _features):
        self.features = _features

    def generate_pairplot(self, input_file, output_file):
        data_raw = pd.read_csv(input_file)
        data = data_raw.dropna()

        num_features = len(self.features)
        fig, axs = plt.subplots(num_features, num_features, figsize=(12, 12))
        categories = data['Hogwarts House'].unique()
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        plt.subplots_adjust(hspace=0.5, wspace=0.5) 

        for i in range(num_features):
            for j in range(num_features):
                ax = axs[i, j]
                if i == j:
                    for k, category in enumerate(categories):
                        category_data = data[data['Hogwarts House'] == category]
                        ax.hist(category_data[self.features[i]], bins=10, alpha=0.5, color=colors[k], label=category)
                else:
                    for k, category in enumerate(categories):
                        category_data = data[data['Hogwarts House'] == category]
                        ax.scatter(category_data[self.features[i]], category_data[self.features[j]], s=2, color=colors[k], label=category)
                
                if i != num_features:
                    ax.set_xticks([])
                    ax.set_xlabel('')
                if j != 0:
                    ax.set_yticks([])
                    ax.set_ylabel('')
                if i == num_features - 1:
                    ax.set_xlabel(self.features[j])
                    ax.xaxis.label.set_rotation(45)

                if j == 0:
                    ax.set_ylabel(self.features[i])
                    ax.yaxis.label.set_rotation(45)

        plt.savefig(output_file)
        plt.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)

    features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination",
                "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions",
                "Care of Magical Creatures", "Charms", "Flying"]
    
    scatters = PairPlot(features)
    scatters.generate_pairplot(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
