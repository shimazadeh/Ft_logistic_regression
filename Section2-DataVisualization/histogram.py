import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Histogram:
    def __init__(self, _features):
        self.features = _features

    def generate_histograms(self, _filename, output_file):
        data_raw = pd.read_csv(_filename)
        data = data_raw.dropna()

        categories = data['Hogwarts House'].unique()
        colors = ['red', 'green', 'blue', 'yellow']
        
        # Create a single figure to hold all the subplots
        fig, axs = plt.subplots(len(self.features), figsize=(11, 7 * len(self.features)))

        for idx, feature in enumerate(self.features):
            for i, category in enumerate(categories):
                category_data = data[data['Hogwarts House'] == category]
                ax = axs[idx] if len(self.features) > 1 else axs
                ax.hist(category_data[feature], bins=10, alpha=0.5, color=colors[i], label=category)

            axs[idx].set_title("Histogram of " + feature + " by Hogwarts House")
            axs[idx].set_xlabel(feature)
            axs[idx].set_ylabel("Frequency")
            axs[idx].legend(title="Hogwarts House")

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)

    features = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination"
    ,"Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures"
    ,"Charms","Flying"]
    histogram = Histogram(features)
    histogram.generate_histograms(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
