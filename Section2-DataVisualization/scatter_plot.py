import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Scatter_plot():
    def __init__(self, _features):
        self.features = _features

    def generate_scatters(self, input_file, output_file):
        data_raw = pd.read_csv(input_file)
        data = data_raw.dropna()
        labels=["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]

        for i in range(len(self.features) - 1):
            for j in range(i + 1, len(self.features)):
                if (abs(data[self.features[i]].corr(data[self.features[j]])) >= 0.95):
                    print("here")
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(data=data, x=self.features[i], y=self.features[j], hue="Hogwarts House", palette="Set1")
                    plt.xlabel(self.features[i])
                    plt.ylabel(self.features[j])
                    plt.legend()
                    plt.title(self.features[i] + " vs " + self.features[j])
                    plt.savefig(output_file + f"_{self.features[i]}_vs_{self.features[j]}.png")
                    plt.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)

    features = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination"
    ,"Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures"
    ,"Charms","Flying"]
    scatters = Scatter_plot(features)
    scatters.generate_scatters(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
