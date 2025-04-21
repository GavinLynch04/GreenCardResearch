from Data.Preprocessing.preprocess import preprocess
import seaborn as sns
import matplotlib.pyplot as plt

X, y = preprocess()

def KDE():
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=y, fill=True)
    plt.title('Distribution of Wait Times (Smoothed)')
    plt.xlabel('Wait Time (years)')
    plt.ylabel('Density')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("./Figures/KDEplot.svg", format="svg")
    plt.show()

def pieChartTopCountries():
    plt.figure(figsize=(10, 6))
    print(X.columns)

if __name__ == "__main__":
    pieChartTopCountries()
    KDE()