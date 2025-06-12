import pandas as pd
import matplotlib.pyplot as plt
import os

RAW_DATA_PATH = "../data/raw/flood_data.csv"
VIS_DIR = "../reports/figures"

def visualize_dataset():
    os.makedirs(VIS_DIR, exist_ok=True)
    df = pd.read_csv(RAW_DATA_PATH)

    # Create plots
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Histogram of FloodProbability
    df['FloodProbability'].plot(kind='hist', ax=ax[0], bins=30, color='skyblue', edgecolor='black')
    ax[0].set_title('Flood Probability Distribution')
    ax[0].set_xlabel('Flood Probability')
    ax[0].set_ylabel('Frequency')

    # Average FloodProbability by one feature (e.g., Urbanization)
    mean_flood_by_urbanization = df.groupby('Urbanization')['FloodProbability'].mean()
    mean_flood_by_urbanization.plot(kind='bar', ax=ax[1], color='orange')
    ax[1].set_title('Average Flood Probability by Urbanization Level')
    ax[1].set_xlabel('Urbanization Level')
    ax[1].set_ylabel('Average Flood Probability')

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'data_visualization.png'))
    print("Visualization saved to", VIS_DIR)

if __name__ == "__main__":
    visualize_dataset()
    print("Data visualization complete.")