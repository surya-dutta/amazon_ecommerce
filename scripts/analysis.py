# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration for plotting
sns.set(style="whitegrid")

# Load data
data = pd.read_csv('/path/to/transformed_output.csv')

# Display the first few rows of the dataframe
print("First few rows of the dataframe:")
print(data.head())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Histogram of a specific feature
def plot_histogram(feature):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True, color='blue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Boxplot for another feature
def plot_boxplot(feature):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[feature])
    plt.title(f'Box Plot of {feature}')
    plt.xlabel(feature)
    plt.show()

# Function calls for plotting
if __name__ == "__main__":
    plot_histogram('categories')  
    plot_boxplot('sentiment')  
