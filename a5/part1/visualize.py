import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def load_results(path):
    """Load and prepare results from CSV file"""
    df = pd.read_csv(path)
    df.columns = ['prediction', 'label']
    df['prediction'] = df['prediction'].astype(int)
    df['label'] = df['label'].astype(int)
    return df

def plot_confusion(df, title):
    """Plot confusion matrix with accuracy score"""
    cm = confusion_matrix(df['label'], df['prediction'])
    acc = accuracy_score(df['label'], df['prediction'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{title} - Confusion Matrix\nAccuracy: {acc:.2f}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

def main():
    # Load results from both models
    svm_df = load_results("svm.csv")
    mlp_df = load_results("mlp.csv")

    # Print classification reports
    print("=== Linear SVM (One-vs-Rest) Classification Report ===")
    print(classification_report(svm_df['label'], svm_df['prediction']))

    print("\n=== Multilayer Perceptron (MLP) Classification Report ===")
    print(classification_report(mlp_df['label'], mlp_df['prediction']))

    # Plot confusion matrices
    plot_confusion(svm_df, "Linear SVM (One-vs-Rest)")
    plot_confusion(mlp_df, "Multilayer Perceptron")

    # Compare accuracies
    svm_acc = accuracy_score(svm_df['label'], svm_df['prediction'])
    mlp_acc = accuracy_score(mlp_df['label'], mlp_df['prediction'])

    # Plot accuracy comparison
    plt.figure(figsize=(8, 5))
    plt.bar(['Linear SVM', 'MLP'], [svm_acc, mlp_acc])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate([svm_acc, mlp_acc]):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
