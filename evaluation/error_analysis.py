import pandas as pd

def get_misclassified_examples(X_test, y_test, y_pred):

    df = pd.DataFrame({'text': X_test, 'true_label': y_test, 'predicted_label': y_pred})
    misclassified = df[df['true_label'] != df['predicted_label']]
    return misclassified

def analyze_errors(misclassified_df, num_examples=5):

    print(f"Total misclassified examples: {len(misclassified_df)}")
    print("\nSample Misclassifications:")
    for i, row in misclassified_df.head(num_examples).iterrows():
        print(f"Text: {row['text']}")
        print(f"True: {row['true_label']}, Pred: {row['predicted_label']}")
        print("-" * 50)
