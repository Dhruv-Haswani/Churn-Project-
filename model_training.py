
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load the cleaned dataset
df = pd.read_csv('data\\processed\\customer_churn_cleaned.csv')

print("Dataset shape:", df.shape)

# 2. Select input features and target
features = [
    'CreditScore',
    'Geography',
    'Gender',
    'Age',
    'Tenure',
    'EstimatedSalary',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember'
]

X = df[features]
y = df['Exited']

# 3. Encode categorical features
X = pd.get_dummies(
    X,
    columns=['Geography', 'Gender'],
    drop_first=True,
    dtype=int
)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 5. Train Decision Tree
decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)

print("\nTraining Decision Tree...")
decision_tree.fit(X_train, y_train)

# 6. Tune and train Random Forest
random_forest = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=random_forest,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("\nTuning Random Forest...")
grid_search.fit(X_train, y_train)

best_random_forest = grid_search.best_estimator_

print("\nBest Random Forest parameters:")
print(grid_search.best_params_)

# 7. Evaluate both models
models = {
    'Decision Tree': decision_tree,
    'Tuned Random Forest': best_random_forest
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-score': round(f1, 4)
    })

    print(f"\n{'=' * 50}")
    print(name)
    print('=' * 50)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['Stayed (0)', 'Churned (1)'],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# 8. Display final comparison
comparison = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))


# 9. Generate visualizations
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
os.makedirs('outputs', exist_ok=True)

# Store predictions for the two final models
predictions = {
    name: model.predict(X_test)
    for name, model in models.items()
}

# ------------------------------------------
# A. Model Performance Comparison
# ------------------------------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']

comparison_plot = comparison.set_index('Model')[metrics] * 100

ax = comparison_plot.plot(
    kind='bar',
    figsize=(10, 6),
    width=0.75
)

ax.set_title('Decision Tree vs Tuned Random Forest')
ax.set_ylabel('Score (%)')
ax.set_xlabel('Model')
ax.set_ylim(0, 100)
ax.legend(title='Evaluation Metric')
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    'outputs/model_comparison.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()
plt.close()

# ------------------------------------------
# B. Confusion Matrices
# ------------------------------------------
for name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Stayed (0)', 'Churned (1)'],
        yticklabels=['Stayed (0)', 'Churned (1)']
    )

    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()

    filename = name.lower().replace(' ', '_')
    plt.savefig(
        f'outputs/{filename}_confusion_matrix.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()
    plt.close()

# ------------------------------------------
# C. Feature Importance
# ------------------------------------------
for name, model in models.items():
    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=True).tail(10)

    plt.figure(figsize=(9, 6))

    importance.plot(kind='barh')

    plt.title(f'{name} - Top 10 Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()

    filename = name.lower().replace(' ', '_')
    plt.savefig(
        f'outputs/{filename}_feature_importance.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()
    plt.close()

print("\nAll visualizations saved in the outputs folder.")
