import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate
import matplotlib.pyplot as plt

df = pd.read_csv('final_file.csv')

# Selecting feature columns and target column
x = df[['Log GDP per capita', 'The Global Cyber Security Index', 'unemployment']]
y = df['Homicide_Rate']  # Target is continuous

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Defining regression models
models = [
    SVR(),
    DecisionTreeRegressor(),
    KNeighborsRegressor(),
    RandomForestRegressor(random_state=42)
]

model_names = []
mae_means = []
mae_stds = []

# Evaluating models using cross-validation
for model in models:
    cross_model = KFold(shuffle=True, random_state=24)
    estimator = Pipeline([
        ('model', model)
    ])
    # Mean Absolute Error (MAE) scoring for regression
    errors = -cross_val_score(
        estimator,
        X_train,
        Y_train,
        cv=cross_model,
        scoring="neg_mean_absolute_error",
        error_score='raise'
    )
    model_names.append(model.__class__.__name__)
    mae_means.append(errors.mean())
    mae_stds.append(errors.std())

# Creating a DataFrame to store results
results_df = pd.DataFrame({
    'Model': model_names,
    'Mean MAE': mae_means,
    'MAE Std': mae_stds
})

results_df = results_df.sort_values(by='Mean MAE')

# Printing results in tabular form
print(tabulate(results_df, headers='keys', tablefmt='pretty'))

# Save the results as a JPG file
plt.figure(figsize=(10, 5))
plt.barh(results_df['Model'], results_df['Mean MAE'], xerr=results_df['MAE Std'], capsize=5)
plt.xlabel('Mean Absolute Error')
plt.title('Model Performance Comparison (Regression)')
plt.tight_layout()
plt.show()
