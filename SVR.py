import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


data = pd.read_csv('final_file.csv')
# Select features and target
X = data[['Log GDP per capita', 'The Global Cyber Security Index', 'unemployment']]
y = data['Homicide_Rate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVR model
svr_model = SVR(kernel='linear')
svr_model.fit(X_train_scaled, y_train)

# Evaluate feature importance using permutation importance
perm_importance = permutation_importance(svr_model, X_test_scaled, y_test, scoring='r2', random_state=42)

# Extract feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)
print(feature_importance)