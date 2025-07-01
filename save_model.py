import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample training data
X = np.array([
    [50, 1, 1.2, 180, 35, 45, 6.5, 3.0, 1.1],
    [45, 0, 0.9, 150, 30, 38, 6.8, 3.2, 1.2],
    [60, 1, 1.5, 200, 40, 50, 6.2, 2.8, 1.0]
])
y = [1, 0, 1]  # 1 = High Risk, 0 = Low Risk

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save both
with open('rf_acc_68.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('normalizer.pkl', 'wb') as f:
    pickle.dump(scaler, f)