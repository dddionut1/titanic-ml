from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import pandas as pd

# datele
titanic = sns.load_dataset("titanic")


# curatare
coloane = ["survived", "pclass", "sex", "age", "fare", "sibsp", "parch"]
df = titanic[coloane].copy()

#  Completăm vârstele lipsă cu mediana
df["age"] = df["age"].fillna(df["age"].median())

#  Transformăm sex din text în număr
df["sex"] = df["sex"].map({"male": 0, "female": 1})

#  X și y
X = df[["pclass", "sex", "age", "fare", "sibsp", "parch"]]
y = df["survived"]

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  MODEL
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictii = model.predict(X_test)

# EVALUARE
print(f"Acuratete: {accuracy_score(y_test, predictii) * 100:.1f}%")
print(classification_report(y_test, predictii))

importanta = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("\nCe conteaza cel mai mult:")
print(importanta)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

modele = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN":           KNeighborsClassifier(n_neighbors=5)
}

print("\n=== Comparatie modele pe Titanic ===")
for nume, m in modele.items():
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    print(f"{nume:20} → {acc * 100:.1f}%")

#regresia logistica
model_lr = LogisticRegression(random_state=42, max_iter=1000)
model_lr.fit(X_train, y_train)
predictii_lr = model_lr.predict(X_test)

print(f"Logistic Regression: {accuracy_score(y_test, predictii_lr) * 100:.1f}%")

# Probabilitati — ce face LR diferit de celelalte modele
probabilitati = model_lr.predict_proba(X_test)
print("\nPrimii 5 pasageri — probabilitate supravietuire:")
for i in range(5):
    print(f"Pasager {i+1}: {probabilitati[i][1] * 100:.1f}% sanse")

    # Scalezi datele(Feature scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN cu scaling
model_knn_scaled = KNeighborsClassifier(n_neighbors=5)
model_knn_scaled.fit(X_train_scaled, y_train)
pred_scaled = model_knn_scaled.predict(X_test_scaled)

print(f"KNN fără scaling: {accuracy_score(y_test, KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train).predict(X_test)) * 100:.1f}%")
print(f"KNN cu scaling:   {accuracy_score(y_test, pred_scaled) * 100:.1f}%")

