import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

# Loading data
file_path = r"C:\Users\Nandita Singh\OneDrive\Desktop\Python Files\housing.xlsx"
housing = pd.read_excel(file_path)

# Creating categories
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Removing temporary income_cat column
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Separation
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Cleaning data by handling missing values
housing_num = housing.drop("ocean_proximity", axis=1)

# SimpleImputer
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(housing_num)

# DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

# Correlations
corr_matrix = strat_train_set.corr(numeric_only=True)
print("Top Correlations with House Value:")
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Visulaisation
strat_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                     s=strat_train_set["population"]/100, label="population", figsize=(10,7),
                     c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()
