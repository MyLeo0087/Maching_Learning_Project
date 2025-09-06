# %% [markdown]
# # Import 

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")
 

# %% [markdown]
# # EDA, Data Cleaning and Data Preprocessing

# %%
df = pd.read_csv("salary.csv")
df.head()


df.describe()
df.info()


# %%
df2 = df[(df["Income"]<0) |( df["Expenses"]<0) | (df["Future_Income"]<0)]
df.drop(df2.index,inplace=True)


# %%
# for col in num_cloumn:
#     plt.figure(figsize=(3,2))
#     sns.histplot(x=df[col],kde=True)

# %%
# for col in cat_column:
#     plt.figure(figsize=(3,2))
#     sns.boxplot(x=df[col])

# %%
from sklearn.model_selection import train_test_split
X = df.drop(columns=["Future_Income"],axis=1)
y = df["Future_Income"]

target = "Future_Income"
num_columns = X.select_dtypes(include=["int64", "float64"]).columns.to_list()
cat_columns = X.select_dtypes(include=["object"]).columns.to_list()

# %%

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# %%
num_transformation = StandardScaler()
cat_transformation = OneHotEncoder(handle_unknown="ignore")

preprocess_st_on = ColumnTransformer(transformers=[
    ('num',num_transformation,num_columns),
    ('cat',cat_transformation,cat_columns)
])

preprocess_on = ColumnTransformer(transformers=[
    ('cat',cat_transformation,cat_columns)
])

# %% [markdown]
# # Simple Model 

# %% [markdown]
# ## Linear Regression

# %%
from sklearn.linear_model import LinearRegression
model_lr = Pipeline(steps=[
    ("preprocessing",preprocess_st_on),
    ("medel",LinearRegression())
])
model_lr.fit(X_train,y_train)

# %%
y_pred = model_lr.predict(X_test)
y_pred

# %%
y_test

# %% [markdown]
# ### R2,Adj R2 and MSE

# %%
from sklearn.metrics import r2_score,mean_absolute_error

r2 = r2_score(y_test,y_pred)
r2

# %%
n = X_test.shape[0]
p = X_test.shape[1]

adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
adj_r2

# %%

mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)


# %% [markdown]
# ## KNN
# 

# %%
from sklearn.neighbors import KNeighborsRegressor

model_knn = Pipeline(steps=[
    ('preprocessing',preprocess_st_on),
    ('model', KNeighborsRegressor(n_neighbors=17,leaf_size=10,weights="distance"))
])

model_knn.fit(X_train,y_train)

# %%
y_pred_knn = model_knn.predict(X_test)
y_pred_knn 

# %%
y_test

# %%
r2_knn =  r2_score(y_test,y_pred_knn)
r2_knn

# %% [markdown]
# 

# %%
adj_r2 = 1 - (1 - r2_knn) * ((n - 1) / (n - p - 1))
adj_r2

# %%
mae_knn = mean_absolute_error(y_test,y_pred_knn)
mae_knn

# %% [markdown]
# ## Decision Tree

# %%
from sklearn.tree import DecisionTreeRegressor

model_dt = Pipeline(steps=[
    ('preprocess',preprocess_st_on),
    ('model',DecisionTreeRegressor(max_depth=10,min_samples_leaf=8,min_samples_split=2))
])

model_dt.fit(X_train,y_train)

# %%
y_pred_dt = model_dt.predict(X_test)

# %%
r2_dt = r2_score(y_test,y_pred_dt)
r2_dt

# %%
adj_r2 = 1 - (1 - r2_dt) * ((n - 1) / (n - p - 1))
adj_r2

# %% [markdown]
# ## SVM

# %%
from sklearn.svm import SVR

model_svr = Pipeline(steps=[
    ('preprocess',preprocess_st_on),
    ('model',SVR(kernel="linear"))
])

model_svr.fit(X_train,y_train)

# %%
y_pred_svr = model_svr.predict(X_test)

# %%
r2_svr = r2_score(y_test,y_pred_svr)
r2_svr

# %% [markdown]
# # Cross validation

# %% [markdown]
# ## for linear regression

# %%
from sklearn.model_selection import cross_val_score

score_lr = cross_val_score(model_lr,X,y,cv=5,scoring='r2')

print(score_lr.mean())

# %% [markdown]
# ## for KNN

# %%
score_knn = cross_val_score(model_knn,X,y,cv=5,scoring="r2")
score_knn.mean()

# %% [markdown]
# # Hyperparameter Tuning

# %% [markdown]
# ## Grid Search CV

# %% [markdown]
# ### For KNN

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_neighbors': [3,5,7,9,11,13,17,25],
    'model__leaf_size': [10,20,30,50,60],
    'model__weights': ['uniform','distance']
}

grid = GridSearchCV(model_knn, param_grid, cv=5,scoring="r2")
grid.fit(X, y)

print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)

# %% [markdown]
# ### For Decision Tree 

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

param_grid = {
    'model__max_depth': [10, 15, 20, 25, 30],
    'model__min_samples_split': [2, 5, 10, 15],
    'model__min_samples_leaf': [1, 2, 4, 8]
}

grid = GridSearchCV(model_dt, param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best R² score:", grid.best_score_)



# %% [markdown]
# ## Random search cv

# %%
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'model__max_depth': [10, 15, 20, 25, 30],
    'model__min_samples_split': [2, 5, 10, 15],
    'model__min_samples_leaf': [1, 2, 4, 8]
}

random = RandomizedSearchCV(model_dt,param_grid,n_iter=5,cv=5,scoring='r2')

random.fit(X_train,y_train)


result = pd.DataFrame(random.cv_results_)
result[['param_model__min_samples_split','param_model__min_samples_leaf','param_model__max_depth','mean_test_score']]


# %% [markdown]
# # Stacking

# %%
from sklearn.ensemble import StackingRegressor

base_learner = [
    ('lr',model_lr),
    ('KNN',model_knn),
    ('DT',model_dt),
    ('SVR',model_svr)
]

Meta_learner = LinearRegression()

stacking_rg = StackingRegressor(
    estimators=base_learner,
    final_estimator=Meta_learner,
    cv=5
)

stacking_rg.fit(X_train,y_train)

y_pred_sr = stacking_rg.predict(X_test)

r2_sr = r2_score(y_test,y_pred_sr)

r2_sr

# %% [markdown]
# # Bagging

# %%
from sklearn.ensemble import RandomForestRegressor

model_rf = Pipeline(steps=[
    ('preprocess',preprocess_st_on),
    ('model',RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42
))
])

model_rf.fit(X_train,y_train)

y_pred_rf = model_rf.predict(X_test)

r2_rf = r2_score(y_test,y_pred_rf)
r2_rf

# %%
param_grid_rf = {
    'model__n_estimators': [100, 200, 500],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__bootstrap': [True, False]
}
model_rf_best = RandomizedSearchCV(model_rf,param_grid_rf,cv=5,scoring='r2',n_iter=50,n_jobs=-1)

model_rf_best.fit(X_train,y_train)

print(model_rf_best.best_params_)
print(model_rf_best.best_score_)


# %% [markdown]
# # Boosting

# %% [markdown]
# ## Ada boost

# %%
from sklearn.ensemble import AdaBoostRegressor

model_ada = Pipeline(steps=[
    ('preprocess',preprocess_st_on),
    ('model',AdaBoostRegressor(n_estimators=100,random_state=42))
])
model_ada.fit(X_train,y_train)

y_pred_ada = model_ada.predict(X_test)

r2_ada = r2_score(y_test,y_pred_ada)
r2_ada

# %% [markdown]
# ## Gradient boost

# %%
from sklearn.ensemble import GradientBoostingRegressor

model_GBR = Pipeline(steps=[
    ('preprocess',preprocess_st_on),
    ('model',GradientBoostingRegressor(n_estimators=100,random_state=42))
])
model_GBR.fit(X_train,y_train)

y_pred_GBR = model_GBR.predict(X_test)

r2_GBR = r2_score(y_test,y_pred_GBR)
r2_GBR

# %%
import xgboost as xgb

model_xgb = Pipeline(steps=[
    ('preprocess',preprocess_st_on),
    ('model',xgb.XGBRegressor(n_estimators=100,random_state=42))
])
model_xgb.fit(X_train,y_train)

y_pred_xgb = model_xgb.predict(X_test)

r2_xgb = r2_score(y_test,y_pred_xgb)
r2_xgb


