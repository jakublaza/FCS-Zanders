from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

df = pd.read_csv('/storage/plzen1/home/jakublaza/FINAL.csv').iloc[:, 1:]

imputer = IterativeImputer(max_iter=10, random_state=42)

imputed_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

imputed_df.to_csv('/storage/plzen1/home/jakublaza/imputed_MICE_bayesian_ridge.csv')
