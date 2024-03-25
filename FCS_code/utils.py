from os import error
import re
import shap # note it is neccessary to use the 0.44.0 version; the new realise 0.45.0 on 8th of March 2024 is substatially different (i.e. use pip install shap==0.44.0)
import pandas as pd
import numpy  as np
import sklearn.metrics
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.linear_model
import sklearn.pipeline
import sklearn.model_selection
from scipy.stats import norm
from imblearn.over_sampling import SMOTE


def load_micro_data(country = 'IT'):
    '''Loads micro data and drops columns with more than 200_000 NANs'''

    data = pd.read_pickle('Erasmus_data_stresstesting_2024.pickle')
    country_data = data[data['country_code']== country].reset_index()

    l = data.columns[data.isna().sum() < 200_000]
    df2 = country_data[l].dropna()

    return df2


def merge_macro_micro_it(country_data, file = 'ITALY_df.xlsx'):
    '''Merges Italy's micro and macro variables. Used for first results. '''

    macro = pd.read_excel(file) 

    macro = macro.iloc[:-2, :-11]

    macro = macro.rename(columns = {'year': 'status_year'}) 
    full_data = pd.merge(country_data,macro, on = 'status_year', how = 'inner') 

    return full_data


def calculate_woe(df, feature, target):
    '''Caluculates weight of evidence for a given feature.'''

    total_positive = df[target].sum()
    total_negative = df.shape[0] - total_positive
    
    woe_l = []
    for category in df[feature].unique():
        subset = df[df[feature] == category]
        positive_instances = subset[target].sum()
        negative_instances = subset.shape[0] - positive_instances
        
        if positive_instances == 0:
            positive_instances = 0.5
        if negative_instances == 0:
            negative_instances = 0.5
        
        woe = np.log((positive_instances / total_positive) / (negative_instances / total_negative))
        
        woe_l.append({feature: category, 'WoE_industry': woe})
    
    df_woe = pd.DataFrame(woe_l)

    return df_woe

def get_train_test(X, y, use_SMOTE = True):
    '''Creates train test split and optionally apply SMOTE'''

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2,  random_state=42, stratify=y)
    
    if use_SMOTE:
        smote = SMOTE(sampling_strategy = 0.3, k_neighbors = 5, random_state=42) # type: ignore
        X_train, y_train = smote.fit_resample(X_train, y_train)      # type: ignore

    return [X_train, y_train, X_test, y_test]


def get_f_importance(model, data):
    '''Gets feature importance from Random Forest or XGBOOST'''

    print(f'The most important features for {model.steps[-1][0]} are:')
    f_imp = []
    if len(model) == 3:
        importances = model.steps[2][1].feature_importances_
        indices = np.argsort(importances)

        for i in range(1, 6):
            vars = model.steps[1][1].get_feature_names_out()[indices[i]]
            numbers = re.findall(r'\d+', vars)
            numb = [int(num) for num in numbers]
            if len(numb) == 2:
                var1 = data[0].iloc[:, numb[0]].name
                var2 = data[0].iloc[:, numb[1]].name
                imp = round(importances[indices[-i]], 3)

                feature = f'{var1} * {var2}'
                f_imp.append(feature)
                print(f'{feature} : {imp}')

            else:
                feature =  f'{data[0].iloc[:, numb[0]].name}^2'
                imp = round(importances[indices[-i]], 3)

                f_imp.append(feature)
                print(f'{feature} : {imp}')
            
    else:
        f_imp = []
        importances = model.steps[1][1].feature_importances_
        indices = np.argsort(importances)

        for i in range(1, 6):
            imp = round(importances[indices[-i]], 3)
            feature = data[0].columns[indices[-i]]

            print(f'{feature} : {imp}')

def load_main_df():
    '''Loads the final dataset for PD modelling, Stress Testings...'''

    df = pd.read_csv('imputed_MICE_forrest.csv').iloc[:, 1:]

    data = pd.read_pickle('Erasmus_data_stresstesting_2024.pickle').iloc[:, 2:20]

    countries_of_interest = ['FI', 'NL', 'AT', 'BE', 'SE', 'DE', 'NO', 'DK', 'IS', 'IT', 'ES', 'PT', 'LV', 'RO', 'HR', 'LT', 'BG', 'SK', 'CZ', 'SI', 'HU', 'PL']

    subset_df = data[data['country_code'].isin(countries_of_interest)]
    info = subset_df.iloc[:, :7]
    info = info.reset_index().iloc[:, 1:]
    df = pd.concat([info, df], axis = 1)

    df2 = pd.read_csv('full_data_woe_rid_-2.csv').iloc[:, 2:]
    macro = df2.iloc[:, 61:]
    macro = pd.concat([df2[['country_code', 'status_year']], macro], axis = 1)

    woe = df2[['country_code', 'industry_code', 'WoE_country', 'WoE_industry']]

    macro_uni = macro.drop_duplicates()
    woe_uni = woe.drop_duplicates()

    df_new = pd.merge(df, woe_uni, on = ['country_code', 'industry_code'])
    df_new = pd.merge(df_new, macro_uni, on = ['country_code', 'status_year'])

    return df_new


def load_statistical_clusters(df):
    '''Create the clusters based on FCS_code2.5_Clustering.ipynb (done manually so we dont have to run the clustering repatedly)'''

    if isinstance(df, pd.DataFrame):
        pass
    else:
        df = load_main_df()

    country1 = df[df['country_code'].isin(['IT', 'PT', 'ES'])]
    country2 = df[df['country_code'].isin(['PL', 'CZ', 'RO','HU','BG','HR','LT','LV','BE','SK','IS'])]
    country3 = df[df['country_code'].isin(['NL', 'SE', 'NO','DK','AT','DE','FI','SI'])]

    Cluster1 = country1[country1['industry_code'].isin(['G', 'C', 'M','J','GX','R','S','K'])]
    Cluster2 = country1[country1['industry_code'].isin(['I', 'Q', 'A','E','D','P','B'])]
    Cluster3 = country1[country1['industry_code'].isin(['F', 'H', 'N','L'])]

    Cluster4 = country2[country2['industry_code'].isin(['F', 'H', 'N', 'L'])]
    Cluster5 = country2[country2['industry_code'].isin(['M', 'J', 'E','D','S','K'])]
    Cluster6 = country2[country2['industry_code'].isin(['A', 'Q', 'I','B','P'])]
    Cluster7 = country2[country2['industry_code'].isin(['G', 'C', 'GX','R'])]

    Cluster8 = country3[country3['industry_code'].isin(['G', 'C', 'M','J','GX','R','S','K'])]
    Cluster9 = country3[country3['industry_code'].isin(['I', 'Q', 'A','E','D','P','B'])]
    Cluster10 = country3[country3['industry_code'].isin(['F', 'H', 'N','L'])]

    clusters = [Cluster1, Cluster2, Cluster3, Cluster4, Cluster5, Cluster6, Cluster7, Cluster8, Cluster9, Cluster10]

    return clusters




def run_model(data, type = 'RF', n_estimators = 50, max_depth = 3, interactions = False, SHAP = False, feat_imp = False, verbose = True, version = 2):
    '''Runs a model based on a given specification'''
    
    if type == 'xgboost':
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    elif type == 'RF':
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    elif type == 'logit':
        model = sklearn.linear_model.LogisticRegression(random_state = 42)
    elif type == 'hist':
        model = sklearn.ensemble.HistGradientBoostingClassifier(max_depth = max_depth)

    if interactions:
        polynomial = sklearn.preprocessing.PolynomialFeatures(2)
        transformer = sklearn.preprocessing.MinMaxScaler()
        pipeline = sklearn.pipeline.Pipeline(
            [('transformet', transformer), ('poly', polynomial), (f'{type}', model)])
    else:
        transformer = sklearn.preprocessing.StandardScaler()
        pipeline = sklearn.pipeline.Pipeline(
            [('transformet', transformer), (f'{type}', model)])

    model = pipeline.fit(data[0], data[1])

    

    if SHAP:
        explainer = shap.TreeExplainer(model.steps[1][1])
        shap_values = explainer.shap_values(data[2])
        shap.summary_plot(shap_values, data[2])

    if verbose:
        predictions = model.predict(data[2])
        cm = sklearn.metrics.confusion_matrix(data[3], predictions)
        if version == 1:
            auc = sklearn.metrics.roc_auc_score(data[3], predictions)
        else:
            auc = sklearn.metrics.roc_auc_score(data[3], model.predict_proba(data[2])[:, 1])
        
        print(f'Results for {type}')
        print(30*'-')
        print(f'Confussion Matrx:\n {cm}')
        print(f'\nAUC: {round(auc, 4)}')  
    if feat_imp:
        print('\nFeature Importance:\n')
        get_f_importance(model, data)
    if verbose: 
        print(30*'=')

    return model

def stress_test(m, test, X, y):
    test_23 = test[test['status_year'] == 2023]
    probs = m.steps[1][1].predict_proba(test_23.iloc[:, 6:])
    predictions = np.where(probs[:, 1] < 0.5, 0, 1)
    preds_23 = predictions.sum()

    test_24 = test[test['status_year'] == 2024][predictions==0]
    test_25 = test[test['status_year'] == 2025][predictions==0]
    probs = m.steps[1][1].predict_proba(test_24.iloc[:, 6:])
    predictions = np.where(probs[:, 1] < 0.5, 0, 1)
    preds_24 = predictions.sum()

    test_25 = test_25[predictions==0]
    probs = m.steps[1][1].predict_proba(test_25.iloc[:, 6:])
    predictions = np.where(probs[:, 1] < 0.5, 0, 1)
    preds_25 = predictions.sum()

    print(f'Total default rate over 3 years: {round(  (preds_23+preds_24+preds_25)/len(test[test["status_year"] == 2023])*100, 3)}\nDefault Rate in the train data: {sum(y)/X.shape[0]}')
    print(f'Stress Test increased PD by multiple of {(preds_23+preds_24+preds_25)/len(test[test["status_year"] == 2023])/(sum(y)/X.shape[0])}')
    print(f'Number of defaults:\nYear 2023 : {preds_23}\nYear 2024 : {preds_24}\nYear 2025 : {preds_25}')


def RWA(PD, LGD, K):
    term1 = LGD * norm.cdf((norm.ppf(PD) / np.sqrt(1 - (0.12 * ((1 - np.exp(-50 * PD)) / (1 - np.exp(-50))) + 0.24 * (1 - (1 - np.exp(-50 * PD)) / (1 - np.exp(-50))))))  \
    + np.sqrt(((0.12 * ((1 - np.exp(-50 * PD)) / (1 - np.exp(-50))) + 0.24 * (1 - (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)))) / (1 - (0.12 * ((1 - np.exp(-50 * PD)) \
    / (1 - np.exp(-50))) + 0.24 * (1 - (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)))))) * norm.ppf(0.999)) - PD * LGD)
    return term1*12.5*K



