
# coding: utf-8
import sys
import pandas as pd
import numpy as np
import scipy.stats as ss

from sklearn.cross_validation import train_test_split

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition.pca import PCA

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import mean_squared_error

import statsmodels.formula.api as smf

from statsmodels.sandbox.stats.multicomp import multipletests

print 'Reading data dictionary file...'
dd = pd.read_csv('CollegeScorecardDataDictionary-09-12-2015.csv', header=0)

print 'Finding columns corresponding to type earnings and repayment...'
earning_cols = dd[dd['dev-category'] == 'earnings']['VARIABLE NAME']
repayment_cols = dd[dd['dev-category'] == 'repayment']['VARIABLE NAME']
print 'Found {0} and {1} columns for earnings and repayment respectively'.format(len(earning_cols), len(repayment_cols))


print 'Finding categorical variables...'
previous_var = None
cat_vars = {'STABBR'}
for index, row in dd.iterrows():
    if (type(row['NAME OF DATA ELEMENT']) == float) and np.isnan(row['NAME OF DATA ELEMENT']):
        # if we encounter a nan name row, then the previous variable is a categorical variable
        cat_vars.add(previous_var)
    else:
        previous_var = row['VARIABLE NAME']

print 'Found {0} categorical variables'.format(len(cat_vars))


print 'Reading the college data for year 2011...'
df = pd.read_csv('MERGED2011_PP.csv', header=0)

print 'Shape of college data:', df.shape

print 'Removing identifier/fine grain columns like OPEID...'

bad_threshold = 1 / 4.0
print 'Removing columns with fraction of bad count > ', bad_threshold

removed_cols = ['\xef\xbb\xbfUNITID', 'OPEID','opeid6', 'ZIP', 'INSTNM', 'CITY', 'sch_deg', 'st_fips']

total_len = df.shape[0]

for col in df.columns:
    # bad count is either a null/nan or a PrivacySuppressed value
    bad_count = sum(df[col].isnull())
    if df[col].dtype == 'object':
        bad_count += df.loc[df[col] == 'PrivacySuppressed'].shape[0]
        
    if bad_count > total_len * bad_threshold:
        removed_cols.append(col)
    
print 'Removed {0} columns with bad data > {1}'.format(len(removed_cols), bad_threshold)


removed_cols_set = set(removed_cols)
removed_cols_set = removed_cols_set.union(list(earning_cols.values))
removed_cols_set = removed_cols_set.union(list(repayment_cols.values))

# include the target variables also in the df
#cols_to_include = {'ADM_RATE', 'mn_earn_wne_p10','md_earn_wne_p10'}
cols_to_include = {'mn_earn_wne_p10','md_earn_wne_p10'}
removed_cols_set = removed_cols_set - cols_to_include

print 'Totally dropping {0} columns'.format(len(removed_cols_set))

removed_cols = list(removed_cols_set)
df.drop(removed_cols, axis=1, inplace=True)

print 'New df shape:', df.shape


print 'Dropping bad rows...'

df.dropna(inplace=True)
print 'New df shape after dropping rows with nans:', df.shape


def is_privacy_surpressed(row):
    '''return True if the given row has a PrivacySuppressed value'''
    for col, value in row.iteritems():
        if value == 'PrivacySuppressed':
            return True
        
    return False
    
privacy_surpressed = df.apply(is_privacy_surpressed, axis=1)
df = df[~privacy_surpressed]

print 'New df shape after dropping rows with PrivacySuppressed values:', df.shape


print 'Converting all categorical variables into one-hot columns..'
for col in cat_vars.intersection(set(df.columns)):
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummies], axis=1)
    df.drop(col, inplace=True, axis=1)
    
print 'New df shape after dummyfying:', df.shape

# rename columns with . in name to _
# this is causing problems in OLS method later
rename_dict = {}
for col in df.columns:
    rename_dict[col] = col.replace('.', '_')

df.rename(columns=rename_dict, inplace=True)

# some columns get object type due to privacy suppressed values
# the categorical varaibles are also converted to one-hot values
# so force conversion of the entire df into numeric values since
# every field is now numeric
print 'Converting entire data frame into numeric values'
df_numeric = df.copy().convert_objects(convert_numeric=True)
df_numeric.drop(df_numeric.std()[df_numeric.std() == 0].index.values, axis=1, inplace=True)

df_y1 = df_numeric['mn_earn_wne_p10']
df_y2 = df_numeric['md_earn_wne_p10']

Y1 = df_y1.values
Y2 = df_y2.values

df_x = df_numeric.drop(['mn_earn_wne_p10', 'md_earn_wne_p10'], axis=1)

# DISCOVERY

print 'Starting Discovery phase...'

def forward_selection(df_x, df_y, k=None):
    '''Given X and y, choose the top k features using a greedy forward selection, based on the MSE on the training data'''
    if not k:
        k = len(df_x.columns)
    
    remaining = set(df_x.columns)
    selected = []

    while remaining and len(selected) < k:
        scores = []
        for candidate in remaining:
            X = df_x[selected + [candidate]]
            vanilla_lr = LinearRegression()
            vanilla_lr = vanilla_lr.fit(X, df_y)
            Y_pred = vanilla_lr.predict(X)
            score = mean_squared_error(df_y, Y_pred)
            scores.append((score, candidate))
        
        score, best_candidate = min(scores)
        #print len(selected), score, best_candidate
        
        remaining.remove(best_candidate)
        selected.append(best_candidate)
            
    return selected


k = 100
print 'Discovering the top {0} features using forward selection:'.format(k)
fw_selected = forward_selection(df_x, df_y2, k)

print 'The top {0} features using forward selection are:'.format(k)
for var in fw_selected:
    print var

print 'Building a linear regression model using the selected features for significance testing...'
formula = "{} ~ {} + 1".format('md_earn_wne_p10',' + '.join(fw_selected))
fw_model = smf.ols(formula, df_numeric).fit()

print 'Performing significance test on individual variables using benjamini hochberg correction...'
pvalues = fw_model.pvalues[1:]
pvalues.sort(axis=1)
reject, _, _, _  = multipletests(pvalues, method='fdr_bh')

significant_vars = []
for i in range(len(reject)):
    if reject[i]:
        significant_vars.append(pvalues.index[i])

print 'Found {0} significant variables out of {1} selected variables'.format(len(significant_vars), len(fw_selected))

print 'The significant variables are:'
for var in significant_vars:
    print var

print 'The coefficents of the top 5 significant vars are:'
for i in range(5):
    var = significant_vars[i]
    coeff = fw_model.params[var]
    print var, coeff

print 'The coefficents of the bottom 5 significant vars are:'
for i in range(5):
    var = significant_vars[-i - 1]
    coeff = fw_model.params[var]
    print var, coeff

# PREDICTION
print 'Starting Prediction phase...'

print 'Standardizing X...'
X = ss.zscore(df_x)
print X.shape

print 'Splitting data into 70 percent training and 30 percent testing...'
X_train, X_test, Y_train, Y_test = train_test_split(X, Y2, test_size=0.3, random_state=42)
df_xtrain, df_xtest, df_ytrain, df_ytest = train_test_split(df_x, df_y2, test_size=0.3, random_state=42)
print 'Shape of train X', X_train.shape
print 'Shape of test X', X_test.shape
print 'Shape of train y', Y_train.shape
print 'Shape of test y', Y_test.shape

print 'Generating error for baseline model: mean prdictor...'
baseline = DummyRegressor()
baseline.fit(X_train, Y_train)
Y_pred = baseline.predict(X_test)
print 'RMSE of the mean predictor model:', mean_squared_error(Y_test, Y_pred) ** 0.5

print 'Training a linear regression model...'
vanilla_lr = LinearRegression()
vanilla_lr = vanilla_lr.fit(X_train, Y_train)
Y_pred = vanilla_lr.predict(X_test)
print 'RMSE of the standardized linear regression model:', mean_squared_error(Y_test, Y_pred) ** 0.5


print 'Training a ridge regression model...'
print 'Using 3-fold cross validation on the training data to select best alpha...'
gs_params = {'alpha':[2**i for i in range(-10,20)]}
print 'Candidate alpha values to search for:', gs_params['alpha']

gc = GridSearchCV(estimator=Ridge(), param_grid=gs_params)
ridge_model = gc.fit(X_train, Y_train)
Y_pred = ridge_model.predict(X_test)
best_alpha = ridge_model.best_params_['alpha']
print 'The best ridge model is obtained wiht an alpha of', best_alpha
print 'RMSE of the best standardized ridge regression model:', mean_squared_error(Y_test, Y_pred) ** 0.5


print 'Training a lasso regression model...'
print 'Using 3-fold cross validation on the training data to select best alpha...'
gs_params = {'alpha':[2**i for i in range(-10,20)]}
print 'Candidate alpha values to search for:', gs_params['alpha']

gc = GridSearchCV(estimator=Lasso(), param_grid=gs_params)
lasso_model = gc.fit(X_train, Y_train)
Y_pred = lasso_model.predict(X_test)
best_alpha = lasso_model.best_params_['alpha']
print 'The best lasso model is obtained wiht an alpha of', best_alpha
print 'RMSE of the best standardized lasso regression model:', mean_squared_error(Y_test, Y_pred) ** 0.5


# split data into train, dev and test
# todo: do this before, so that we have errors measured on the same test set
for i in range(1, 9):
    n_components = 2 ** i
    #n_components = i
    pca = PCA(n_components=n_components)
    X_reduced_train = pca.fit_transform(X_train)
    X_reduced_test = pca.transform(X_test)

    vanilla_lr = LinearRegression()
    vanilla_lr = vanilla_lr.fit(X_reduced_train, Y_train)
    Y_pred = vanilla_lr.predict(X_reduced_test)
    print 'MSE for ', n_components, ' components with LR ', mean_squared_error(Y_test, Y_pred) ** 0.5

    gs_params = {'alpha':[2**i for i in range(-10,20)]}
    gc = GridSearchCV(estimator=Ridge(), param_grid=gs_params)
    ridge_model = gc.fit(X_reduced_train, Y_train)
    Y_pred = ridge_model.predict(X_reduced_test)
    print 'RMSE for ', n_components, ' components with Ridge ', mean_squared_error(Y_test, Y_pred) ** 0.5


