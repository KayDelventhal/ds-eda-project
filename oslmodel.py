import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.graphics.regressionplots import plot_partregress_grid

from statsmodels.tools.eval_measures import mse, rmse
from statsmodels.iolib.smpickle import load_pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import scipy.stats as stats

import warnings
warnings.filterwarnings('ignore')

def main():
    '''main function of my EDA with LRM'''
    LOG = False

    file = 'us_bank_wages/us_bank_wages_data.csv'
    data = read_csv(file)

    if 'Unnamed: 0' in data.columns.to_list():
        data.drop('Unnamed: 0', axis=1, inplace=True)

    # de-skewing - explaination: the SALERY colmuns are right skewed and to compensate for that I use log()
    if LOG:
        data.eval('LSALARY = log(SALARY)', inplace = True)
        data.eval('LSALBEGIN = log(SALBEGIN)', inplace = True)

    # spliting data into train and test data
    train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True) 

    save_csv(data,'work_data/us_bank_wages_data.csv')
    save_csv(train,'work_data/us_bank_wages_test.csv')
    save_csv(test,'work_data/us_bank_wages_test.csv')

    if LOG:
        gcc = gen_column_combos(['SALBEGIN','LSALBEGIN','GENDER','C(MINORITY)','C(JOBCAT)','C(EDUC)'])
    else:
        gcc = gen_column_combos(['SALBEGIN','GENDER','C(MINORITY)','C(JOBCAT)','C(EDUC)'])

    model_result = {}
    for combo in gcc:
        if LOG:
            formula = 'LSALARY ~ ' + ' + '.join(combo)
        else:
            formula = 'SALARY ~ ' + ' + '.join(combo)

        model = train_model(data,formula)
        model = model_fit(model)

        rsquared_adj = model.rsquared_adj

        Y = train['SALARY'].astype(float)
        model_actual, model_predict = model_predict(data,Y)
        RMSE = model_rmse(model_actual, model_predict)
        
        model_result[rsquared_adj] = [RMSE,formula]

    # list model fittings - top 10
    model_fit_result_list = sorted(model_fit_result.keys())[-11:]
    for fit in model_fit_result_list:
        print('rsquared_adj:', fit, '\t<-', model_fit_result[fit])


def gen_column_combos(columns: list=['SALBEGIN','LSALBEGIN','GENDER','C(MINORITY)','C(JOBCAT)','C(EDUC)']):
    '''funcction to generate all combination for columns'''
    gcc = []
    for i in range(len(columns)):
        # remove first
        cols = columns[i+1:]
        # add first to end
        cols += columns[:i+1]
        
        for i in range(len(cols)):
            if not [cols[i]] in gcc:
                gcc.append([cols[i]])
                
            c = cols.copy()
            c.remove(cols[i])
            while len(c):
                if not sorted([x for x in c]) in gcc:
                    gcc.append(sorted([x for x in c]))
                c.pop()

    return gcc

def train_model(data: pd.DataFrame, formula='SALARY ~ C(EDUC) + C(JOBCAT) + SALBEGIN'):
    '''function to train a L.R.M. (OSL model'''

    return smf.ols(formula=formula, data=data)

def model_fit(model):
    '''function to fit() a trained model'''

    return model.fit()

def model_rsquared_adj(model):
    '''function to fit() a trained model'''

    return model.rsquared_adj

def model_rsquared(model):
    '''function to fit() a trained model'''

    return model.rsquared


def model_predict(data: pd.DataFrame, y_col: str='SALARY'):
    '''function to compute the prediction of the osl-model'''
    model_predict = model_fit.predict(data);
    model_actual = data[y_col].astype(float);

    return model_actual, model_predict


def model_error(model_actual, model_predict):
    '''print common error parameter'''
    print("Mean Absolute Error (MAE)        : {}".format(
        mean_absolute_error(model_actual, model_predict)))
    print("Mean Squared Error (MSE)         : {}".format(
        mse(model_actual, model_predict)))
    print("Root Mean Squared Error (RMSE)   : {}".format(
        rmse(model_actual, model_predict)))
    print("Mean Absolute Perc. Error (MAPE) : {}".format(
        np.mean(np.abs((model_actual - model_predict) / model_actual)) * 100))


def model_rmse(model_actual, model_predict):
    ''' calculate the RMSE'''
    return rmse(model_actual, model_predict)


def save_model(model_fit, file: str='default.pickle'):
    '''function to save the osl-model'''
    model_fit.save(file)


def read_model(file: str='default.csv'):
    '''function to load the osl-model'''
    model_fit = load_pickle(file)

    return model_fit


def save_csv(data: pd.DataFrame, file: str='default.csv'):
    '''function to save a pandas DataFrame into a CSV file'''
    data.to_csv(file)


def read_csv(file: str='default.csv'):
    '''function to load a pandas DataFrame into a CSV file'''
    
    return pd.read_csv(file)


if __name__ == "__main__":

    main()

    pass

# EOF