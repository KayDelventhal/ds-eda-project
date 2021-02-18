import pandas as pd

import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import mse, rmse
from statsmodels.iolib.smpickle import load_pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

def single_line(formula: str='SALARY ~ C(EDUC) + C(JOBCAT) + C(MINORITY) + SALBEGIN'):
    '''main function of my EDA with LRM'''
    LOG = False

    data = pd.read_csv('us_bank_wages/us_bank_wages.txt', delimiter='\t')

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

    workdata = train.copy()

    result_rmse = {}
    result_radj = {}

    model = smf.ols(formula=formula, data=workdata)
    model_fit = model.fit()

    rsquared_adj = model_fit.rsquared_adj

    Y = workdata['SALARY'].astype(float)
    predict = model_fit.predict(workdata);
    actual = Y.astype(float);

    RMSE = rmse(actual, predict)
    
    result_radj[rsquared_adj] = [RMSE,formula]
    result_rmse[RMSE] = [rsquared_adj,formula]

    file = 'us_bank_wages/us_bank_wages_train_model_fit.pickle'
    print('\n--- save model:', file)
    model_fit.save(file)
    
    print('\n--- result_radj ---')
    model_fit_result_list = sorted(result_radj.keys())[-11:]
    for fit in model_fit_result_list:
        print('rsquared_adj:', fit, '\t<-', result_radj[fit])

    print('\n--- result_rmse ---')
    model_fit_result_list = sorted(result_rmse.keys())[:11]
    for fit in model_fit_result_list:
        print('rsquared_adj:', fit, '\t<-', result_rmse[fit])


def brute_force():
    '''main function of my EDA with LRM'''
    LOG = False

    data = pd.read_csv('us_bank_wages/us_bank_wages.txt', delimiter='\t')

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

    workdata = train.copy()

    if LOG:
        gcc = gen_column_combos(['SALBEGIN','LSALBEGIN','GENDER','C(MINORITY)','C(JOBCAT)','C(EDUC)'])
    else:
        gcc = gen_column_combos(['SALBEGIN','GENDER','C(MINORITY)','C(JOBCAT)','C(EDUC)'])

    result_rmse = {}
    result_radj = {}
    for combo in gcc:
        if LOG:
            formula = 'LSALARY ~ ' + ' + '.join(combo)
        else:
            formula = 'SALARY ~ ' + ' + '.join(combo)

        model = smf.ols(formula=formula, data=workdata)
        model_fit = model.fit()

        rsquared_adj = model_fit.rsquared_adj

        Y = workdata['SALARY'].astype(float)
        predict = model_fit.predict(workdata);
        actual = Y.astype(float);

        RMSE = rmse(actual, predict)
        
        result_radj[rsquared_adj] = [RMSE,formula]
        result_rmse[RMSE] = [rsquared_adj,formula]

    file = 'us_bank_wages/us_bank_wages_train_model_fit.pickle'
    print('\n--- save model:', file)
    model_fit.save(file)
    
    print('\n--- result_radj ---')
    model_fit_result_list = sorted(result_radj.keys())[-11:]
    for fit in model_fit_result_list:
        print('rsquared_adj:', fit, '\t<-', result_radj[fit])

    print('\n--- result_rmse ---')
    model_fit_result_list = sorted(result_rmse.keys())[:11]
    for fit in model_fit_result_list:
        print('rsquared_adj:', fit, '\t<-', result_rmse[fit])


def gen_osl_model(formula, workdata):    
    '''function to compute an osl() model'''
    model = smf.ols(formula=formula, data=workdata)
    model_fit = model.fit()

    rsquared_adj = model_fit.rsquared_adj

    Y = workdata['SALARY'].astype(float)
    predict = model_fit.predict(workdata);
    actual = Y.astype(float);

    RMSE = rmse(actual, predict)

    return model, model, rsquared_adj, RMSE, actual, predict


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


def save_csv(data: pd.DataFrame, file: str='default.csv'):
    '''function to save a pandas DataFrame into a CSV file'''
    data.to_csv(file)


if __name__ == "__main__":

    choice = input('1:brute-force; 2:enter-string')
    if '1' in choice:
        
        brute_force()

    elif '2' in choice:

        exp = input('SALARY ~ C(EDUC) + C(JOBCAT) + C(MINORITY) + SALBEGIN')

        single_line(exp)

    pass

# EOF

'''
--- result_radj ---
rsquared_adj: 0.8082399293083553        <- [7721.604831880341, 'SALARY ~ C(EDUC) + SALBEGIN']
rsquared_adj: 0.8084254256228924        <- [7707.375868347392, 'SALARY ~ C(EDUC) + C(MINORITY) + SALBEGIN']
rsquared_adj: 0.8099951231848639        <- [7675.735100135453, 'SALARY ~ C(EDUC) + GENDER + SALBEGIN']
rsquared_adj: 0.8107492238448184        <- [7650.044288763978, 'SALARY ~ C(EDUC) + C(MINORITY) + GENDER + SALBEGIN']
rsquared_adj: 0.8154961818637975        <- [7635.597940584766, 'SALARY ~ C(JOBCAT) + C(MINORITY) + SALBEGIN']
rsquared_adj: 0.8157848960520228        <- [7639.814675320462, 'SALARY ~ C(JOBCAT) + SALBEGIN']
rsquared_adj: 0.818790245414134         <- [7567.129377447405, 'SALARY ~ C(JOBCAT) + GENDER + SALBEGIN']
rsquared_adj: 0.8187965885289206        <- [7556.873858021025, 'SALARY ~ C(JOBCAT) + C(MINORITY) + GENDER + SALBEGIN']
rsquared_adj: 0.8255631713443903        <- [7334.492950941693, 'SALARY ~ C(EDUC) + C(JOBCAT) + C(MINORITY) + SALBEGIN']
rsquared_adj: 0.8256204234943031        <- [7343.3279507185025, 'SALARY ~ C(EDUC) + C(JOBCAT) + SALBEGIN']
rsquared_adj: 0.8262946570863963        <- [7319.098514513653, 'SALARY ~ C(EDUC) + C(JOBCAT) + GENDER + SALBEGIN']

--- result_rmse ---
rsquared_adj: 7319.098514513653         <- [0.8262946570863963, 'SALARY ~ C(EDUC) + C(JOBCAT) + GENDER + SALBEGIN']
rsquared_adj: 7334.492950941693         <- [0.8255631713443903, 'SALARY ~ C(EDUC) + C(JOBCAT) + C(MINORITY) + SALBEGIN']
rsquared_adj: 7343.3279507185025        <- [0.8256204234943031, 'SALARY ~ C(EDUC) + C(JOBCAT) + SALBEGIN']
rsquared_adj: 7556.873858021025         <- [0.8187965885289206, 'SALARY ~ C(JOBCAT) + C(MINORITY) + GENDER + SALBEGIN']
rsquared_adj: 7567.129377447405         <- [0.818790245414134, 'SALARY ~ C(JOBCAT) + GENDER + SALBEGIN']
rsquared_adj: 7635.597940584766         <- [0.8154961818637975, 'SALARY ~ C(JOBCAT) + C(MINORITY) + SALBEGIN']
rsquared_adj: 7639.814675320462         <- [0.8157848960520228, 'SALARY ~ C(JOBCAT) + SALBEGIN']
rsquared_adj: 7650.044288763978         <- [0.8107492238448184, 'SALARY ~ C(EDUC) + C(MINORITY) + GENDER + SALBEGIN']
rsquared_adj: 7675.735100135453         <- [0.8099951231848639, 'SALARY ~ C(EDUC) + GENDER + SALBEGIN']
rsquared_adj: 7707.375868347392         <- [0.8084254256228924, 'SALARY ~ C(EDUC) + C(MINORITY) + SALBEGIN']
rsquared_adj: 7721.604831880341         <- [0.8082399293083553, 'SALARY ~ C(EDUC) + SALBEGIN']
'''