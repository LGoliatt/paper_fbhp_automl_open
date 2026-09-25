# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl    
from sklearn.model_selection import train_test_split

pl.rc('text', usetex=True)
pl.rc('font',**{'family':'serif','serif':['Palatino']})

np.bool=np.bool_

from sklearn.preprocessing import LabelEncoder

dataset='FBHP'; plot=True; 

def read_fbhp(dataset):
    
    if dataset=='F1':
        return read_shammari(target='FBHP', dataset=dataset)    
    if dataset=='F2':
        return read_nwanwe(target='FBHP', dataset=dataset)    
    
#%%
def read_nwanwe(target='FBHP', dataset=None, seed=None, plot=False, latex=True):
    #%% 
    fn='./data/data_nwanwe/nwanwe.xlsx'
    
    for idx,sheet_name in [('Train','al_shammari'), ('Test','ayoub')]:
        
        X = pd.read_excel(fn,sheet_name=sheet_name)
        cols_to_drop=['SN',] if idx=='Train' else ['SN','WHT']
        X.drop(cols_to_drop,axis=1,inplace=True)
        target_names=['FBHP']
        
        variable_names=list(X.columns.drop(target_names))
           
        X=X[variable_names+target_names]
        X.dropna(inplace=True)
           
        categorical_columns=[]   
        for cc in categorical_columns:
            #print(cc)       
            le = LabelEncoder(); 
            le.fit(X[cc].values.ravel()); 
            X[cc] = le.transform(X[cc].values.ravel()).reshape(-1,1)
            #classes = dict(zip(le.transform(le.classes_), le.classes_))
    
        df = X[variable_names+target_names].copy()       
        if plot:
            pl.figure(figsize=(7, 7))
            corr = df.corr()
            mask = np.triu(np.ones_like(corr, dtype=np.bool))
            heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
            #heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
            heatmap.set_title(idx+': Correlation Heatmap ', fontdict={'fontsize':12}, pad=12);
            pl.savefig(idx+'_heatmap_correlation'+'.png',  bbox_inches='tight', dpi=300)
            pl.show()

        if idx=='Train':            
            X_train, y_train = X[variable_names], X[target_names]
            if latex:
                df=X_train.copy(); df[target_names]=y_train
                stat = df.describe().T
                stat.to_latex(buf=(dataset+'_train'+'.tex').lower(), index=True, caption='Basic statistics for dataset '+dataset+'.')
        else:
            X_test , y_test  = X[variable_names], X[target_names]
            if latex:
                df=X_test.copy(); df[target_names]=y_test
                stat = df.describe().T
                stat.to_latex(buf=(dataset+'_test'+'.tex').lower(), index=True, caption='Basic statistics for dataset '+dataset+'.')
    
            
        
    n=len(y_train);     
    n_samples, n_features = X_train.shape 

    
    
    task = 'regression' if target_names[0]=='UCS' else 'classification'
    task = 'regression'
    regression_data =  {
      'task'            : task,
      'name'            : dataset,
      'feature_names'   : np.array(variable_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train.values,
      'y_train'         : y_train.values.T,
      'X_test'          : X_test.values,
      'y_test'          : y_test.values.T,
      'targets'         : target_names,
       #'true_labels'     : classes,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'reference'       : "https://doi.org/10.1016/j.petlm.2023.03.003",
      'items'           : None,
      'normalize'       : None,
      }
    #%%
    return regression_data
        
#%%

   
#%%
def read_shammari(target='FBHP', dataset=None, seed=None, plot=False):
    #%% 
    fn='./data/data_shammari/al_shammari.xlsx'
    X = pd.read_excel(fn)
    cols_to_drop=['SN']
    X.drop(cols_to_drop,axis=1,inplace=True)
    target_names=['FBHP']
    
    variable_names=list(X.columns.drop(target_names))
       
    X=X[variable_names+target_names]
    X.dropna(inplace=True)
       
    categorical_columns=[]   
    for cc in categorical_columns:
        #print(cc)       
        le = LabelEncoder(); 
        le.fit(X[cc].values.ravel()); 
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1,1)
        #classes = dict(zip(le.transform(le.classes_), le.classes_))

    n=596
    X_train, y_train = X[variable_names][:n], X[target_names][:n]
    X_test , y_test  = X[variable_names][n:], X[target_names][n:]
    #X_train, X_test, y_train, y_test = train_test_split(X[variable_names], X[target_names], test_size=0.3, random_state=seed)

    
    df = X[variable_names+target_names].copy()
   
    if plot:
        pl.figure(figsize=(7, 7))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
        #heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
        heatmap.set_title(dataset+': Correlation Heatmap ', fontdict={'fontsize':12}, pad=12);
        pl.savefig(dataset+'_heatmap_correlation'+'.png',  bbox_inches='tight', dpi=300)
        pl.show()
        
    n=len(y_train);     
    n_samples, n_features = X_train.shape 

    df_train=X_train.copy(); df_train[target_names]=y_train
    stat_train = df_train.describe().T
    #print(stat_train.to_latex(),)
    stat_train.to_latex(buf=(dataset+'_train'+'.tex').lower(), index=True, caption='Basic statistics for dataset '+dataset+'.')

    
    task = 'regression' if target_names[0]=='UCS' else 'classification'
    task = 'regression'
    regression_data =  {
      'task'            : task,
      'name'            : dataset,
      'feature_names'   : np.array(variable_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train.values,
      'y_train'         : y_train.values.T,
      'X_test'          : X_test.values,
      'y_test'          : y_test.values.T,
      'targets'         : target_names,
       #'true_labels'     : classes,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'reference'       : "https://www.mdpi.com/1996-1944/15/21/7800",
      'items'           : None,
      'normalize'       : None,
      }
    #%%
    return regression_data
        
#%%

if __name__ == "__main__": 
    
    ds_names=['D'+str(i+1) for i in range(2)]
    for d in ds_names:
        D=read_fbhp(d)
        print(D['name'], D['n_samples'])
        print(D['feature_names'])
    
