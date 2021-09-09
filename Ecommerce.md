# Segmentation of ecommerce customers


# Table of Contents:
   
- [1. Introduction](#first-bullet)
- [2. Reading the file](#second-bullet)
- [3. EDA](#EDA)
    - [3.1. Keyword study](#KW)
- [4. Data engineering](#pre)    
- [5. Clustering](#clus)
    - [5.1 Elbow method](#elb)
    - [5.2 PCA](#pca)
-[6. Conclusions](#con)    



```python
pip install wordcloud
```

    Requirement already satisfied: wordcloud in c:\users\victo\anaconda3\lib\site-packages (1.8.1)
    Requirement already satisfied: matplotlib in c:\users\victo\anaconda3\lib\site-packages (from wordcloud) (3.3.4)
    Requirement already satisfied: pillow in c:\users\victo\anaconda3\lib\site-packages (from wordcloud) (8.2.0)
    Requirement already satisfied: numpy>=1.6.1 in c:\users\victo\anaconda3\lib\site-packages (from wordcloud) (1.20.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib->wordcloud) (2.4.7)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib->wordcloud) (2.8.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib->wordcloud) (0.10.0)Note: you may need to restart the kernel to use updated packages.
    
    Requirement already satisfied: six in c:\users\victo\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib->wordcloud) (1.15.0)
    


```python
pip install missingno
```

    Requirement already satisfied: missingno in c:\users\victo\anaconda3\lib\site-packages (0.5.0)
    Requirement already satisfied: scipy in c:\users\victo\anaconda3\lib\site-packages (from missingno) (1.6.2)
    Requirement already satisfied: numpy in c:\users\victo\anaconda3\lib\site-packages (from missingno) (1.20.1)
    Requirement already satisfied: seaborn in c:\users\victo\anaconda3\lib\site-packages (from missingno) (0.11.1)
    Requirement already satisfied: matplotlib in c:\users\victo\anaconda3\lib\site-packages (from missingno) (3.3.4)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib->missingno) (1.3.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib->missingno) (2.4.7)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib->missingno) (2.8.1)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib->missingno) (8.2.0)
    Requirement already satisfied: cycler>=0.10 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib->missingno) (0.10.0)
    Requirement already satisfied: six in c:\users\victo\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib->missingno) (1.15.0)
    Requirement already satisfied: pandas>=0.23 in c:\users\victo\anaconda3\lib\site-packages (from seaborn->missingno) (1.2.4)
    Requirement already satisfied: pytz>=2017.3 in c:\users\victo\anaconda3\lib\site-packages (from pandas>=0.23->seaborn->missingno) (2021.1)
    Note: you may need to restart the kernel to use updated packages.
    


```python
pip install pandas_profiling
```

    Requirement already satisfied: pandas_profiling in c:\users\victo\anaconda3\lib\site-packages (3.0.0)
    Requirement already satisfied: numpy>=1.16.0 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (1.20.1)
    Requirement already satisfied: pandas!=1.0.0,!=1.0.1,!=1.0.2,!=1.1.0,>=0.25.3 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (1.2.4)
    Requirement already satisfied: phik>=0.11.1 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (0.11.2)
    Requirement already satisfied: joblib in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (1.0.1)
    Requirement already satisfied: requests>=2.24.0 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (2.25.1)
    Requirement already satisfied: jinja2>=2.11.1 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (2.11.3)
    Requirement already satisfied: missingno>=0.4.2 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (0.5.0)
    Requirement already satisfied: tangled-up-in-unicode==0.1.0 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (0.1.0)
    Requirement already satisfied: tqdm>=4.48.2 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (4.59.0)
    Requirement already satisfied: seaborn>=0.10.1 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (0.11.1)
    Requirement already satisfied: pydantic>=1.8.1 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (1.8.2)
    Requirement already satisfied: visions[type_image_path]==0.7.1 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (0.7.1)
    Requirement already satisfied: PyYAML>=5.0.0 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (5.4.1)
    Requirement already satisfied: htmlmin>=0.1.12 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (0.1.12)
    Requirement already satisfied: matplotlib>=3.2.0 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (3.3.4)
    Requirement already satisfied: scipy>=1.4.1 in c:\users\victo\anaconda3\lib\site-packages (from pandas_profiling) (1.6.2)
    Requirement already satisfied: bottleneck in c:\users\victo\anaconda3\lib\site-packages (from visions[type_image_path]==0.7.1->pandas_profiling) (1.3.2)
    Requirement already satisfied: networkx>=2.4 in c:\users\victo\anaconda3\lib\site-packages (from visions[type_image_path]==0.7.1->pandas_profiling) (2.5)
    Requirement already satisfied: multimethod==1.4 in c:\users\victo\anaconda3\lib\site-packages (from visions[type_image_path]==0.7.1->pandas_profiling) (1.4)
    Requirement already satisfied: attrs>=19.3.0 in c:\users\victo\anaconda3\lib\site-packages (from visions[type_image_path]==0.7.1->pandas_profiling) (20.3.0)
    Requirement already satisfied: Pillow in c:\users\victo\anaconda3\lib\site-packages (from visions[type_image_path]==0.7.1->pandas_profiling) (8.2.0)
    Requirement already satisfied: imagehash in c:\users\victo\anaconda3\lib\site-packages (from visions[type_image_path]==0.7.1->pandas_profiling) (4.2.0)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\users\victo\anaconda3\lib\site-packages (from jinja2>=2.11.1->pandas_profiling) (1.1.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.2.0->pandas_profiling) (2.4.7)
    Requirement already satisfied: cycler>=0.10 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.2.0->pandas_profiling) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.2.0->pandas_profiling) (1.3.1)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\victo\anaconda3\lib\site-packages (from matplotlib>=3.2.0->pandas_profiling) (2.8.1)
    Requirement already satisfied: six in c:\users\victo\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib>=3.2.0->pandas_profiling) (1.15.0)
    Requirement already satisfied: decorator>=4.3.0 in c:\users\victo\anaconda3\lib\site-packages (from networkx>=2.4->visions[type_image_path]==0.7.1->pandas_profiling) (5.0.6)
    Requirement already satisfied: pytz>=2017.3 in c:\users\victo\anaconda3\lib\site-packages (from pandas!=1.0.0,!=1.0.1,!=1.0.2,!=1.1.0,>=0.25.3->pandas_profiling) (2021.1)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in c:\users\victo\anaconda3\lib\site-packages (from pydantic>=1.8.1->pandas_profiling) (3.7.4.3)
    Requirement already satisfied: chardet<5,>=3.0.2 in c:\users\victo\anaconda3\lib\site-packages (from requests>=2.24.0->pandas_profiling) (4.0.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\victo\anaconda3\lib\site-packages (from requests>=2.24.0->pandas_profiling) (2020.12.5)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\victo\anaconda3\lib\site-packages (from requests>=2.24.0->pandas_profiling) (1.26.4)
    Requirement already satisfied: idna<3,>=2.5 in c:\users\victo\anaconda3\lib\site-packages (from requests>=2.24.0->pandas_profiling) (2.10)
    Requirement already satisfied: PyWavelets in c:\users\victo\anaconda3\lib\site-packages (from imagehash->visions[type_image_path]==0.7.1->pandas_profiling) (1.1.1)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import pandas as pd
import pandas_profiling
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

import datetime, nltk, warnings
import gc
import matplotlib.cm as cm
import itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
%matplotlib inline




import warnings
# current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

import missingno as msno # missing data visualization module for Python



%matplotlib inline
color = sns.color_palette()


pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 100)
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.2.0.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




# Auxiliar functions


```python
#Function to detect substitute the rare lables
def rare_encoding(X_train, X_test, variable, tolerance):

    X_train = X_train.copy()
    X_test = X_test.copy()

    # find the most frequent category
    frequent_cat = find_non_rare_labels(X_train, variable, tolerance)

    # re-group rare labels
    X_train[variable] = np.where(X_train[variable].isin(
        frequent_cat), X_train[variable], 'Rare')
    
    X_test[variable] = np.where(X_test[variable].isin(
        frequent_cat), X_test[variable], 'Rare')

    return X_train, X_test
    
################################################################################################################################

#Function to detect the rare labels
def find_non_rare_labels(df, variable, tolerance):
    
    temp = df.groupby([variable])[variable].count() / len(df)
    
    non_rare = [x for x in temp.loc[temp>tolerance].index.values]
    
    return non_rare
################################################################################################################################

#Function to return a pivot table relating two variables
def Pivot_Table(X_train,y_train,variable):
    X_train['target'] = y_train
    
    temp = X_train[[variable, 'target']].groupby([variable],
                                                    as_index=False).mean().sort_values(by='target', ascending=False)
    
    X_train.drop('target',axis=1,inplace = True)
    
    return temp

################################################################################################################################

#Function to plot distributions   
def diagnostic_plots(df, variable):
    # function takes a dataframe (df) and
    # the variable of interest as arguments

    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.distplot(df[variable],fit=norm, bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('Variable quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()
    
################################################################################################################################
    
#Function to plot densities to compare imputed features with originals   
def DensityPlots(X_train, X_test, X_train_imputed,X_test_imputed, variable):
    # plot the distribution of the imputed variable

    fig = plt.figure(figsize=(16, 4))


    ax = fig.add_subplot(131)

    X_train[variable].plot(kind='kde', ax=ax, color='orange')
    X_train_imputed[variable].plot(kind='kde', ax=ax, color='black')

    # add legends
    lines, labels = ax.get_legend_handles_labels()
    labels = ['train orig', 'train imp']
    ax.legend(lines, labels, loc='best')
    plt.title(variable +' train distribution')

    #Second image
    ay = fig.add_subplot(132)

    X_test[variable].plot(kind='kde', ax=ay, color='orange')
    X_test_imputed[variable].plot(kind='kde', ax=ay, color='black')

    # add legends
    lines, labels = ay.get_legend_handles_labels()
    labels2 = ['test orig', 'test imp']
    ay.legend(lines, labels2, loc='best')
    plt.title(variable +' test distribution')

    plt.show()
    
################################################################################################################################
   
#Function that returns features with less categories than the tolerance    
def FewCategories (X_train, tol) :

    columns = []
    
    for i in X_train.columns:
    
        if(X_train[i].value_counts().count() <= tol):
            columns.append(i)
            
            
    return columns

################################################################################################################################

#Function that returns features with more categories than the tolerance
def ManyCategories (X_train, tol) :
    
    columns = []
    
    for i in X_train.columns:
    
        if(X_train[i].value_counts().count() >= tol):
            columns.append(i)
            
            
    return columns

################################################################################################################################

#Function to fit and cross validate the algorithm
def fit_ml_algo(algo, X_train, y_train, cv):
    
    # One Pass
    model = algo.fit(X_train, y_train)
    scores = cross_val_score(algo, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    
    # Cross Validation 
    
    train_pred = model_selection.cross_val_predict(algo, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs = -1)
    # Cross-validation accuracy metric
    acc_cv = np.sqrt((-1)* scores).mean()
    
    return train_pred, scores, acc_cv

################################################################################################################################

#Function that plots relationships from the target and a variable
def Relationships (X_train, y_train, variables):
    temp = X_train
    temp['target'] = y_train
    for var in X_train[variables].columns:

        fig = plt.figure()
        fig = X_train.groupby([var])['target'].mean().plot()
        fig.set_title('Relationship between {} and target'.format(var))
        fig.set_ylabel('Mean value of target')
        plt.show()
        
################################################################################################################################
  
#Function that returns a dictionary with the mappings for ordered encoding
def find_category_mappings(df, variable, target):

    # first  we generate an ordered list with the labels
    ordered_labels = df.groupby([variable
                                 ])[target].mean().sort_values().index

    # return the dictionary with mappings
    return {k: i for i, k in enumerate(ordered_labels, 0)}

################################################################################################################################

#Function that does the encoding based on a map
def integer_encode(train, test, variable, ordinal_mapping):

    train[variable] = train[variable].map(ordinal_mapping)
    test[variable] = test[variable].map(ordinal_mapping)
    
################################################################################################################################

#Finds the boundaries for outliers
def find_skewed_boundaries(df, variable, distance):

    # Let's calculate the boundaries outside which sit the outliers
    # for skewed distributions

    # distance passed as an argument, gives us the option to
    # estimate 1.5 times or 3 times the IQR to calculate
    # the boundaries.

    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary

################################################################################################################################

#Function that returns the informations of a dataset. Perfect for the first look.
def rstr(df, pred=None): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: x.unique())
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape:', df.shape)
    
    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis = 1)
        str.columns = cols

    else:
        corr = df.corr()[pred]
        #str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        #cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]
        data = {'types': types, 'counts': counts, 'distincts': distincts,
                'nulls':nulls, 'missing_ration':missing_ration,
                'uniques':uniques, 'skewness':skewness, 'kurtosis':kurtosis, corr_col:corr }
        str = pd.DataFrame(data) 
        
    
   
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n',str.types.value_counts())
    print('___________________________')
    return str

################################################################################################################################

#Returns the correlation table for the features in a model
def Correlation_Table(X_train, model):

    coeff_df = pd.DataFrame(X_train.columns.delete(0))
    coeff_df.columns = ['Feature']
    coeff_df["Correlation"] = pd.Series(model.coef_)

    return coeff_df.sort_values(by='Correlation', ascending=False)

################################################################################################################################

#Returns the correlation plot in a heatmap

def Correlation_Plot(X_train,y_train,variables):
    sns.heatmap(pd.concat([X_train[variables], y_train], axis=1).corr(),
                annot=True, fmt=".2f",cbar_kws={'label': 'Percentage %'},cmap="plasma",ax=ax)
    ax.set_title("Correlation Plot")
    plt.show()
    
################################################################################################################################

#Plots the comparison between variables befor and after transformation  
def transformation_skew (X, kind='log'):
    import feature_engine.transformation as vt

    if (kind == 'log'):
        X_unsk = X.apply(lambda x: np.log(1+x)) # X_unsk --> Unskewed X
    if(kind == 'boxcox'):
        positives = []
        for var in X.columns:
            if ((X[var]>0).astype(int).sum() == X[var].count()):
                positives.append(var)
                
        bct = vt.BoxCoxTransformer(variables = positives)
        bct.fit(X)
        X_unsk = bct.transform(X)
    if(kind == 'yj'):
        yjt = vt.YeoJohnsonTransformer()
        yjt.fit(X)
        X_unsk = yjt.transform(X)

    X_unsk_vals = X_unsk.skew()             # X_unsk_vals --> Skew coefficients of X_unsk

    ax = sns.barplot(x=X.skew().values, y=X.skew().index, alpha=0.2)
    ax2 = ax.twinx()
    sns.barplot(x=X_unsk_vals.values, y=X_unsk_vals.index, ax=ax2)
    plt.show()
```

<a class="anchor" id="second-bullet"></a>
# 2. Reading the file


```python
#df will remain untouched
df = pd.read_csv('data.csv', encoding = 'ISO-8859-1')

df_initial = pd.read_csv('data.csv',encoding="ISO-8859-1",
                         dtype={'CustomerID': str,'InvoiceID': str})

df_initial['InvoiceDate'] = pd.to_datetime(df_initial['InvoiceDate'])
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 541909 entries, 0 to 541908
    Data columns (total 8 columns):
     #   Column       Non-Null Count   Dtype  
    ---  ------       --------------   -----  
     0   InvoiceNo    541909 non-null  object 
     1   StockCode    541909 non-null  object 
     2   Description  540455 non-null  object 
     3   Quantity     541909 non-null  int64  
     4   InvoiceDate  541909 non-null  object 
     5   UnitPrice    541909 non-null  float64
     6   CustomerID   406829 non-null  float64
     7   Country      541909 non-null  object 
    dtypes: float64(2), int64(1), object(5)
    memory usage: 33.1+ MB
    


```python
df.isnull().sum().sort_values(ascending = False)
```




    CustomerID     135080
    Description      1454
    InvoiceNo           0
    StockCode           0
    Quantity            0
    InvoiceDate         0
    UnitPrice           0
    Country             0
    dtype: int64




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>12/1/2010 8:26</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>12/1/2010 8:26</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>12/1/2010 8:26</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536365</td>
      <td>84029G</td>
      <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
      <td>6</td>
      <td>12/1/2010 8:26</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536365</td>
      <td>84029E</td>
      <td>RED WOOLLY HOTTIE WHITE HEART.</td>
      <td>6</td>
      <td>12/1/2010 8:26</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>



<a class="anchor" id="EDA"></a>
# 3. Exploratory data analysis



```python
df['invoice_date'] = pd.to_datetime(df.InvoiceDate, format='%m/%d/%Y %H:%M')
```


```python
#Separating month and day to plot 
df_new = df.dropna()
df_new = df_new[df_new.Quantity > 0]
df_new['amount_spent'] = df_new['Quantity'] * df_new['UnitPrice']


df_new.insert(loc=2, column='year_month', value=df_new['invoice_date'].map(lambda x: 100*x.year + x.month))
df_new.insert(loc=3, column='month', value=df_new.invoice_date.dt.month)
# +1 to make Monday=1.....until Sunday=7
df_new.insert(loc=4, column='day', value=(df_new.invoice_date.dt.dayofweek)+1)
df_new.insert(loc=5, column='hour', value=df_new.invoice_date.dt.hour)
```


```python
#Orders per month
ax = df_new.groupby('InvoiceNo')['year_month'].unique().value_counts().sort_index().plot(kind = 'bar',
                                                                                         color='#3192B3',figsize=(15,6))
ax.set_xlabel('Month',fontsize=15)
ax.set_ylabel('Number of Orders',fontsize=15)
ax.set_title('Number of orders for different Months (1st Dec 2010 - 9th Dec 2011)',fontsize=15)
ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','Jun_11','July_11','Aug_11',
                    'Sep_11','Oct_11','Nov_11','Dec_11'), rotation='horizontal', fontsize=13)
plt.grid(False)
plt.show()
```


    
![png](output_15_0.png)
    



```python
#Orders per day of the week
ax = df_new.groupby('InvoiceNo')['day'].unique().value_counts().sort_index().plot(kind = 'bar',figsize=(15,6),
                                                                                  color = '#F9b342')
ax.set_xlabel('Day',fontsize=15)
ax.set_ylabel('Number of Orders',fontsize=15)
ax.set_title('Number of orders for different Days',fontsize=15)
ax.set_xticklabels(('Monday','Tuesday','Wednesday','Thursday','Friday','Sunday'), rotation='horizontal', fontsize=15)
plt.grid(False)
plt.show()
```


    
![png](output_16_0.png)
    



```python
#Checking free items and how often they're given
df_free = df_new[df_new.UnitPrice == 0]
ax = df_free.year_month.value_counts().sort_index().plot(kind = 'bar',figsize=(12,6), color='#706f6f')
ax.set_xlabel('Month',fontsize=15)
ax.set_ylabel('Frequency',fontsize=15)
ax.set_title('Frequency for different Months (Dec 2010 - Dec 2011)',fontsize=15)
ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','July_11',
                    'Aug_11','Sep_11','Oct_11','Nov_11'), rotation='horizontal', fontsize=13)
plt.grid(False)
plt.show()
```


    
![png](output_17_0.png)
    



```python
#Let's start to use od df_initial and drop the orders with no customer ID

df_initial.dropna(axis = 0, subset = ['CustomerID'], inplace = True)
df_initial.isnull().sum().sort_values(ascending = False)
```




    InvoiceNo      0
    StockCode      0
    Description    0
    Quantity       0
    InvoiceDate    0
    UnitPrice      0
    CustomerID     0
    Country        0
    dtype: int64




```python
#Let's remove the duplicates
print('removed rows: {}'.format(df_initial.duplicated().sum()))
df_initial.drop_duplicates(inplace = True)
```

    removed rows: 5225
    


```python
#Let's take a look at the countries
temp = df_initial[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
temp = temp.reset_index(drop = False)
countries = temp['Country'].value_counts()
print('Number of different countries: {}'.format(len(countries)))
```

    Number of different countries: 37
    


```python
#We can explore a chloropleth map
data = dict(type='choropleth',
locations = countries.index,
locationmode = 'country names', z = countries,
text = countries.index, colorbar = {'title':'Order no.'},
colorscale=[[0, 'rgb(224,255,255)'],
            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
            [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],
            [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
            [1, 'rgb(227,26,28)']],    
reversescale = False
           )
#_______________________
layout = dict(title='Orders per country in a map',
geo = dict(showframe = True, projection={'type':'mercator'}))
#______________
choromap = go.Figure(data = [data], layout = layout)
iplot(choromap, validate=False)
```


<div>                            <div id="71052f20-cf48-401c-9483-7a8fe22d98f5" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("71052f20-cf48-401c-9483-7a8fe22d98f5")) {                    Plotly.newPlot(                        "71052f20-cf48-401c-9483-7a8fe22d98f5",                        [{"colorbar":{"title":{"text":"Order no."}},"colorscale":[[0,"rgb(224,255,255)"],[0.01,"rgb(166,206,227)"],[0.02,"rgb(31,120,180)"],[0.03,"rgb(178,223,138)"],[0.05,"rgb(51,160,44)"],[0.1,"rgb(251,154,153)"],[0.2,"rgb(255,255,0)"],[1,"rgb(227,26,28)"]],"locationmode":"country names","locations":["United Kingdom","Germany","France","EIRE","Belgium","Spain","Netherlands","Switzerland","Portugal","Australia","Italy","Finland","Sweden","Norway","Channel Islands","Japan","Poland","Denmark","Cyprus","Austria","Singapore","Malta","Unspecified","Iceland","USA","Canada","Greece","Israel","Czech Republic","European Community","Lithuania","United Arab Emirates","Saudi Arabia","Bahrain","Lebanon","Brazil","RSA"],"reversescale":false,"text":["United Kingdom","Germany","France","EIRE","Belgium","Spain","Netherlands","Switzerland","Portugal","Australia","Italy","Finland","Sweden","Norway","Channel Islands","Japan","Poland","Denmark","Cyprus","Austria","Singapore","Malta","Unspecified","Iceland","USA","Canada","Greece","Israel","Czech Republic","European Community","Lithuania","United Arab Emirates","Saudi Arabia","Bahrain","Lebanon","Brazil","RSA"],"type":"choropleth","z":[19857,603,458,319,119,105,101,71,70,69,55,48,46,40,33,28,24,21,20,19,10,10,8,7,7,6,6,6,5,5,4,3,2,2,1,1,1]}],                        {"geo":{"projection":{"type":"mercator"},"showframe":true},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Orders per country in a map"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('71052f20-cf48-401c-9483-7a8fe22d98f5');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
#Investigating how many unique values are there in some columns
pd.DataFrame([{'products': len(df_initial['StockCode'].value_counts()),    
               'transactions': len(df_initial['InvoiceNo'].value_counts()),
               'customers': len(df_initial['CustomerID'].value_counts()),  
              }],
             columns = ['products', 'transactions', 'customers'], index = ['quantity'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>products</th>
      <th>transactions</th>
      <th>customers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>quantity</th>
      <td>3684</td>
      <td>22190</td>
      <td>4372</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Investigating the number of products in each purchase
temp = df_initial.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
nb_products_per_basket = temp.rename(columns = {'InvoiceDate':'Number of products'})
nb_products_per_basket[:10].sort_values('CustomerID')

#With these informations, we already have a brief idea of some clusters
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>InvoiceNo</th>
      <th>Number of products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12346</td>
      <td>541431</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12346</td>
      <td>C541433</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12347</td>
      <td>537626</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12347</td>
      <td>542237</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12347</td>
      <td>549222</td>
      <td>24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12347</td>
      <td>556201</td>
      <td>18</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12347</td>
      <td>562032</td>
      <td>22</td>
    </tr>
    <tr>
      <th>7</th>
      <td>12347</td>
      <td>573511</td>
      <td>47</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12347</td>
      <td>581180</td>
      <td>11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12348</td>
      <td>539318</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Dealing with the cancelation (C starting the InvoiceNo)

nb_products_per_basket['order_canceled'] = nb_products_per_basket['InvoiceNo'].apply(lambda x:int('C' in x))
display(nb_products_per_basket.head())
#______________________________________________________________________________________________
n1 = nb_products_per_basket['order_canceled'].sum()
n2 = nb_products_per_basket.shape[0]
print('Number of orders canceled: {}/{} ({:.2f}%) '.format(n1, n2, n1/n2*100))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>InvoiceNo</th>
      <th>Number of products</th>
      <th>order_canceled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12346</td>
      <td>541431</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12346</td>
      <td>C541433</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12347</td>
      <td>537626</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12347</td>
      <td>542237</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12347</td>
      <td>549222</td>
      <td>24</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    Number of orders canceled: 3654/22190 (16.47%) 
    


```python
#Note that when we have a canceled order, we sometimes have a "duplicate" of it before, but with the positive quantity.
#There are some orders made before dec 2010 that have been canceled and are on the dataset, so they don't have the counterparts

df_cleaned = df_initial.copy(deep = True)
df_cleaned['QuantityCanceled'] = 0

entry_to_remove = [] ; doubtfull_entry = []

for index, col in  df_initial.iterrows():
    if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue        
    df_test = df_initial[(df_initial['CustomerID'] == col['CustomerID']) &
                         (df_initial['StockCode']  == col['StockCode']) & 
                         (df_initial['InvoiceDate'] < col['InvoiceDate']) & 
                         (df_initial['Quantity']   > 0)].copy()
    #_________________________________
    # Cancelation WITHOUT counterpart
    if (df_test.shape[0] == 0): 
        doubtfull_entry.append(index)
    #________________________________
    # Cancelation WITH a counterpart
    elif (df_test.shape[0] == 1): 
        index_order = df_test.index[0]
        df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
        entry_to_remove.append(index)        
    #______________________________________________________________
    # Various counterparts exist in orders: we delete the last one
    elif (df_test.shape[0] > 1): 
        df_test.sort_index(axis=0 ,ascending=False, inplace = True)        
        for ind, val in df_test.iterrows():
            if val['Quantity'] < -col['Quantity']: continue
            df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
            entry_to_remove.append(index) 
            break
```


```python
print("entry_to_remove: {}".format(len(entry_to_remove)))
print("doubtfull_entry: {}".format(len(doubtfull_entry)))
```

    entry_to_remove: 7521
    doubtfull_entry: 1226
    


```python
df_cleaned.drop(entry_to_remove, axis = 0, inplace = True)
df_cleaned.drop(doubtfull_entry, axis = 0, inplace = True)
remaining_entries = df_cleaned[(df_cleaned['Quantity'] < 0) & (df_cleaned['StockCode'] != 'D')]
print("nb of entries to delete: {}".format(remaining_entries.shape[0]))
remaining_entries[:5]

#These are canceled orders without previous activity by the customer
```

    nb of entries to delete: 48
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>QuantityCanceled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>77598</th>
      <td>C542742</td>
      <td>84535B</td>
      <td>FAIRY CAKES NOTEBOOK A6 SIZE</td>
      <td>-94</td>
      <td>2011-01-31 16:26:00</td>
      <td>0.65</td>
      <td>15358</td>
      <td>United Kingdom</td>
      <td>0</td>
    </tr>
    <tr>
      <th>90444</th>
      <td>C544038</td>
      <td>22784</td>
      <td>LANTERN CREAM GAZEBO</td>
      <td>-4</td>
      <td>2011-02-15 11:32:00</td>
      <td>4.95</td>
      <td>14659</td>
      <td>United Kingdom</td>
      <td>0</td>
    </tr>
    <tr>
      <th>111968</th>
      <td>C545852</td>
      <td>22464</td>
      <td>HANGING METAL HEART LANTERN</td>
      <td>-5</td>
      <td>2011-03-07 13:49:00</td>
      <td>1.65</td>
      <td>14048</td>
      <td>United Kingdom</td>
      <td>0</td>
    </tr>
    <tr>
      <th>116064</th>
      <td>C546191</td>
      <td>47566B</td>
      <td>TEA TIME PARTY BUNTING</td>
      <td>-35</td>
      <td>2011-03-10 10:57:00</td>
      <td>0.70</td>
      <td>16422</td>
      <td>United Kingdom</td>
      <td>0</td>
    </tr>
    <tr>
      <th>132642</th>
      <td>C547675</td>
      <td>22263</td>
      <td>FELT EGG COSY LADYBIRD</td>
      <td>-49</td>
      <td>2011-03-24 14:07:00</td>
      <td>0.66</td>
      <td>17754</td>
      <td>United Kingdom</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Let's analyse stock code and look only for the letters
list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
list_special_codes
```




    array(['POST', 'D', 'C2', 'M', 'BANK CHARGES', 'PADS', 'DOT'],
          dtype=object)




```python
#We need to create a variable storing the total price of the purchases
df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])
df_cleaned.sort_values('CustomerID').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>QuantityCanceled</th>
      <th>TotalPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61619</th>
      <td>541431</td>
      <td>23166</td>
      <td>MEDIUM CERAMIC TOP STORAGE JAR</td>
      <td>74215</td>
      <td>2011-01-18 10:01:00</td>
      <td>1.04</td>
      <td>12346</td>
      <td>United Kingdom</td>
      <td>74215</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>148288</th>
      <td>549222</td>
      <td>22375</td>
      <td>AIRLINE BAG VINTAGE JET SET BROWN</td>
      <td>4</td>
      <td>2011-04-07 10:43:00</td>
      <td>4.25</td>
      <td>12347</td>
      <td>Iceland</td>
      <td>0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>428971</th>
      <td>573511</td>
      <td>22698</td>
      <td>PINK REGENCY TEACUP AND SAUCER</td>
      <td>12</td>
      <td>2011-10-31 12:25:00</td>
      <td>2.95</td>
      <td>12347</td>
      <td>Iceland</td>
      <td>0</td>
      <td>35.4</td>
    </tr>
    <tr>
      <th>428970</th>
      <td>573511</td>
      <td>47559B</td>
      <td>TEA TIME OVEN GLOVE</td>
      <td>10</td>
      <td>2011-10-31 12:25:00</td>
      <td>1.25</td>
      <td>12347</td>
      <td>Iceland</td>
      <td>0</td>
      <td>12.5</td>
    </tr>
    <tr>
      <th>428969</th>
      <td>573511</td>
      <td>47567B</td>
      <td>TEA TIME KITCHEN APRON</td>
      <td>6</td>
      <td>2011-10-31 12:25:00</td>
      <td>5.95</td>
      <td>12347</td>
      <td>Iceland</td>
      <td>0</td>
      <td>35.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
#But this still isn't ideal. Let's try to compute the total amount per order

#___________________________________________
# sum
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
basket_price = temp.rename(columns = {'TotalPrice':'Basket Price'})
#_____________________
# date 
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
#______________________________________
# selection
basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID')

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>InvoiceNo</th>
      <th>Basket Price</th>
      <th>InvoiceDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>12347</td>
      <td>537626</td>
      <td>711.79</td>
      <td>2010-12-07 14:57:00.000001024</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12347</td>
      <td>542237</td>
      <td>475.39</td>
      <td>2011-01-26 14:29:59.999999744</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12347</td>
      <td>549222</td>
      <td>636.25</td>
      <td>2011-04-07 10:42:59.999999232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12347</td>
      <td>556201</td>
      <td>382.52</td>
      <td>2011-06-09 13:01:00.000000256</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12347</td>
      <td>562032</td>
      <td>584.91</td>
      <td>2011-08-02 08:48:00.000000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18619</th>
      <td>18283</td>
      <td>557956</td>
      <td>192.80</td>
      <td>2011-06-23 19:20:00.000000000</td>
    </tr>
    <tr>
      <th>18628</th>
      <td>18283</td>
      <td>580872</td>
      <td>208.00</td>
      <td>2011-12-06 12:02:00.000001792</td>
    </tr>
    <tr>
      <th>18630</th>
      <td>18287</td>
      <td>570715</td>
      <td>1001.32</td>
      <td>2011-10-12 10:22:59.999998720</td>
    </tr>
    <tr>
      <th>18629</th>
      <td>18287</td>
      <td>554065</td>
      <td>765.28</td>
      <td>2011-05-22 10:38:59.999998976</td>
    </tr>
    <tr>
      <th>18631</th>
      <td>18287</td>
      <td>573167</td>
      <td>70.68</td>
      <td>2011-10-28 09:29:00.000000000</td>
    </tr>
  </tbody>
</table>
<p>18398 rows × 4 columns</p>
</div>




```python
#Let's visualize the purchases in price intervals
price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
count_price = []
for i, price in enumerate(price_range):
    if i == 0: continue
    val = basket_price[(basket_price['Basket Price'] < price) &
                       (basket_price['Basket Price'] > price_range[i-1])]['Basket Price'].count()
    count_price.append(val)
#____________________________________________
       
plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(11, 6))
colors = ['yellowgreen', 'gold', 'wheat', 'c', 'violet', 'royalblue','firebrick']
labels = [ '{}<x<{}'.format(price_range[i-1], s) for i,s in enumerate(price_range) if i != 0]
sizes  = count_price
explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(sizes))]
ax.pie(sizes, explode = explode, labels=labels, colors = colors,
       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
       shadow = False, startangle=0)
ax.axis('equal')
f.text(0.5, 1.01, "Pie chart of price ranges", ha='center', fontsize = 18);
```


    
![png](output_31_0.png)
    


<a class="anchor" id="KW"></a>
## 3.1. Keywords study 


```python
#Now, let's try to extract the keywords from the description.

is_noun = lambda pos: pos[:2] == 'NN'

def keywords_inventory(dataframe, column = 'Description'):
    stemmer = nltk.stem.SnowballStemmer("english")
    keywords_roots  = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys   = []
    count_keywords  = dict()
    icount = 0
    for s in dataframe[column]:
        if pd.isnull(s): continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        
        for t in nouns:
            t = t.lower() ; racine = stemmer.stem(t)
            if racine in keywords_roots:                
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1                
            else:
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1
    
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:  
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k ; min_length = len(k)            
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
                   
    print("Number of keywords in variable '{}': {}".format(column,len(category_keys)))
    return category_keys, keywords_roots, keywords_select, count_keywords
```


```python
#Now, to call the function
#Downloading packages from nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#Calling the function

df_produits = pd.DataFrame(df_initial['Description'].unique()).rename(columns = {0:'Description'})
keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_produits)
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\victo\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     C:\Users\victo\AppData\Roaming\nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!
    

    Number of keywords in variable 'Description': 1483
    


```python
list_products = []
for k,v in count_keywords.items():
    list_products.append([keywords_select[k],v])
list_products.sort(key = lambda x:x[1], reverse = True)
list_products
```




    [['heart', 267],
     ['vintage', 211],
     ['set', 206],
     ['pink', 189],
     ['bag', 165],
     ['box', 155],
     ['glass', 149],
     ['christmas', 137],
     ['design', 128],
     ['candle', 121],
     ['holder', 120],
     ['flower', 120],
     ['decoration', 109],
     ['metal', 99],
     ['retrospot', 90],
     ['card', 90],
     ['necklac', 85],
     ['paper', 84],
     ['blue', 80],
     ['art', 75],
     ['silver', 72],
     ['cake', 70],
     ['polkadot', 68],
     ['cover', 68],
     ['mug', 66],
     ['tin', 66],
     ['wrap', 65],
     ['sign', 64],
     ['pack', 61],
     ['egg', 61],
     ['bracelet', 61],
     ['star', 59],
     ['bowl', 57],
     ['mini', 56],
     ['tea', 55],
     ['garden', 55],
     ['wall', 55],
     ['ivory', 54],
     ['cushion', 54],
     ['frame', 52],
     ['mirror', 50],
     ['cream', 49],
     ['home', 49],
     ['gift', 49],
     ['earrings', 49],
     ['ring', 46],
     ['bird', 45],
     ['bottle', 44],
     ['clock', 44],
     ['paisley', 42],
     ['letter', 42],
     ['party', 42],
     ['charm', 41],
     ['wood', 40],
     ['ribbon', 40],
     ['jar', 39],
     ['garland', 39],
     ['hook', 39],
     ['gold', 39],
     ['easter', 39],
     ['bead', 38],
     ['drawer', 37],
     ['cup', 37],
     ['zinc', 37],
     ['water', 36],
     ['photo', 36],
     ['bell', 36],
     ['plate', 35],
     ['tray', 35],
     ['pencil', 35],
     ['skull', 34],
     ['spot', 33],
     ['butterfly', 33],
     ['children', 32],
     ['case', 31],
     ['enamel', 31],
     ['round', 30],
     ['tissue', 30],
     ['sweetheart', 30],
     ['stand', 30],
     ['sticker', 30],
     ['spaceboy', 29],
     ['light', 29],
     ['cutlery', 29],
     ['diamante', 29],
     ['union', 28],
     ['rabbit', 28],
     ['magnet', 28],
     ['tree', 28],
     ['pantry', 28],
     ['lunch', 27],
     ['pot', 27],
     ['colour', 26],
     ['storage', 26],
     ['book', 26],
     ['bunny', 25],
     ['chocolate', 25],
     ['basket', 25],
     ['birthday', 25],
     ['cat', 25],
     ['hair', 25],
     ['feltcraft', 24],
     ['coffee', 24],
     ['dog', 24],
     ['fairy', 23],
     ['trinket', 23],
     ['w', 23],
     ['gingham', 23],
     ['flock', 23],
     ['door', 23],
     ['drop', 23],
     ['towel', 22],
     ['london', 22],
     ['cabinet', 22],
     ['baroque', 22],
     ['sweet', 22],
     ['wire', 22],
     ['kit', 22],
     ['babushka', 22],
     ['t-light', 22],
     ['notebook', 22],
     ['reel', 22],
     ['antique', 22],
     ['retro', 22],
     ['number', 22],
     ['jack', 21],
     ['woodland', 21],
     ['strawberry', 21],
     ['wicker', 21],
     ['apple', 21],
     ['tube', 21],
     ['daisy', 21],
     ['shell', 21],
     ['hand', 20],
     ['kitchen', 20],
     ['purse', 20],
     ['dinner', 20],
     ['style', 20],
     ['hanger', 19],
     ['i', 19],
     ['table', 19],
     ['pen', 19],
     ['chick', 19],
     ['leaf', 19],
     ['jam', 18],
     ['tape', 18],
     ['toy', 18],
     ['knob', 18],
     ['doormat', 18],
     ['ball', 18],
     ['doiley', 18],
     ['warmer', 17],
     ['wreath', 17],
     ['stripe', 17],
     ['shape', 17],
     ['parasol', 16],
     ['cherry', 16],
     ['travel', 16],
     ['doilies', 16],
     ['biscuit', 16],
     ['regency', 16],
     ['hen', 16],
     ['jigsaw', 15],
     ['rack', 15],
     ['coaster', 15],
     ['money', 15],
     ['cottage', 15],
     ['incense', 15],
     ['crystal', 15],
     ['tag', 15],
     ['medium', 15],
     ['dish', 15],
     ['image', 15],
     ['green', 14],
     ['picture', 14],
     ['wooden', 14],
     ['piece', 14],
     ['bathroom', 14],
     ['girl', 14],
     ['fruit', 14],
     ['childs', 14],
     ['pan', 14],
     ['food', 14],
     ['orbit', 14],
     ['alphabet', 13],
     ['toadstool', 13],
     ['+', 13],
     ['time', 13],
     ['milk', 13],
     ['funky', 13],
     ['tidy', 13],
     ['plant', 13],
     ['diner', 13],
     ['house', 13],
     ['square', 13],
     ['craft', 13],
     ['lace', 13],
     ['jewel', 13],
     ['wallet', 13],
     ['point', 13],
     ['flag', 12],
     ['circus', 12],
     ['bin', 12],
     ['clip', 12],
     ['candy', 12],
     ['treasure', 12],
     ['cotton', 12],
     ['pocket', 12],
     ['shop', 12],
     ['pearl', 12],
     ['charlotte', 11],
     ['doll', 11],
     ['chain', 11],
     ['dinosaur', 11],
     ['breakfast', 11],
     ['jug', 11],
     ['hanging', 11],
     ['orange', 11],
     ['crochet', 11],
     ['soap', 11],
     ['print', 11],
     ['bunting', 11],
     ['blossom', 11],
     ['felt', 10],
     ['saucer', 10],
     ['stick', 10],
     ['beaker', 10],
     ['owl', 10],
     ['balloon', 10],
     ['doorstop', 10],
     ['spoon', 10],
     ['napkins', 10],
     ['lamp', 10],
     ['string', 10],
     ['lavender', 10],
     ['slate', 10],
     ['kids', 10],
     ['luggage', 10],
     ['vanilla', 10],
     ['animal', 10],
     ['king', 10],
     ['herb', 10],
     ['english', 10],
     ['set/4', 10],
     ['jewellery', 10],
     ['tutti', 10],
     ['container', 10],
     ['boudicca', 10],
     ['pastel', 10],
     ['block', 9],
     ['love', 9],
     ['snack', 9],
     ['parade', 9],
     ['charlie', 9],
     ['board', 9],
     ['picnic', 9],
     ['placemat', 9],
     ['candleholder', 9],
     ['peg', 9],
     ['calendar', 9],
     ['apron', 9],
     ['purple', 9],
     ['office', 9],
     ['horse', 9],
     ['marker', 9],
     ['mint', 9],
     ['handbag', 9],
     ['morris', 9],
     ['honeycomb', 9],
     ['patch', 9],
     ['shelf', 9],
     ['glitter', 9],
     ['sugar', 9],
     ['baby', 9],
     ['chunky', 9],
     ['amethyst', 9],
     ['hoop', 9],
     ['circle', 9],
     ['phone', 9],
     ['curtain', 9],
     ['murano', 9],
     ['calm', 9],
     ['lantern', 8],
     ['bath', 8],
     ['sheet', 8],
     ['lola', 8],
     ['set/6', 8],
     ['suki', 8],
     ['bank', 8],
     ['fire', 8],
     ['bucket', 8],
     ['umbrella', 8],
     ['cone', 8],
     ['tv', 8],
     ['oval', 8],
     ['flannel', 8],
     ['empire', 8],
     ['jet', 8],
     ['roll', 8],
     ['multicolour', 8],
     ['tile', 8],
     ['jardin', 8],
     ['day', 8],
     ['monkey', 8],
     ['set/3', 8],
     ['frog', 8],
     ['washbag', 8],
     ['danish', 8],
     ['coat', 7],
     ['poppy', 7],
     ['alarm', 7],
     ['billboard', 7],
     ['charlie+lola', 7],
     ['thermometer', 7],
     ['duck', 7],
     ['bread', 7],
     ['fridge', 7],
     ['measure', 7],
     ['candlestick', 7],
     ['cookie', 7],
     ['cutter', 7],
     ['bake', 7],
     ['tote', 7],
     ['decoupage', 7],
     ['sock', 7],
     ['turquoise', 7],
     ['mushroom', 7],
     ['beach', 7],
     ['choice', 7],
     ['place', 7],
     ['size', 7],
     ['pad', 7],
     ['birdhouse', 7],
     ['fun', 7],
     ['ass', 7],
     ['sketchbook', 7],
     ['porcelain', 7],
     ['dec', 7],
     ['passport', 7],
     ['pudding', 7],
     ['pair', 7],
     ['fan', 7],
     ['midnight', 7],
     ['organiser', 7],
     ['feather', 7],
     ['mobile', 7],
     ['magic', 7],
     ['copper', 7],
     ['twist', 7],
     ['cluster', 7],
     ['mat', 7],
     ['knack', 7],
     ['park', 7],
     ['bicycle', 7],
     ['cosy', 6],
     ['recipe', 6],
     ['game', 6],
     ['airline', 6],
     ['lid', 6],
     ['medina', 6],
     ['lovebird', 6],
     ['disco', 6],
     ['plasters', 6],
     ['holiday', 6],
     ['matches', 6],
     ['invites', 6],
     ['strand', 6],
     ['tassle', 6],
     ['roses', 6],
     ['shoe', 6],
     ['village', 6],
     ['boy', 6],
     ['grip', 6],
     ['pack/2', 6],
     ['stationery', 6],
     ['stamp', 6],
     ['tier', 6],
     ['envelope', 6],
     ['polyester', 6],
     ['filler', 6],
     ['scarf', 6],
     ['cut', 6],
     ['journal', 6],
     ['glove', 6],
     ['col', 6],
     ['spring', 6],
     ['scales', 6],
     ['raffia', 6],
     ['helicopter', 6],
     ['jingle', 6],
     ['planter', 6],
     ['space', 6],
     ['monster', 6],
     ['teatime', 6],
     ['carousel', 6],
     ['lattice', 6],
     ['ice', 6],
     ['white/pink', 6],
     ['m.o.p', 6],
     ['rosebud', 6],
     ['list', 6],
     ['acapulco', 6],
     ['parlour', 6],
     ['landmark', 6],
     ['paris', 5],
     ['night', 5],
     ['chicken', 5],
     ['shoulder', 5],
     ['chalkboard', 5],
     ['cube', 5],
     ['font', 5],
     ['scotty', 5],
     ['sponge', 5],
     ['chest', 5],
     ['snowflake', 5],
     ['car', 5],
     ['angel', 5],
     ['scent', 5],
     ['pattern', 5],
     ["b'fly", 5],
     ['bauble', 5],
     ['advent', 5],
     ['scissor', 5],
     ['ruby', 5],
     ['pears', 5],
     ['pizza', 5],
     ['toast', 5],
     ['chateau', 5],
     ['screen', 5],
     ['straws', 5],
     ['fabric', 5],
     ['teddy', 5],
     ['collage', 5],
     ['base', 5],
     ['folkart', 5],
     ['fuschia', 5],
     ['brush', 5],
     ['lampshade', 5],
     ['badges', 5],
     ['pin', 5],
     ['portrait', 5],
     ['plastic', 5],
     ['teapot', 5],
     ['dairy', 5],
     ['ruler', 5],
     ['post', 5],
     ['posy', 5],
     ['brooch', 5],
     ['gemstone', 5],
     ['fob', 5],
     ['farm', 5],
     ['necklace+bracelet', 5],
     ['camphor', 5],
     ['summer', 5],
     ['pendant', 5],
     ['iron', 5],
     ['fork', 5],
     ['chair', 5],
     ['stone', 5],
     ['windmill', 5],
     ['eau', 5],
     ['stool', 5],
     ['sharpener', 5],
     ['slide', 5],
     ['comb', 5],
     ['bill', 5],
     ['triple', 5],
     ['backpack', 5],
     ['parisienne', 5],
     ['glaze', 5],
     ['villa', 5],
     ['playhouse', 4],
     ['england', 4],
     ['bakelike', 4],
     ['tail', 4],
     ['finish', 4],
     ['gin', 4],
     ['winkie', 4],
     ['ladder', 4],
     ['caravan', 4],
     ['skittles', 4],
     ['carnival', 4],
     ['doughnut', 4],
     ['silk', 4],
     ['cornice', 4],
     ['family', 4],
     ['toilet', 4],
     ['eraser', 4],
     ['top', 4],
     ['bangle', 4],
     ['shopper', 4],
     ['slipper', 4],
     ['chalk', 4],
     ['hairband', 4],
     ['rubber', 4],
     ['canvas', 4],
     ['record', 4],
     ['seed', 4],
     ['leaves', 4],
     ['button', 4],
     ['path', 4],
     ['des', 4],
     ['confetti', 4],
     ['suction', 4],
     ['school', 4],
     ['person', 4],
     ['candlepot', 4],
     ['magazine', 4],
     ['carriage', 4],
     ['pony', 4],
     ['stack', 4],
     ['hut', 4],
     ['key', 4],
     ['ma', 4],
     ['campagne', 4],
     ['frutti', 4],
     ['brocante', 4],
     ['robot', 4],
     ['tassel', 4],
     ['bow', 4],
     ['gnome', 4],
     ['choc', 4],
     ['check', 4],
     ['la', 4],
     ['album', 4],
     ['ashtray', 4],
     ['queen', 4],
     ['sky', 4],
     ['deco', 4],
     ['neckl', 4],
     ['goose', 4],
     ['las', 4],
     ['mould', 4],
     ['origami', 4],
     ['incense/candl', 4],
     ['rococo', 4],
     ['bertie', 4],
     ['chandelier', 4],
     ['st', 4],
     ['george', 4],
     ['tie', 4],
     ['buffalo', 4],
     ['dollcraft', 4],
     ['snow', 4],
     ['bundle', 4],
     ['bon', 4],
     ['crackle', 4],
     ['doorknob', 4],
     ['bonne', 4],
     ['sucker', 4],
     ['forest', 4],
     ['rucksack', 4],
     ['nest', 3],
     ['dot', 3],
     ['princess', 3],
     ['building', 3],
     ['word', 3],
     ['fashion', 3],
     ['head', 3],
     ['seaside', 3],
     ['planet', 3],
     ['s/3', 3],
     ['tool', 3],
     ['calculator', 3],
     ['rain', 3],
     ['rope', 3],
     ['cloth', 3],
     ['ladies', 3],
     ['set/10', 3],
     ['santa', 3],
     ['pirate', 3],
     ['giant', 3],
     ['jazz', 3],
     ['football', 3],
     ['trellis', 3],
     ['pannetone', 3],
     ['rectangle', 3],
     ['cowboy', 3],
     ['marbles', 3],
     ['boom', 3],
     ['dress', 3],
     ['friends', 3],
     ['fair', 3],
     ['cupcake', 3],
     ['spade', 3],
     ['caddy', 3],
     ['chrysanthemum', 3],
     ['montana', 3],
     ['scandinavian', 3],
     ['cakestand', 3],
     ['crate', 3],
     ['platter', 3],
     ['teacup', 3],
     ['slice', 3],
     ['sand', 3],
     ['postcard', 3],
     ['life', 3],
     ['peace', 3],
     ['cocktail', 3],
     ['man', 3],
     ['camouflage', 3],
     ['dragonfly', 3],
     ['shark', 3],
     ['moon', 3],
     ['hammock', 3],
     ['cardholder', 3],
     ['s/4', 3],
     ['mum', 3],
     ['s', 3],
     ['floor', 3],
     ['muff', 3],
     ['headphones', 3],
     ['memo', 3],
     ['pig', 3],
     ['elephant', 3],
     ['rainbow', 3],
     ['maid', 3],
     ['thank', 3],
     ['lilac', 3],
     ['cafe', 3],
     ['hairclip', 3],
     ['chopsticks', 3],
     ['puppet', 3],
     ['wine', 3],
     ['curio', 3],
     ['butter', 3],
     ['cadet', 3],
     ['citronella', 3],
     ['folk', 3],
     ['rest', 3],
     ['rustic', 3],
     ['palmiera', 3],
     ['pack/12', 3],
     ['field', 3],
     ['liners', 3],
     ['mouse', 3],
     ['multi', 3],
     ['cactus', 3],
     ['cockerel', 3],
     ['trowel', 3],
     ['pop', 3],
     ['microwave', 3],
     ['madras', 3],
     ['oil', 3],
     ['tumbler', 3],
     ['brown', 3],
     ['bear', 3],
     ['photoframe', 3],
     ['compact', 3],
     ['neckl.36', 3],
     ['class', 3],
     ['mosaic', 3],
     ['turq', 3],
     ['sundae', 3],
     ['droplet', 3],
     ['beaten', 3],
     ['chalice', 3],
     ['enamel+glass', 3],
     ['diamond', 3],
     ['frangipani', 3],
     ['icon', 3],
     ['crystal+glass', 3],
     ['glass/silver', 3],
     ['hunt', 3],
     ['start', 3],
     ['vint', 3],
     ['lip', 3],
     ['gloss', 3],
     ['marie', 3],
     ['secret', 3],
     ['dispenser', 3],
     ['peony', 3],
     ['willie', 3],
     ['trim', 3],
     ['refectory', 3],
     ['cracker', 3],
     ['hairslide', 3],
     ['linen', 3],
     ['henrietta', 3],
     ['panettone', 3],
     ['cupid', 2],
     ['postage', 2],
     ['wastepaper', 2],
     ['flora', 2],
     ['doorsign', 2],
     ['pc', 2],
     ['lounge', 2],
     ['cloche', 2],
     ['gumball', 2],
     ['snake', 2],
     ['birdcage', 2],
     ['tale', 2],
     ['reindeer', 2],
     ['popcorn', 2],
     ['set/20', 2],
     ['deluxe', 2],
     ['grey', 2],
     ['soldier', 2],
     ['dominoes', 2],
     ['sew', 2],
     ['battery', 2],
     ['basil', 2],
     ['snowmen', 2],
     ['cigar', 2],
     ['hour', 2],
     ['celebration', 2],
     ['set/5', 2],
     ['harmonica', 2],
     ['hldr', 2],
     ['squarecushion', 2],
     ['acrylic', 2],
     ['c/cover', 2],
     ['speaker', 2],
     ['rosie', 2],
     ['sack', 2],
     ['fawn', 2],
     ['vase', 2],
     ['tealight', 2],
     ['present', 2],
     ['cinammon', 2],
     ['playing', 2],
     ['repair', 2],
     ['world', 2],
     ['merry', 2],
     ['level', 2],
     ["p'weight", 2],
     ['confectionery', 2],
     ['joy', 2],
     ['polka', 2],
     ['salt', 2],
     ['crossbones', 2],
     ['cordon', 2],
     ['barrier', 2],
     ['childhood', 2],
     ['memory', 2],
     ['b', 2],
     ['c', 2],
     ['geisha', 2],
     ['yuletide', 2],
     ['shade', 2],
     ['cacti', 2],
     ['marshmallow', 2],
     ['swallows', 2],
     ['m', 2],
     ['ear', 2],
     ['ladybird', 2],
     ['bamboo', 2],
     ['room', 2],
     ['bomb', 2],
     ['stencil', 2],
     ['purdey', 2],
     ['area', 2],
     ['moody', 2],
     ['stripey', 2],
     ['dragon', 2],
     ['army', 2],
     ['bookcover', 2],
     ['glass+bead', 2],
     ['juicy', 2],
     ['creepy', 2],
     ['crawlie', 2],
     ['wash', 2],
     ['clam', 2],
     ['garage', 2],
     ['champagne', 2],
     ['charge', 2],
     ['tinsel', 2],
     ['way', 2],
     ['fishing', 2],
     ['strawbery', 2],
     ['pillar', 2],
     ['cabin', 2],
     ['throw', 2],
     ['salad', 2],
     ['display', 2],
     ['neighbourhood', 2],
     ['witch', 2],
     ['hello', 2],
     ['sailor', 2],
     ['jasmine', 2],
     ['bone', 2],
     ['skirt', 2],
     ['wool', 2],
     ['gardenia', 2],
     ['bouquet', 2],
     ['cast', 2],
     ['lime', 2],
     ['poncho', 2],
     ['flask', 2],
     ['keyring', 2],
     ['petit', 2],
     ['dream', 2],
     ['delight', 2],
     ['silicon', 2],
     ['seat', 2],
     ['rocket', 2],
     ['bitty', 2],
     ['glow', 2],
     ['s/12', 2],
     ['bib', 2],
     ['embroidery', 2],
     ['shower', 2],
     ['cap', 2],
     ['economy', 2],
     ['burner', 2],
     ['paperweight', 2],
     ['travelogue', 2],
     ['sunglasses', 2],
     ['smokey', 2],
     ['sunflower', 2],
     ['pouffe', 2],
     ['masala', 2],
     ['nut', 2],
     ['vegas', 2],
     ['voile', 2],
     ['jungle', 2],
     ['popsicles', 2],
     ['strap', 2],
     ['cd', 2],
     ['island', 2],
     ['grass', 2],
     ['plaque', 2],
     ['popart', 2],
     ['rect', 2],
     ['asst', 2],
     ['opium', 2],
     ['sandlewood', 2],
     ['goblet', 2],
     ['vip', 2],
     ['dad', 2],
     ['garld', 2],
     ['neckl36', 2],
     ['flowe', 2],
     ['charger', 2],
     ['aperitif', 2],
     ['cannister', 2],
     ['steel', 2],
     ['asstd', 2],
     ['blackboard', 2],
     ['fly', 2],
     ['swat', 2],
     ['fruitbowl', 2],
     ['botanical', 2],
     ['groovy', 2],
     ['fluffy', 2],
     ['band', 2],
     ['delphinium', 2],
     ['dolphin', 2],
     ['ladle', 2],
     ['wedding', 2],
     ['bar', 2],
     ['elvis', 2],
     ['resin', 2],
     ['rasta', 2],
     ['hen+chicks', 2],
     ['petals', 2],
     ['hamper', 2],
     ['suede', 2],
     ['flame', 2],
     ['ceramic', 2],
     ['unit', 2],
     ['peach', 2],
     ['gauze', 2],
     ['radio', 2],
     ['jade', 2],
     ['sea', 2],
     ['pole', 2],
     ['monte', 2],
     ['carlo', 2],
     ['pill', 2],
     ['raspberry', 2],
     ['riviera', 2],
     ['bull', 2],
     ['kukui', 2],
     ['coconut', 2],
     ['strainer', 2],
     ['antoinette', 2],
     ['l', 2],
     ['exercise', 2],
     ['bedside', 2],
     ['laurel', 2],
     ['icicle', 2],
     ['gymkhana', 2],
     ['spike', 2],
     ['songbird', 2],
     ['notepad', 2],
     ['ceature', 2],
     ['jampot', 2],
     ['botanique', 2],
     ['makers', 2],
     ['grinder', 2],
     ['dovecote', 2],
     ['saftey', 2],
     ['stop', 2],
     ['licence', 2],
     ['ahoy', 2],
     ['nature', 2],
     ['show', 2],
     ['baseball', 2],
     ['boot', 2],
     ['triobase', 2],
     ['feeder', 2],
     ['street', 2],
     ['cross', 2],
     ['ornament', 1],
     ['bedroom', 1],
     ['teaspoons', 1],
     ['panda', 1],
     ['globe', 1],
     ['set/2', 1],
     ['puzzles', 1],
     ['paint', 1],
     ['love/hate', 1],
     ['sympathy', 1],
     ['tomato', 1],
     ['making', 1],
     ['line', 1],
     ['cook', 1],
     ['monochrome', 1],
     ['fancy', 1],
     ['discount', 1],
     ['chilli', 1],
     ['butterfiles', 1],
     ['homemade', 1],
     ['enfants', 1],
     ['black/blue', 1],
     ['keepsake', 1],
     ['hat', 1],
     ['mice', 1],
     ['skipping', 1],
     ['seventeen', 1],
     ['sideboard', 1],
     ['quilt', 1],
     ['boudoir', 1],
     ['s/6', 1],
     ['reds', 1],
     ['rotator', 1],
     ['gentlemen', 1],
     ['baroquecandlestick', 1],
     ['transfer', 1],
     ['tattoos', 1],
     ['face', 1],
     ['spinning', 1],
     ['candelabra', 1],
     ['guns', 1],
     ['filigree', 1],
     ['bobbles', 1],
     ['parcel', 1],
     ['olivia', 1],
     ['rex', 1],
     ['cash+carry', 1],
     ['pinkwhite', 1],
     ['red/white', 1],
     ['valentine', 1],
     ['snap', 1],
     ['drawerknob', 1],
     ['catch', 1],
     ['namaste', 1],
     ['swagat', 1],
     ['s/15', 1],
     ['wc', 1],
     ['washroom', 1],
     ['dia', 1],
     ['aid', 1],
     ['monsoon', 1],
     ['mittens', 1],
     ['provence', 1],
     ['longboard', 1],
     ['shirt', 1],
     ['foxy', 1],
     ['gents', 1],
     ["'n", 1],
     ['grow', 1],
     ['champion', 1],
     ['hi', 1],
     ['gazebo', 1],
     ['market', 1],
     ['bed', 1],
     ['message', 1],
     ['rosemary', 1],
     ['chives', 1],
     ['parsley', 1],
     ['thyme', 1],
     ['foil', 1],
     ['season', 1],
     ['squeezer', 1],
     ['licorice', 1],
     ['pistachio', 1],
     ['gecko', 1],
     ['lizard', 1],
     ['engine/car', 1],
     ['psychedelic', 1],
     ['flowerpower', 1],
     ['rounders', 1],
     ['taper', 1],
     ['stopper', 1],
     ['junk', 1],
     ['mail', 1],
     ['finger', 1],
     ['beurre', 1],
     ...]




```python
#Let's plot the keywords

lista = sorted(list_products, key = lambda x:x[1], reverse = True)
#_______________________________
plt.rc('font', weight='normal')
fig, ax = plt.subplots(figsize=(7, 25))
y_axis = [i[1] for i in lista[:125]]
x_axis = [k for k,i in enumerate(lista[:125])]
x_label = [i[0] for i in lista[:125]]
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 12)
plt.yticks(x_axis, x_label)
plt.xlabel("Occurences", fontsize = 18, labelpad = 10)
ax.barh(x_axis, y_axis, align = 'center', color = '#3192B3')
ax = plt.gca()
ax.invert_yaxis()
#_______________________________________________________________________________________
plt.title("Keywords",
         # bbox={'facecolor':'k', 'pad':5},
          color='black',fontsize = 25)
plt.show()
```


    
![png](output_36_0.png)
    



```python
#Now we can divide the products into categories

list_products = []
for k,v in count_keywords.items():
    word = keywords_select[k]
    if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
    if len(word) < 3 or v < 13: continue
    if ('+' in word) or ('/' in word): continue
    list_products.append([word, v])
#______________________________________________________    
list_products.sort(key = lambda x:x[1], reverse = True)
print('words kept:', len(list_products))

liste_produits = df_cleaned['Description'].unique()
X = pd.DataFrame()
for key, occurence in list_products:
    X.loc[:, key] = list(map(lambda x:int(key.upper() in x), liste_produits))
```

    words kept: 193
    


```python
threshold = [0, 1, 2, 3, 5, 10]
label_col = []
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])
    label_col.append(col)
    X.loc[:, col] = 0

for i, prod in enumerate(liste_produits):
    prix = df_cleaned[ df_cleaned['Description'] == prod]['UnitPrice'].mean()
    j = 0
    while prix > threshold[j]:
        j+=1
        if j == len(threshold): break
    X.loc[i, label_col[j-1]] = 1
```


```python
print("{:<8} {:<20} \n".format('gamma', 'no produits') + 20*'-')
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])    
    print("{:<10}  {:<20}".format(col, X.loc[:, col].sum()))
    

```

    gamma    no produits          
    --------------------
    0<.<1       964                 
    1<.<2       1009                
    2<.<3       673                 
    3<.<5       606                 
    5<.<10      470                 
    .>10        156                 
    


```python
matrix = X
for n_clusters in range(3,10):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
```

    For n_clusters = 3 The average silhouette_score is : 0.10158702596012364
    For n_clusters = 4 The average silhouette_score is : 0.1268004588393788
    For n_clusters = 5 The average silhouette_score is : 0.14562442905455303
    For n_clusters = 6 The average silhouette_score is : 0.1450621616291374
    For n_clusters = 7 The average silhouette_score is : 0.14959907226711688
    For n_clusters = 8 The average silhouette_score is : 0.14689677795453093
    For n_clusters = 9 The average silhouette_score is : 0.13724506930089098
    


```python
n_clusters = 5
silhouette_avg = -1
while silhouette_avg < 0.145:
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    
    #km = kmodes.KModes(n_clusters = n_clusters, init='Huang', n_init=2, verbose=0)
    #clusters = km.fit_predict(matrix)
    #silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    
pd.Series(clusters).value_counts()
```

    For n_clusters = 5 The average silhouette_score is : 0.14708700459493795
    




    4    1009
    1     964
    3     762
    0     673
    2     470
    dtype: int64




```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>heart</th>
      <th>vintage</th>
      <th>set</th>
      <th>bag</th>
      <th>box</th>
      <th>glass</th>
      <th>christmas</th>
      <th>design</th>
      <th>candle</th>
      <th>holder</th>
      <th>flower</th>
      <th>decoration</th>
      <th>metal</th>
      <th>retrospot</th>
      <th>card</th>
      <th>necklac</th>
      <th>paper</th>
      <th>art</th>
      <th>silver</th>
      <th>cake</th>
      <th>polkadot</th>
      <th>cover</th>
      <th>mug</th>
      <th>tin</th>
      <th>wrap</th>
      <th>sign</th>
      <th>pack</th>
      <th>egg</th>
      <th>bracelet</th>
      <th>star</th>
      <th>bowl</th>
      <th>mini</th>
      <th>tea</th>
      <th>garden</th>
      <th>wall</th>
      <th>ivory</th>
      <th>cushion</th>
      <th>frame</th>
      <th>mirror</th>
      <th>cream</th>
      <th>home</th>
      <th>gift</th>
      <th>earrings</th>
      <th>ring</th>
      <th>bird</th>
      <th>bottle</th>
      <th>clock</th>
      <th>paisley</th>
      <th>letter</th>
      <th>party</th>
      <th>...</th>
      <th>stripe</th>
      <th>shape</th>
      <th>parasol</th>
      <th>cherry</th>
      <th>travel</th>
      <th>doilies</th>
      <th>biscuit</th>
      <th>regency</th>
      <th>hen</th>
      <th>jigsaw</th>
      <th>rack</th>
      <th>coaster</th>
      <th>money</th>
      <th>cottage</th>
      <th>incense</th>
      <th>crystal</th>
      <th>medium</th>
      <th>dish</th>
      <th>image</th>
      <th>picture</th>
      <th>wooden</th>
      <th>piece</th>
      <th>bathroom</th>
      <th>girl</th>
      <th>fruit</th>
      <th>childs</th>
      <th>pan</th>
      <th>food</th>
      <th>orbit</th>
      <th>alphabet</th>
      <th>toadstool</th>
      <th>time</th>
      <th>milk</th>
      <th>funky</th>
      <th>tidy</th>
      <th>plant</th>
      <th>diner</th>
      <th>house</th>
      <th>square</th>
      <th>craft</th>
      <th>lace</th>
      <th>jewel</th>
      <th>wallet</th>
      <th>point</th>
      <th>0&lt;.&lt;1</th>
      <th>1&lt;.&lt;2</th>
      <th>2&lt;.&lt;3</th>
      <th>3&lt;.&lt;5</th>
      <th>5&lt;.&lt;10</th>
      <th>.&gt;10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 199 columns</p>
</div>




```python
liste = pd.DataFrame(liste_produits)
liste_words = [word for (word, occurence) in list_products]

occurence = [dict() for _ in range(n_clusters)]

for i in range(n_clusters):
    liste_cluster = liste.loc[clusters == i]
    for word in liste_words:
        if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']: continue
        occurence[i][word] = sum(liste_cluster.loc[:, 0].str.contains(word.upper()))
```


```python
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)
#________________________________________________________________________
def make_wordcloud(liste, increment):
    ax1 = fig.add_subplot(4,2,increment)
    words = dict()
    trunc_occurences = liste[0:150]
    for s in trunc_occurences:
        words[s[0]] = s[1]
    #________________________________________________________
    wordcloud = WordCloud(width=1000,height=400, background_color='lightgrey', 
                          max_words=1628,relative_scaling=1,
                          color_func = random_color_func,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    ax1.imshow(wordcloud, interpolation="bilinear")
    ax1.axis('off')
    plt.title('cluster nº{}'.format(increment-1))
#________________________________________________________________________
fig = plt.figure(1, figsize=(14,14))
color = [0, 160, 130, 95, 280, 40, 330, 110, 25]
for i in range(n_clusters):
    list_cluster_occurences = occurence[i]

    tone = color[i] # define the color of the words
    liste = []
    for key, value in list_cluster_occurences.items():
        liste.append([key, value])
    liste.sort(key = lambda x:x[1], reverse = True)
    make_wordcloud(liste, i+1)    
```


    
![png](output_44_0.png)
    



```python
corresp = dict()
for key, val in zip (liste_produits, clusters):
    corresp[key] = val 
#__________________________________________________________________________
df_cleaned['categ_product'] = df_cleaned.loc[:, 'Description'].map(corresp)
```

<a class="anchor" id="pre"></a>
# 4. Data Engineering


```python
#Let's use a different kind of one hot encoder, instead of 1, let's put in the amount spent
for i in range(5):
    col = 'categ_' + str(i)        
    df_temp = df_cleaned[df_cleaned['categ_product'] == i]
    price_temp = df_temp['UnitPrice'] * (df_temp['Quantity'] - df_temp['QuantityCanceled'])
    price_temp = price_temp.apply(lambda x:x if x > 0 else 0)
    df_cleaned.loc[:, col] = price_temp
    df_cleaned[col].fillna(0, inplace = True)
#__________________________________________________________________________________________________
df_cleaned[['InvoiceNo', 'Description', 'categ_product', 'categ_0', 'categ_1', 'categ_2', 'categ_3','categ_4']].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>InvoiceNo</th>
      <th>Description</th>
      <th>categ_product</th>
      <th>categ_0</th>
      <th>categ_1</th>
      <th>categ_2</th>
      <th>categ_3</th>
      <th>categ_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>0</td>
      <td>15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>WHITE METAL LANTERN</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.34</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536365</td>
      <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.34</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536365</td>
      <td>RED WOOLLY HOTTIE WHITE HEART.</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.34</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>536365</td>
      <td>SET 7 BABUSHKA NESTING BOXES</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>15.3</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>536365</td>
      <td>GLASS STAR FROSTED T-LIGHT HOLDER</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25.50</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>536366</td>
      <td>HAND WARMER UNION JACK</td>
      <td>0</td>
      <td>11.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>536366</td>
      <td>HAND WARMER RED POLKA DOT</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>11.10</td>
    </tr>
    <tr>
      <th>9</th>
      <td>536367</td>
      <td>ASSORTED COLOUR BIRD ORNAMENT</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>54.08</td>
    </tr>
  </tbody>
</table>
</div>




```python
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
basket_price = temp.rename(columns = {'TotalPrice':'Basket Price'})
#____________________________________________________________
# pourcentage du prix de la commande / categorie de produit
for i in range(5):
    col = 'categ_' + str(i)
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()
    basket_price[col] = temp[col] 
#_____________________
# date de la commande
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
#______________________________________
# selection des entrées significatives:
basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID', ascending = True).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>InvoiceNo</th>
      <th>Basket Price</th>
      <th>categ_0</th>
      <th>categ_1</th>
      <th>categ_2</th>
      <th>categ_3</th>
      <th>categ_4</th>
      <th>InvoiceDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>12347</td>
      <td>537626</td>
      <td>711.79</td>
      <td>83.40</td>
      <td>23.40</td>
      <td>124.44</td>
      <td>293.35</td>
      <td>187.2</td>
      <td>2010-12-07 14:57:00.000001024</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12347</td>
      <td>542237</td>
      <td>475.39</td>
      <td>53.10</td>
      <td>84.34</td>
      <td>0.00</td>
      <td>207.45</td>
      <td>130.5</td>
      <td>2011-01-26 14:29:59.999999744</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12347</td>
      <td>549222</td>
      <td>636.25</td>
      <td>71.10</td>
      <td>81.00</td>
      <td>0.00</td>
      <td>153.25</td>
      <td>330.9</td>
      <td>2011-04-07 10:42:59.999999232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12347</td>
      <td>556201</td>
      <td>382.52</td>
      <td>78.06</td>
      <td>41.40</td>
      <td>19.90</td>
      <td>168.76</td>
      <td>74.4</td>
      <td>2011-06-09 13:01:00.000000256</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12347</td>
      <td>562032</td>
      <td>584.91</td>
      <td>119.70</td>
      <td>61.30</td>
      <td>97.80</td>
      <td>196.41</td>
      <td>109.7</td>
      <td>2011-08-02 08:48:00.000000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Investigating data for each customer
transactions_per_user=basket_price.groupby(by=['CustomerID'])['Basket Price'].agg(['count','min','max','mean','sum'])
for i in range(5):
    col = 'categ_{}'.format(i)
    transactions_per_user.loc[:,col] = basket_price.groupby(by=['CustomerID'])[col].sum() /\
                                            transactions_per_user['sum']*100

transactions_per_user.reset_index(drop = False, inplace = True)
basket_price.groupby(by=['CustomerID'])['categ_0'].sum()
transactions_per_user.sort_values('CustomerID', ascending = True).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>count</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>sum</th>
      <th>categ_0</th>
      <th>categ_1</th>
      <th>categ_2</th>
      <th>categ_3</th>
      <th>categ_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12347</td>
      <td>7</td>
      <td>224.82</td>
      <td>1294.32</td>
      <td>615.714286</td>
      <td>4310.00</td>
      <td>20.805104</td>
      <td>11.237123</td>
      <td>7.604176</td>
      <td>33.977726</td>
      <td>26.375870</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12348</td>
      <td>4</td>
      <td>227.44</td>
      <td>892.80</td>
      <td>449.310000</td>
      <td>1797.24</td>
      <td>0.000000</td>
      <td>38.016069</td>
      <td>0.000000</td>
      <td>20.030714</td>
      <td>41.953217</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12349</td>
      <td>1</td>
      <td>1757.55</td>
      <td>1757.55</td>
      <td>1757.550000</td>
      <td>1757.55</td>
      <td>12.245455</td>
      <td>4.513101</td>
      <td>20.389178</td>
      <td>36.346050</td>
      <td>26.506216</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12350</td>
      <td>1</td>
      <td>334.40</td>
      <td>334.40</td>
      <td>334.400000</td>
      <td>334.40</td>
      <td>27.900718</td>
      <td>11.692584</td>
      <td>0.000000</td>
      <td>11.961722</td>
      <td>48.444976</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12352</td>
      <td>7</td>
      <td>144.35</td>
      <td>840.30</td>
      <td>340.815714</td>
      <td>2385.71</td>
      <td>4.071325</td>
      <td>1.299404</td>
      <td>14.691643</td>
      <td>64.232451</td>
      <td>15.705178</td>
    </tr>
  </tbody>
</table>
</div>




```python
#How much time has passed from the first purchase to the last one?
last_date = basket_price['InvoiceDate'].max().date()

first_registration = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
last_purchase      = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].max())

test  = first_registration.applymap(lambda x:(last_date - x.date()).days)
test2 = last_purchase.applymap(lambda x:(last_date - x.date()).days)

transactions_per_user.loc[:, 'LastPurchase'] = test2.reset_index(drop = False)['InvoiceDate']
transactions_per_user.loc[:, 'FirstPurchase'] = test.reset_index(drop = False)['InvoiceDate']

transactions_per_user.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>count</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>sum</th>
      <th>categ_0</th>
      <th>categ_1</th>
      <th>categ_2</th>
      <th>categ_3</th>
      <th>categ_4</th>
      <th>LastPurchase</th>
      <th>FirstPurchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12347</td>
      <td>7</td>
      <td>224.82</td>
      <td>1294.32</td>
      <td>615.714286</td>
      <td>4310.00</td>
      <td>20.805104</td>
      <td>11.237123</td>
      <td>7.604176</td>
      <td>33.977726</td>
      <td>26.375870</td>
      <td>2</td>
      <td>367</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12348</td>
      <td>4</td>
      <td>227.44</td>
      <td>892.80</td>
      <td>449.310000</td>
      <td>1797.24</td>
      <td>0.000000</td>
      <td>38.016069</td>
      <td>0.000000</td>
      <td>20.030714</td>
      <td>41.953217</td>
      <td>75</td>
      <td>358</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12349</td>
      <td>1</td>
      <td>1757.55</td>
      <td>1757.55</td>
      <td>1757.550000</td>
      <td>1757.55</td>
      <td>12.245455</td>
      <td>4.513101</td>
      <td>20.389178</td>
      <td>36.346050</td>
      <td>26.506216</td>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12350</td>
      <td>1</td>
      <td>334.40</td>
      <td>334.40</td>
      <td>334.400000</td>
      <td>334.40</td>
      <td>27.900718</td>
      <td>11.692584</td>
      <td>0.000000</td>
      <td>11.961722</td>
      <td>48.444976</td>
      <td>310</td>
      <td>310</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12352</td>
      <td>7</td>
      <td>144.35</td>
      <td>840.30</td>
      <td>340.815714</td>
      <td>2385.71</td>
      <td>4.071325</td>
      <td>1.299404</td>
      <td>14.691643</td>
      <td>64.232451</td>
      <td>15.705178</td>
      <td>36</td>
      <td>296</td>
    </tr>
  </tbody>
</table>
</div>




```python
n1 = transactions_per_user[transactions_per_user['count'] == 1].shape[0]
n2 = transactions_per_user.shape[0]
print("Number of customers with one purchase: {:<2}/{:<5} ({:<2.2f}%)".format(n1,n2,n1/n2*100))
```

    Number of customers with one purchase: 1489/4327  (34.41%)
    


```python
transactions_per_user['recurrent'] = np.where(transactions_per_user['count'] == 1,0,
                                              transactions_per_user['FirstPurchase'] - transactions_per_user['LastPurchase'] )
temp = transactions_per_user[['CustomerID','count','recurrent','sum']]
df2 = pd.merge(df_cleaned, temp, on='CustomerID')
```


```python
df2.drop(['InvoiceNo','Description','UnitPrice','InvoiceDate','StockCode','Quantity','recurrent'],axis=1,inplace=True)
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>QuantityCanceled</th>
      <th>TotalPrice</th>
      <th>categ_product</th>
      <th>categ_0</th>
      <th>categ_1</th>
      <th>categ_2</th>
      <th>categ_3</th>
      <th>categ_4</th>
      <th>count</th>
      <th>sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17850</td>
      <td>United Kingdom</td>
      <td>0</td>
      <td>15.30</td>
      <td>0</td>
      <td>15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>34</td>
      <td>5322.84</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17850</td>
      <td>United Kingdom</td>
      <td>0</td>
      <td>20.34</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.34</td>
      <td>0.0</td>
      <td>34</td>
      <td>5322.84</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17850</td>
      <td>United Kingdom</td>
      <td>0</td>
      <td>22.00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22.00</td>
      <td>0.0</td>
      <td>34</td>
      <td>5322.84</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17850</td>
      <td>United Kingdom</td>
      <td>0</td>
      <td>20.34</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.34</td>
      <td>0.0</td>
      <td>34</td>
      <td>5322.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17850</td>
      <td>United Kingdom</td>
      <td>0</td>
      <td>20.34</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.34</td>
      <td>0.0</td>
      <td>34</td>
      <td>5322.84</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers = df2.groupby(['CustomerID','Country'], as_index = False).sum()
customers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>QuantityCanceled</th>
      <th>TotalPrice</th>
      <th>categ_product</th>
      <th>categ_0</th>
      <th>categ_1</th>
      <th>categ_2</th>
      <th>categ_3</th>
      <th>categ_4</th>
      <th>count</th>
      <th>sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12347</td>
      <td>Iceland</td>
      <td>0</td>
      <td>4310.00</td>
      <td>431</td>
      <td>896.70</td>
      <td>484.32</td>
      <td>327.74</td>
      <td>1464.44</td>
      <td>1136.80</td>
      <td>1274</td>
      <td>784420.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12348</td>
      <td>Finland</td>
      <td>0</td>
      <td>1797.24</td>
      <td>60</td>
      <td>0.00</td>
      <td>683.24</td>
      <td>0.00</td>
      <td>360.00</td>
      <td>754.00</td>
      <td>124</td>
      <td>55714.44</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12349</td>
      <td>Italy</td>
      <td>0</td>
      <td>1757.55</td>
      <td>174</td>
      <td>215.22</td>
      <td>79.32</td>
      <td>358.35</td>
      <td>638.80</td>
      <td>465.86</td>
      <td>73</td>
      <td>128301.15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12350</td>
      <td>Norway</td>
      <td>0</td>
      <td>334.40</td>
      <td>42</td>
      <td>93.30</td>
      <td>39.10</td>
      <td>0.00</td>
      <td>40.00</td>
      <td>162.00</td>
      <td>17</td>
      <td>5684.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12352</td>
      <td>Norway</td>
      <td>63</td>
      <td>2385.71</td>
      <td>233</td>
      <td>97.13</td>
      <td>31.00</td>
      <td>350.50</td>
      <td>1532.40</td>
      <td>374.68</td>
      <td>595</td>
      <td>202785.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
countries = customers.groupby('Country', as_index = False).sum().sort_values(by = 'count', ascending = False)

countries['country'] = [len(countries['Country'])-i for i in range(len(countries['Country']))]
countries.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>QuantityCanceled</th>
      <th>TotalPrice</th>
      <th>categ_product</th>
      <th>categ_0</th>
      <th>categ_1</th>
      <th>categ_2</th>
      <th>categ_3</th>
      <th>categ_4</th>
      <th>count</th>
      <th>sum</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>United Kingdom</td>
      <td>157503</td>
      <td>6828858.554</td>
      <td>732976</td>
      <td>1697453.96</td>
      <td>795641.373</td>
      <td>1125696.38</td>
      <td>1643355.531</td>
      <td>1573919.02</td>
      <td>5709120</td>
      <td>2.674117e+09</td>
      <td>37</td>
    </tr>
    <tr>
      <th>10</th>
      <td>EIRE</td>
      <td>4040</td>
      <td>254839.400</td>
      <td>15491</td>
      <td>54241.10</td>
      <td>28298.560</td>
      <td>38245.05</td>
      <td>83595.990</td>
      <td>50893.21</td>
      <td>1206296</td>
      <td>9.350347e+08</td>
      <td>36</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Netherlands</td>
      <td>328</td>
      <td>284731.140</td>
      <td>5163</td>
      <td>77252.91</td>
      <td>38328.590</td>
      <td>11389.13</td>
      <td>55365.920</td>
      <td>102600.99</td>
      <td>152940</td>
      <td>5.824931e+08</td>
      <td>35</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Germany</td>
      <td>1607</td>
      <td>223435.730</td>
      <td>19654</td>
      <td>43909.03</td>
      <td>26312.890</td>
      <td>27540.51</td>
      <td>71117.920</td>
      <td>54568.13</td>
      <td>87841</td>
      <td>4.841451e+07</td>
      <td>34</td>
    </tr>
    <tr>
      <th>13</th>
      <td>France</td>
      <td>1580</td>
      <td>196763.140</td>
      <td>18137</td>
      <td>42625.55</td>
      <td>25175.030</td>
      <td>20322.29</td>
      <td>57808.690</td>
      <td>55083.13</td>
      <td>78660</td>
      <td>4.889598e+07</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>QuantityCanceled</th>
      <th>TotalPrice</th>
      <th>categ_product</th>
      <th>categ_0</th>
      <th>categ_1</th>
      <th>categ_2</th>
      <th>categ_3</th>
      <th>categ_4</th>
      <th>count</th>
      <th>sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12347</td>
      <td>Iceland</td>
      <td>0</td>
      <td>4310.00</td>
      <td>431</td>
      <td>896.70</td>
      <td>484.32</td>
      <td>327.74</td>
      <td>1464.44</td>
      <td>1136.80</td>
      <td>1274</td>
      <td>784420.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12348</td>
      <td>Finland</td>
      <td>0</td>
      <td>1797.24</td>
      <td>60</td>
      <td>0.00</td>
      <td>683.24</td>
      <td>0.00</td>
      <td>360.00</td>
      <td>754.00</td>
      <td>124</td>
      <td>55714.44</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12349</td>
      <td>Italy</td>
      <td>0</td>
      <td>1757.55</td>
      <td>174</td>
      <td>215.22</td>
      <td>79.32</td>
      <td>358.35</td>
      <td>638.80</td>
      <td>465.86</td>
      <td>73</td>
      <td>128301.15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12350</td>
      <td>Norway</td>
      <td>0</td>
      <td>334.40</td>
      <td>42</td>
      <td>93.30</td>
      <td>39.10</td>
      <td>0.00</td>
      <td>40.00</td>
      <td>162.00</td>
      <td>17</td>
      <td>5684.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12352</td>
      <td>Norway</td>
      <td>63</td>
      <td>2385.71</td>
      <td>233</td>
      <td>97.13</td>
      <td>31.00</td>
      <td>350.50</td>
      <td>1532.40</td>
      <td>374.68</td>
      <td>595</td>
      <td>202785.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers = pd.merge(customers,countries[['Country','country']],on = 'Country')
customers.drop('Country',axis =1,inplace=True)
customers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>QuantityCanceled</th>
      <th>TotalPrice</th>
      <th>categ_product</th>
      <th>categ_0</th>
      <th>categ_1</th>
      <th>categ_2</th>
      <th>categ_3</th>
      <th>categ_4</th>
      <th>count</th>
      <th>sum</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12347</td>
      <td>0</td>
      <td>4310.00</td>
      <td>431</td>
      <td>896.70</td>
      <td>484.32</td>
      <td>327.74</td>
      <td>1464.44</td>
      <td>1136.80</td>
      <td>1274</td>
      <td>784420.00</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12348</td>
      <td>0</td>
      <td>1797.24</td>
      <td>60</td>
      <td>0.00</td>
      <td>683.24</td>
      <td>0.00</td>
      <td>360.00</td>
      <td>754.00</td>
      <td>124</td>
      <td>55714.44</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12375</td>
      <td>1</td>
      <td>455.42</td>
      <td>17</td>
      <td>268.32</td>
      <td>25.50</td>
      <td>31.80</td>
      <td>129.80</td>
      <td>0.00</td>
      <td>34</td>
      <td>7742.14</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12405</td>
      <td>0</td>
      <td>1710.39</td>
      <td>99</td>
      <td>354.61</td>
      <td>213.72</td>
      <td>360.40</td>
      <td>625.70</td>
      <td>155.96</td>
      <td>54</td>
      <td>92361.06</td>
      <td>26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12428</td>
      <td>33</td>
      <td>7877.20</td>
      <td>561</td>
      <td>1248.69</td>
      <td>886.99</td>
      <td>1317.95</td>
      <td>3418.88</td>
      <td>1004.69</td>
      <td>2646</td>
      <td>2315896.80</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import normalize
IDs = customers['CustomerID']
customers.drop(['CustomerID','categ_product'],axis=1,inplace = True)

```


```python
sum1 = customers['sum']
customers.drop(['count','sum'],axis=1,inplace = True)
#customers['QuantityCanceled'] = np.where(customers['QuantityCanceled']==0,0,1)
```


```python
customers_n = normalize(customers) 
```

<a class="anchor" id="clus"></a>
# 5. Clustering


```python
#Clustering with k means
cluster_range = range(2, 15)

km_WCSS_scores = []
km_sil_scores = []  # sil is nick name for silhouette

for kay in cluster_range:
    kh_mins = KMeans(n_clusters=kay)
    y_tmp = kh_mins.fit_predict(customers_n)  # y_tmp are the cluster results based on Xs (Stan-dar-dised)
    km_WCSS_scores.append(kh_mins.inertia_)  # that's how you get the score
    km_sil_scores.append(silhouette_score(customers, y_tmp))
```

<a class="anchor" id="elb"></a>
# 5.1 Elbow method


```python
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.title('WCSS')
plt.plot(cluster_range, km_WCSS_scores, marker='*')


plt.subplot(1, 2, 2)
plt.title('silhouette scores')
plt.plot(cluster_range, km_sil_scores, marker='*')
```




    [<matplotlib.lines.Line2D at 0x21481d72700>]




    
![png](output_64_1.png)
    



```python
kmeans = KMeans(n_clusters= 3).fit(customers_n)
```


```python
pca_test = PCA().fit(customers_n)
plt.plot(pca_test.explained_variance_ratio_.cumsum())
# the following plot tells us the variance we manage to get aganist the number of dimensions.
```




    [<matplotlib.lines.Line2D at 0x21481df5f70>]




    
![png](output_66_1.png)
    


<a class="anchor" id="pca"></a>
# 5.2 PCA


```python
pca_2d = PCA(n_components=2)
pca_3d = PCA(n_components=3)
Xn_2d = pca_2d.fit_transform(customers_n)
Xn_3d = pca_3d.fit_transform(customers_n)

```


```python
plt.figure(figsize=(20, 9))
x_, y_ = Xn_2d.T

plt.subplot(1, 2, 1)
sns.scatterplot(x=x_, y=y_)

plt.subplot(1, 2, 2)
sns.scatterplot(x=x_, y=y_, hue=kmeans.labels_)
```




    <AxesSubplot:>




    
![png](output_69_1.png)
    



```python
pd.DataFrame(kmeans.labels_).value_counts()
```




    1    1662
    2    1612
    0    1061
    dtype: int64




```python
import plotly.express as px

x_, y_, z_ = Xn_3d.T
fig = px.scatter_3d(x=x_, y=y_, z=z_, color=kmeans.labels_)
fig.show()
```


<div>                            <div id="d3a29ddc-778c-4439-bd8d-f19917f79026" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("d3a29ddc-778c-4439-bd8d-f19917f79026")) {                    Plotly.newPlot(                        "d3a29ddc-778c-4439-bd8d-f19917f79026",                        [{"hovertemplate":"x=%{x}<br>y=%{y}<br>z=%{z}<br>color=%{marker.color}<extra></extra>","legendgroup":"","marker":{"color":[1,2,0,1,1,1,0,1,1,1,1,1,1,1,2,1,1,1,1,1,2,0,1,2,1,1,2,2,1,1,1,1,2,1,1,1,1,1,2,2,1,1,2,1,2,2,2,1,1,1,2,1,1,1,1,2,1,2,0,2,2,2,0,2,1,1,1,2,1,2,1,0,0,1,1,1,1,2,0,2,2,2,1,1,0,1,0,1,1,1,2,1,2,1,1,1,1,2,1,1,1,2,1,1,2,0,2,0,1,1,1,0,1,1,1,2,1,0,2,1,1,2,2,1,2,1,1,2,1,2,0,0,2,1,2,1,1,2,1,0,2,0,1,0,1,0,1,2,1,2,1,2,2,1,1,2,2,2,2,1,1,2,1,1,2,2,1,2,1,0,0,0,1,0,0,2,0,0,1,0,1,2,0,1,2,1,2,1,0,1,1,0,0,1,1,2,0,0,2,2,1,2,2,2,2,2,2,0,2,0,2,2,2,1,0,1,2,2,2,1,1,0,2,0,0,1,0,2,1,0,0,2,2,1,2,2,1,1,1,1,2,1,2,2,2,1,2,1,1,1,2,1,1,0,1,1,1,1,1,2,1,2,1,0,1,2,2,2,1,1,1,2,2,2,1,2,1,2,2,1,0,2,0,0,2,0,1,2,0,0,1,0,1,2,1,1,1,1,1,1,2,2,2,1,2,1,0,1,1,1,0,1,2,0,1,0,0,0,2,2,0,2,2,2,2,1,2,0,0,2,0,2,1,2,1,2,1,2,0,2,1,2,1,2,2,2,1,1,2,1,2,0,0,2,1,1,1,2,1,2,1,1,1,1,0,1,0,2,1,1,2,2,2,1,2,0,1,1,2,1,1,0,1,0,2,1,2,2,0,1,0,0,1,0,1,2,0,2,1,0,1,1,0,0,2,2,0,1,0,2,2,2,2,2,1,2,0,0,1,1,2,2,0,0,2,0,2,2,2,0,1,1,1,1,2,0,1,2,2,0,2,2,2,1,2,2,2,1,1,0,0,1,1,1,2,1,2,2,0,1,2,0,1,0,2,2,2,0,2,2,1,1,1,1,0,2,0,2,0,0,0,1,2,1,2,0,0,1,0,1,0,1,0,1,2,1,1,2,2,1,2,1,1,1,0,1,2,1,1,0,2,2,2,2,2,2,1,1,2,1,0,1,1,1,0,1,2,0,1,2,2,2,1,1,1,0,2,0,1,1,0,1,1,0,1,1,0,0,2,2,2,1,1,0,1,1,0,1,1,1,0,2,0,0,0,1,1,1,2,2,2,1,1,2,1,2,1,1,2,2,1,0,2,2,2,2,1,1,2,2,2,1,1,2,1,2,1,0,2,2,2,2,0,1,0,1,1,2,2,2,2,1,0,0,0,0,1,1,1,1,1,2,1,1,1,2,2,2,2,0,2,2,2,2,0,1,2,1,1,1,2,1,2,2,0,2,1,0,1,1,2,2,1,1,2,2,2,1,2,0,2,2,2,0,1,2,0,0,2,1,1,0,2,0,0,0,0,1,1,1,2,1,2,0,2,1,0,1,1,1,2,2,0,2,2,1,0,1,1,1,2,0,2,2,1,2,0,1,2,1,2,1,2,2,0,2,1,0,1,2,2,2,1,0,2,2,1,1,2,2,1,0,2,1,1,2,2,1,0,1,0,1,1,1,2,2,1,1,2,1,0,0,2,2,0,0,2,1,1,0,1,1,1,0,2,1,2,1,1,2,2,0,1,1,1,1,1,2,2,2,1,2,1,0,1,0,2,2,1,2,2,1,0,1,1,1,0,2,0,1,1,0,2,0,1,0,2,1,2,1,1,1,0,1,0,2,2,1,2,2,0,2,0,0,1,1,0,2,0,0,1,2,2,2,1,2,1,1,0,2,0,0,1,0,1,0,2,2,1,1,2,1,2,1,0,2,2,1,0,2,0,1,2,1,0,1,2,0,0,1,2,2,1,2,2,1,2,2,2,2,0,2,2,2,2,1,1,1,1,0,2,1,1,0,2,2,0,1,1,1,1,0,0,0,1,1,0,2,2,0,0,0,1,0,2,2,1,1,2,0,0,1,2,1,0,1,0,2,1,0,2,1,2,1,1,1,2,0,1,0,1,0,0,2,2,1,1,2,1,0,0,2,2,1,2,2,0,2,1,0,1,2,1,0,0,2,1,0,2,2,1,0,0,1,0,2,2,2,0,0,0,2,1,0,1,2,2,2,1,1,0,2,0,0,1,2,2,2,0,1,0,0,2,2,1,2,1,1,2,2,1,2,2,2,1,2,2,2,2,1,2,2,2,1,1,1,0,1,1,1,1,2,1,1,1,0,2,1,1,2,2,2,1,0,1,0,2,2,2,1,0,0,1,0,2,1,0,0,1,0,0,1,0,1,2,1,1,0,0,0,2,2,1,0,2,2,0,0,2,2,0,1,1,0,2,1,0,0,1,1,1,2,0,2,1,2,2,1,2,1,0,0,0,2,2,0,0,1,1,2,0,0,1,2,2,1,0,1,0,0,0,0,1,1,2,1,2,1,2,0,1,2,0,1,2,1,2,1,1,2,1,2,1,2,1,0,0,2,2,2,2,0,0,2,1,0,2,1,0,0,1,1,2,0,1,1,2,2,2,2,2,2,1,0,2,1,2,0,2,1,2,1,1,1,1,0,1,1,2,0,2,0,0,2,2,1,2,0,1,0,1,0,0,2,2,2,0,2,2,1,1,2,0,2,0,2,2,1,2,0,2,2,2,1,2,2,1,2,1,0,0,1,1,1,0,2,2,1,1,2,2,1,2,1,0,2,1,2,2,2,1,0,2,2,1,1,2,2,0,0,2,0,1,1,1,2,2,2,2,1,2,2,1,2,1,1,2,1,2,2,0,2,1,0,0,1,2,2,1,1,2,0,2,1,2,0,1,1,1,2,2,2,2,2,0,0,2,2,2,2,1,0,2,2,1,2,1,1,2,1,1,0,2,2,1,1,1,2,2,0,1,1,0,0,1,1,1,0,1,0,1,2,1,1,1,1,0,2,0,0,1,2,0,1,1,2,1,1,2,2,0,1,0,2,0,2,1,1,0,0,1,2,2,1,1,2,0,1,0,2,1,2,2,1,1,2,0,1,1,0,1,2,2,0,1,0,0,1,2,0,2,1,2,1,1,1,2,2,0,2,1,0,1,1,2,2,2,1,1,2,1,0,0,2,2,1,2,2,1,0,1,2,0,0,2,2,0,0,2,2,2,2,0,2,2,0,1,2,2,2,0,0,0,2,0,1,2,2,0,1,1,0,2,2,1,1,0,1,1,2,0,0,1,1,1,2,1,0,2,2,2,0,0,1,0,0,1,1,0,0,2,2,2,1,2,0,1,0,0,0,1,2,1,2,0,0,2,1,2,1,2,0,1,1,2,1,0,2,1,2,2,0,1,1,2,2,2,1,2,1,2,1,1,2,2,1,2,1,1,1,1,1,1,1,2,0,1,0,2,0,0,0,2,0,0,2,1,2,2,0,0,2,2,0,1,1,2,0,1,2,1,0,1,1,1,1,1,2,1,2,2,0,1,2,0,2,0,1,0,1,2,0,1,1,2,1,2,1,1,0,2,1,1,1,1,2,1,0,0,0,0,2,2,1,2,1,1,2,1,0,1,0,1,0,1,0,0,1,2,0,2,1,0,2,1,1,0,2,1,2,2,2,2,1,2,1,1,0,1,2,1,1,0,1,0,0,1,2,1,0,1,1,0,2,1,0,0,2,1,2,2,0,0,2,1,1,2,1,1,0,1,2,1,2,1,1,2,2,0,2,0,2,2,2,0,2,2,0,1,2,2,2,0,2,0,1,1,1,2,2,1,2,2,1,1,2,2,0,1,0,2,1,1,0,2,1,2,0,1,0,0,2,1,0,1,0,1,1,1,0,0,2,2,1,2,2,1,2,2,0,2,1,0,1,1,1,2,0,2,0,1,2,1,2,2,2,2,2,1,1,0,2,2,2,1,2,2,1,1,1,1,2,2,1,2,1,0,1,1,2,2,0,0,2,1,0,1,1,0,0,1,0,0,2,1,2,0,2,2,2,0,0,0,0,2,2,0,2,1,0,1,2,2,0,2,1,2,0,2,0,1,2,1,1,1,1,1,2,1,2,1,2,2,2,0,0,0,1,2,0,1,1,0,1,2,1,0,2,2,0,1,2,0,2,2,0,0,1,2,2,1,2,0,1,2,2,1,2,1,1,1,0,1,2,2,1,2,2,2,1,0,1,0,1,1,1,1,0,2,1,1,2,1,2,0,1,0,2,1,1,0,1,2,1,2,0,2,1,2,1,1,1,0,0,1,1,2,2,2,2,2,2,1,0,2,2,0,0,1,0,0,1,0,1,1,2,2,2,0,2,2,1,2,0,1,0,0,2,1,0,1,1,1,0,2,1,0,1,1,0,0,2,0,0,0,2,2,1,1,0,2,1,2,2,2,2,1,0,1,2,1,1,1,2,1,1,1,1,2,2,1,1,0,2,1,1,1,0,2,2,0,2,2,1,0,0,2,0,1,2,0,2,2,2,0,1,2,2,1,1,2,1,1,1,1,0,1,1,1,1,1,1,1,1,1,2,2,2,0,2,1,2,0,1,2,1,1,1,2,1,2,0,0,2,0,2,0,2,1,0,1,1,1,0,0,0,1,1,0,1,2,1,2,1,1,0,0,1,2,0,2,0,1,2,2,2,2,2,2,2,2,0,0,0,2,2,2,2,0,1,0,1,1,0,2,1,1,2,1,1,0,1,2,2,1,2,2,1,2,1,2,1,1,1,1,2,2,1,1,1,0,1,1,2,1,0,1,2,1,2,0,0,0,2,2,0,2,1,1,0,2,2,2,2,1,2,1,2,2,2,0,0,1,1,1,0,2,1,2,0,1,1,1,0,2,2,2,0,0,1,2,0,0,1,2,1,2,2,0,1,1,0,2,1,2,2,2,0,0,1,1,1,2,2,0,1,2,0,1,1,1,1,1,0,2,0,0,2,0,1,2,2,1,2,0,2,0,2,1,1,1,0,1,2,1,2,1,2,2,2,2,2,0,2,1,2,0,2,1,2,0,1,1,2,2,1,1,0,2,0,2,0,1,2,2,1,2,2,0,2,1,2,0,1,1,1,1,1,1,2,1,1,1,1,2,1,0,1,2,1,0,2,2,1,1,2,1,1,2,2,0,0,1,1,1,1,2,2,1,0,2,1,2,1,2,1,0,2,1,2,2,1,1,1,2,0,2,1,0,1,1,1,1,1,0,1,1,1,1,0,0,2,1,2,2,1,2,1,2,0,2,2,2,2,2,1,1,2,2,2,0,1,1,0,2,1,2,0,0,1,0,2,2,2,0,2,1,0,1,1,0,2,1,2,1,0,1,1,1,1,2,0,0,1,2,1,1,2,2,2,1,2,2,2,0,2,0,0,0,2,2,1,0,2,2,2,2,1,1,2,0,2,1,0,1,1,2,1,0,1,1,1,2,0,2,1,1,0,2,0,0,1,0,0,1,1,0,2,1,2,0,0,1,0,2,2,1,1,2,1,0,0,1,0,0,0,2,0,2,0,1,1,1,1,1,2,0,1,0,1,1,2,1,0,0,2,1,1,2,2,2,2,1,0,0,0,0,0,2,1,2,2,1,0,1,2,1,0,2,1,1,1,0,2,0,2,1,2,1,1,2,0,2,1,2,2,1,0,1,0,0,2,2,2,1,2,2,2,2,1,0,1,2,0,2,1,1,2,1,1,2,1,1,1,2,0,2,1,2,0,2,0,0,1,2,1,1,1,1,1,2,1,2,0,2,1,1,1,1,0,2,0,1,1,0,1,0,2,0,2,0,0,2,2,1,2,2,0,1,2,1,1,0,0,2,0,0,0,0,0,0,1,2,0,1,1,1,0,1,0,2,2,2,0,2,0,2,0,2,2,2,2,2,2,2,1,1,2,1,1,1,2,2,0,0,0,2,2,1,0,2,1,0,0,1,2,1,1,2,2,2,2,0,2,1,1,2,0,0,2,1,2,2,1,2,0,0,1,2,2,1,1,1,0,1,1,2,0,1,1,2,1,1,2,1,0,0,0,0,2,2,0,0,2,2,1,1,2,1,0,0,1,1,1,2,2,0,0,0,2,2,1,0,2,1,2,2,2,0,2,1,1,1,1,2,0,1,1,0,1,0,1,2,1,2,0,1,2,2,0,2,2,2,1,2,0,2,2,2,0,1,1,0,0,0,2,1,2,1,2,2,1,2,1,0,1,0,1,2,0,2,2,2,2,1,2,2,0,0,0,1,1,2,2,0,1,0,1,1,1,1,1,0,0,1,1,2,2,0,2,0,0,1,1,2,2,0,1,0,2,1,0,0,1,2,2,2,0,0,2,2,2,0,1,1,0,1,1,1,1,2,0,0,1,0,0,1,0,1,0,2,2,0,2,0,2,2,1,2,2,2,2,2,0,2,0,1,1,1,2,1,2,1,0,0,1,2,2,0,2,1,1,1,2,0,0,2,1,1,1,2,1,0,0,2,1,0,0,1,0,2,2,0,1,0,1,1,2,2,0,2,0,1,0,2,2,2,2,1,2,0,1,0,2,2,2,0,2,2,1,0,0,2,2,2,1,1,2,2,2,2,2,2,1,2,2,0,2,0,2,2,1,0,2,1,1,2,2,2,2,0,0,0,2,0,1,1,2,0,2,2,1,0,2,0,2,0,2,1,1,2,2,0,0,1,2,0,2,0,2,0,1,1,2,2,1,2,1,2,1,2,2,2,2,0,1,0,1,1,2,0,2,1,1,1,1,2,2,2,2,1,0,0,0,1,0,2,0,2,1,0,1,0,1,0,1,0,0,2,2,1,2,1,0,1,1,0,1,2,1,1,2,2,1,2,2,2,2,2,2,1,0,0,1,1,2,1,2,1,1,2,1,0,2,2,1,2,2,2,0,2,1,0,0,1,2,2,2,2,0,1,0,0,0,1,1,0,2,1,0,2,1,0,2,1,1,2,1,2,2,0,2,1,1,0,2,1,2,0,2,2,1,2,1,0,0,1,2,1,2,1,2,0,2,0,2,1,1,2,2,1,1,1,1,0,0,2,1,1,1,1,2,1,1,1,0,2,1,0,1,1,2,2,2,1,1,2,2,1,2,2,1,1,2,2,2,1,1,2,2,2,0,2,0,1,2,1,0,2,0,0,0,2,2,2,1,1,0,1,1,0,1,1,0,1,0,1,1,2,1,0,1,2,1,0,1,0,1,2,1,1,2,2,0,2,1,1,2,1,2,2,2,0,0,1,2,1,0,1,2,1,2,0,0,1,2,1,0,0,0,1,0,2,0,2,1,1,2,2,1,2,2,1,2,1,0,2,1,0,0,1,2,0,2,1,0,1,2,2,1,2,1,2,0,1,2,1,1,1,1,1,0,1,1,2,1,2,1,2,1,0,1,1,2,2,1,2,2,1,2,0,0,1,2,0,1,1,2,2,2,0,2,0,1,1,0,1,2,0,1,0,2,2,1,0,2,1,0,1,2,1,1,1,0,2,0,2,2,2,1,2,1,1,0,1,2,0,1,2,1,0,1,0,2,0,1,0,0,1,0,0,1,0,1,1,1,2,0,2,2,1,2,0,2,2,0,2,0,2,0,2,2,2,0,1,1,2,2,1,1,1,2,0,0,0,2,0,2,1,1,1,0,2,1,2,0,1,0,0,2,0,1,1,1,0,1,2,1,0,2,2,0,2,1,0,0,2,1,1,1,2,1,1,2,1,2,1,2,1,1,2,2,0,1,2,2,1,1,2,2,1,0,1,2,1,2,2,1,1,1,1,2,1,0,2,2,1,2,2,2,2,1,2,2,1,2,2,2,1,0,2,0,1,2,2,1,2,0,1,1,0,1,1,2,1,1,2,2,0,2,0,1,0,0,1,1,1,0,1,2,0,1,2,0,2,0,1,1,0,2,1,2,1,1,1,0,2,1,2,2,0,2,1,1,2,1,2,2,2,0,1,0,2,1,2,2,2,2,1,1,0,2,2,2,1,2,1,2,1,2,0,1,0,2,2,1,0,2,1,2,0,0,2,0,1,1,1,2,2,2,1,1,1,2,2,2,1,1,1,0,1,1,2,0,2,2,2,1,1,2,2,2,0,0,2,1,1,0,2,2,2,1,2,1,0,2,1,1,1,1,0,2,0,0,1,2,0,2,2,1,2,1,2,1,0,1,1,1,1,1,1,1,1,1,1,2,2,1,2,0,1,0,2,0,0,1,1,0,2,0,2,2,2,1,0,1,2,2,2,2,1,0,0,0,0,2,2,0,1,2,0,2,1,0,0,1,2,0,2,2,2,2,0,1,0,2,1,0,2,0,1,1,2,1,1,0,0,2,0,1,0,2,2,0,2,1,0,1,0,2,2,2,2,2,2,2,1,0,1,1,2,1,1,2,1,2,2,1,2,1,2,1,2,2,2,1,2,2,1,0,2,0,1,0,2,2,2,0,2,0,2,2,0,1,0,2,1,2,1,2,0,0,2,2,0,2,1,1,2,2,1,0,2,2,2,0,0,0,0,0,1,0,2,0,1,2,1,1,2,1,1,0,1,2,0,2,0,0,1,0,1,1,1,1,2,0,0,2,1,1,0,2,1,1,0,2,2,1,1,0,0,1,0,2,2,0,2,0,2,1,1,2,2,2,0,1,0,2,2,1,2,0,1,2,0,1,0,2,1,2,1,2,1,1,2,1,0,2,0,2,1,1,1,1,1,2,2,0,2,1,0,2,2,0,2,2,1,1,1,2,1,1,1,2,2,0,1,0,2,2,1,2,1,1,0,1,1,1,0,1,1,2,2,2,1,2,0,0,0,1,0,0,2,1,2,1,0,0,2,0,1,2,1,1,1,1,2,1,1,2,2,1,0,2,0,2,0,1,2,1,1,1,1,1,1,1,1,2,2,1,1,1,1,2,1,2,2,0,2,0,1,0,2,0,1,1,2,1,0,1,1,2,2,0,1,1,1,2,2,0,2,1,2,2,1,2,2,1,1,1,0,1,2,2,1,2,2,2,1,1,0,1,1,1,0,1,1,1,2,2,2,2,2,1,2,2,2,0,1,2,0,1,1,0,2,1,1,2,1,0,1,2,1,0,2,2,2,2,2,1,1,2,1,1,1,2,2,2,1,1,0,2,1,1,0,2,2,2,1,0,1,1,2,2,1,1,0,2,0,2,1,0,1,1,2,2,1,0,0,2,2,2,0,1,0,2,1,1,1,1,2,1,2,1,2,0,0,1,1,0,1,1,1,1,0,2,2,2,0,0,0,0,2,0,1,2,2,1,2,2,0,1,0,0,2,2,0,0,0,1,1,0,1,1,1,1,1,2,1,0,2,0,0,1,2,2,2,2,2,1,2,2,2,1,1,1,1,2,0,2,2,0,2,2,2,2,2,1,1,0,1,1,1,1,2,1,1,1,0,1,0,0,0,2,0,0,0,2,1,0,2,1,0,2,0,2,0,1,0,2,2,2,1,1,2,1,1,0,2,0,2,0,1,2,2,1,2,2,2,0,1,0,2,1,1,1,0,1,1,1,2,1,2,2,1,1,1,2,1,1,2,1,2,1,2,0,1,2,1,2,2,1,1,1,1,2,0,1,2,1,0,1,1,0,2,1,2,0,2,2,2,0,1,2,2,2,0,1,0,0,2,2,1,1,2,1,1,1,1,0,1,1,1,0,1,1,1,1,1,2,2,2,2,1,1],"coloraxis":"coloraxis","symbol":"circle"},"mode":"markers","name":"","scene":"scene","showlegend":false,"type":"scatter3d","x":[0.041341485926379776,-0.17104229255420342,0.07670535601659713,0.17732097521340734,0.19854063871960662,0.11486239020748962,0.04460555677200918,0.3202757249191134,0.09727180418323231,0.17482111539663175,0.48093047832221836,0.19241028335098434,0.07510113376396002,0.12687869837371932,-0.015691326265338056,0.2502591167851221,0.39973407383038034,0.07456859314916305,0.37150260679959046,0.25212732056372955,0.005464829771254512,-0.14826056436014418,0.04143349849816444,-0.08557110130343838,0.24135725014217624,0.11143022372519411,-0.17358382667071084,-0.22549518183306375,0.3294583690055389,0.1164804478823256,0.24109474288326432,0.18860757781065227,-0.0684545986850738,0.26239214629649815,0.14527242925150546,0.29706051565090963,0.14948240549558145,0.24287985196082065,-0.051454975329810845,0.007933428942780601,0.05047247506011559,0.034347052220784664,-0.15750696483935436,0.051149586357019834,0.00869839456596976,-0.3239333311965771,-0.11584436806529375,0.11292818049141312,0.1065709919980336,0.32991957921254633,-0.046790671854772065,0.19779451980503607,0.13586902506299947,0.10355977584908395,0.08263383360241512,-0.04620288538653103,0.049169886710676236,-0.06303729749571739,0.046299248253048715,-0.07395870131734548,-0.12780255589061892,0.019751637543929598,-0.29763109116768266,-0.033015059922299396,0.329782739325888,0.14669453443091027,0.11014247669216413,-0.006110438683060724,0.04465102888030364,-0.1962187671599827,0.04092551241169362,-0.24711397720925413,-0.022235511241292593,0.03344018895073679,0.05526227741805288,0.06374217273775208,0.16709045722353102,-0.10888040428898701,-0.07886861268087701,-0.04870328512077871,-0.23987171984840447,-0.21874023026258788,0.4222369888070405,0.1688068949265185,0.023262316115112525,0.16233492108849035,-0.09804668239488304,0.175261341519024,0.20462071950170327,0.03215999374110615,-0.05982363143975342,0.3053774262610856,-0.09732114514090873,0.11709045554040598,0.15082887657496874,0.17976317739460798,0.09059701732366142,-0.11969606729696947,0.06440403582958389,0.08197648949202235,0.1297812092600249,-0.07829516036326169,0.23694510955555212,0.0640566827497088,-0.14649275730603062,0.026260498091384886,0.021233501754485046,-0.0753140786270104,0.2861725956860427,0.0864245308558571,0.38345569052798034,-0.1275948361724751,0.0512006421187117,0.10678233930610885,0.17582778227921333,-0.11879762759364336,0.16310251722396457,0.1410590020097479,-0.026092889624433714,0.3018973669189455,0.21395067435252627,-0.06498242443971375,-0.04057883620450659,0.21262500380583987,-0.16033651321848397,0.24215136455904807,0.23563172531028734,-0.28946972246706953,0.06946516010750865,-0.20164676640989673,-0.07320489704329503,0.07390972065392605,-0.021255904239369846,0.09778237271357121,-0.12483930823540014,0.028147966449288603,0.09483704006833982,-0.2324381646891361,0.4634563044974524,-0.13961141061384494,-0.07255374051044965,0.017520318827316483,0.25696969899539757,-0.0967367518986599,0.03202241638332326,-0.04873105651181176,0.12726354131003811,-0.11976017878871678,0.45759872405494606,-0.02712907631929038,0.24509642634129689,-0.046519815650563914,-0.21634957000700425,0.08502119919771514,0.18398157526817624,-0.013703704746734439,-0.1881842090218195,-0.16154831555943458,0.06473214679854566,0.48908383125285704,0.03269001699953224,-0.022065758235150106,0.25736987448778076,0.22704679459603203,-0.16163425495479095,-0.20597975751544095,0.05095517360156336,-0.09092639027372333,0.04324637069277949,-0.20492832760571514,0.060700782294308454,-0.003795663420629915,0.10621148903738734,-0.027036826650936742,0.08102533332004022,0.014693407326405825,0.039827374088741624,0.07148675455213427,0.24054120785127245,-0.07715995886674536,0.05433926800148601,-0.13790455243737187,0.011757960935640403,0.16974643370245077,-0.04267618810055842,0.3885169545086288,-0.16766898900858354,0.12538990277177042,0.10159140748829731,0.2987193969903439,0.39085918829121663,-0.00827769298120034,-0.03710393229569296,0.1109241496213286,0.0747420978591345,-0.29107120215152027,-0.1415994529195566,0.1536233886897069,-0.25169505205804793,-0.0590424166140559,0.07503111737513682,-0.009873750522886337,-0.25368571423368763,-0.08123001148311342,-0.1537989395978381,-0.16653882701481706,-0.17504492541216163,-0.03092988933600073,-0.04688582436873271,-0.2182599787445896,-0.17219941750611611,-0.0016288191994968271,0.0001743260402979834,0.270460662865535,-0.1374581699509531,0.09423607082615311,0.04544460612676474,0.006673172481740975,-0.03132029397435889,0.18646719332961484,0.14671448333330211,-0.14154274696586625,-0.08359721027426603,-0.06528756848000948,0.05129420483385132,0.11510179715132114,-0.03664958642858471,-0.06484808666517274,0.06434600793556362,-0.019367741605974122,-0.00046778687989575086,-0.0666103749691327,-0.1954451317218828,0.0670665635212969,-0.14824534086247768,-0.05460303034216305,0.15405301386092124,0.06467940144859301,0.10585299100511943,0.23926184628416872,-0.07079578426749465,0.14021541727413717,-0.06978044534892446,-0.04313349065228477,-0.0413450106055642,0.08764333097486675,-0.05800280883027946,0.015610475884353516,0.06800916441457418,0.06339876648563363,-0.18630122692626952,0.049089609400010004,0.025864617896126835,-0.02416346439445227,0.051993464404913724,0.10431656768531984,0.3893609385563957,0.15568383990745122,0.3681878269492904,-0.1598238120128978,0.10596177532267816,-0.028566787082991405,0.06426426429077733,-0.034664762071718175,0.18660462275673717,-0.09537255171326411,-0.042822076283311264,-0.10360515006474641,0.031144576135428276,0.17501863906745893,0.2877207925102926,-0.0417957402185314,0.008890337165859569,0.012323722586829032,0.04260781531358616,-0.2541125244570122,0.08513907457307707,-0.09063469305164859,-0.034002349869910936,0.1179211547164044,-0.11567598844959671,-0.07809161865675071,-0.05646300529029241,-0.10083063306323699,0.019615718265433613,-0.07131410550263975,0.04153142984320143,-0.10642988790333395,-0.035002697297143406,-0.11745985186873238,0.16022131718320803,0.06338437149730416,0.0967638592621682,-0.0875536124570343,0.1095111595082488,0.09680005533088185,0.1190985560608388,0.06819201566748569,0.13447964697657272,0.20098815536067352,-0.16916119309882904,-0.23756792665002102,-0.013455433547345559,0.1418278736942408,-0.2964664206156949,0.20644527799968015,-0.020083425871133143,0.48907665068315925,0.04017881278793821,0.12758747940559462,-0.036856548333658985,0.26588522756477745,-0.1740979420877728,-0.0010651802133908935,0.15458377230929396,-0.12645158533495013,0.0059031032093946716,-0.009141292659309824,-0.09076663318581969,-0.11951193187111431,0.022077650401605878,-0.09332490316768066,-0.11583197885794574,-0.00953107541139957,-0.19593659174762146,0.1669346074930376,-0.05550790774344578,-0.14426626456328315,-0.06135244454079812,0.013570367986701692,-0.037751500405291884,-0.12651818638275503,0.11487999210947612,-0.052503096990489564,0.1746822265827467,-0.062317839137248095,0.3890307244933972,-0.13622820399030142,0.0012789816714654223,0.013598171528118031,0.17593911704620507,-0.07895695028776542,0.2793860480905242,0.036924976385767055,-0.04550620852858207,-0.11007162171337849,0.3149701065019302,0.28961305814081567,-0.12271312427958397,0.183988429260487,0.03530359253210292,-0.051719131816585724,-0.046409357614240004,-0.12504927846759162,0.07921111266401948,0.1788162335884177,0.2587640007535713,-0.002658494773939954,0.28557824008622545,-0.29108134873096364,0.10308800138545654,0.06429050309984767,0.045885604153658444,0.08262695789877787,-0.02665322671527843,0.10747353243523286,-0.06860344324317956,-0.13469866036281194,0.062059157072015696,0.0965321463134932,-0.014454231782103501,-0.19434046463477928,-0.16760005373349854,0.2962116673299674,-0.011431059007410264,0.0726268815847536,0.2594467497260414,0.29988704011331413,-0.058889076081891374,0.21558154821793518,0.22208332144421083,-0.10820926920099509,0.3212705848040828,0.06266640101562825,0.00568301587843415,0.17190938020207705,-0.11671167826677978,-0.1712882332097357,-0.10980784403318913,0.24465387353108026,-0.04172334452765962,-0.21718766388391963,0.18619514176623342,0.03689191515231583,0.2898245342580764,-0.036394798911803826,-0.17606497173571625,0.011535265109064184,0.051234679196917245,0.04639846338070188,0.2192617553055139,0.3164884583844,0.014397805305664722,-0.17200427610848407,-0.28685181358994943,-0.0848972797646062,-0.06533342494378992,0.10629585632078693,0.020836286436915746,-0.2697301030448802,-0.329332418464667,-0.013479281047175069,-0.030944338295507953,-0.08250844904174165,0.06446903690635214,-0.0636998133857485,-0.15199920678786658,-0.014369557568829606,0.13702997514812454,0.17363026958011935,-0.20800187969140296,-0.16714865703671433,-0.04265625295670628,-0.1987527897084625,-0.19491751435284654,-0.0691679428570634,-0.21170485869797143,-0.049455140963538695,-0.2609431083844992,-0.1521901439867134,0.48163867478612393,0.2636888972813736,0.2734160547908921,0.3581592254537661,-0.044027066954006655,-0.015209553717774882,0.46505807998802157,-0.09439745355023725,-0.21163147697148382,-0.13879765201705024,-0.12470948270490015,-0.20741246854643647,-0.23348352148389856,0.1717730984934476,-0.017043496276907946,-0.021009694618514995,-0.21258117020000683,0.10201748369524544,0.06640523993415151,0.14431493859539202,-0.08149094913560916,0.3017081184549263,0.12154518562779042,0.24471021584584227,-0.2219404185095265,0.2530120052856525,-0.10929538239610596,-0.24054242938448356,0.11085902918190663,0.11826983017597534,-0.20779570306296788,-0.16882076566469975,0.25871867166894724,0.012945243316358901,-0.19156087035740207,-0.07952302209067932,-0.19896093299368248,-0.11239663214005619,-0.11254290739137471,-0.2951787714350732,0.2866971949093229,0.17830124899504649,0.24885328806790466,0.05616465394443831,0.04752890657030663,-0.08411718813614111,0.0008235428908180264,-0.10351082497162443,-0.17613998199438122,-0.07827733311535612,0.046992643546650795,0.15165945920510776,-0.21519415136249645,0.12312062017640181,-0.07996626982497988,-0.036731558713851274,0.05431687160895485,0.14780208974295647,-0.0373143121202935,0.41815767376875396,-0.2517830894548388,0.013704550409304718,-0.09084977706294871,0.242708816649578,-0.3346060648980584,0.30353127060483565,0.4650525082609146,-0.3060577870517345,-0.19200075879291473,0.3666345653505825,-0.019622189156012607,0.09937896317392463,0.19978804781190596,0.010890455713665027,-0.13129207378781255,0.07538218228591596,-0.14040453610503228,0.15240135577862826,0.23148372511087723,-0.039578335925931,-0.05295654808673618,-0.04597419389544329,-0.049816020943715816,-0.32391900398201845,-0.2918736387855954,-0.14933091693589243,0.1396971920492282,0.33121801214166885,-0.25829535321719793,0.1957270534142693,-0.040230875934189614,0.07025339723270681,0.22020804189140056,0.06291240526199268,-0.11353722641321858,0.48628887742839794,-0.16961917089170633,0.07246162966267126,0.16413102685066477,-0.2927508494626155,-0.21298904127087234,-0.3455445518067278,0.4369671242166864,0.12987737326465107,0.0712164770172508,-0.04435515187430329,-0.08896295984600067,-0.2569116329284154,0.09446535348139161,0.2812427550281604,-0.00010141224023578546,0.0327293809329521,0.08551206030717214,-0.26443502667355673,0.1045044428472092,0.15628573198823126,-0.17861114268094308,0.00465122117700313,-0.06545478236045445,-0.1311106863889191,-0.1491534805454029,0.4456569085028488,0.3052750282638042,-0.040759915204678254,0.1216289119170362,0.15414372957737377,0.059971737209779,0.15172353546930292,0.08369328070278703,0.314706439634575,-0.13106382361136923,-0.18452314783719237,-0.13021193812724338,0.023197584534252916,-0.23784193727211603,0.059383019241349595,0.17944412080572222,0.3030125824206174,-0.25753284257844744,-0.06508583665417399,-0.045097234140394384,0.1376791374803472,0.04622876470801631,-0.016190120086330105,0.35192046310085484,-0.3065884221108622,0.2394976783327645,0.04952402813821993,-0.3693664894068015,-0.2986634882261497,0.2102204423436563,-0.047725783761146154,-0.210151050670352,-0.09360005052746204,-0.16522798022978624,-0.002305012934073335,0.13677299553195246,0.45158966525282845,-0.1273970616622855,-0.15349239187901215,-0.013317232036921138,0.20036714145046142,0.2593602494612136,-0.13214712927907485,0.06449786813911636,-0.094161482998951,0.15607398918598453,-0.09167510997920664,-0.24189164800293322,-0.1763004057584368,-0.04129491176588164,-0.048236060152074246,-0.054492296298732056,0.16180222053610305,0.02932781345592543,0.3452679681935723,0.2640939104581271,-0.035210334792380955,-0.21294173361788102,0.0008276471312700628,-0.11087787164292873,0.09230157564358225,-0.02150528645936848,0.001861969902026874,-0.12900030704938367,-0.06064219323454894,0.2504560874046885,0.4068012403162528,0.17581254535190613,0.24471459189755426,0.020796399608698196,-0.065218892775694,0.22299020232130617,0.116701103999199,0.02034224297543089,-0.21924414746135182,-0.17441984130534466,-0.018773209367453452,-0.27136177198403283,0.057127940124459334,-0.20308248621615965,-0.3238296757629379,-0.05389309807784431,-0.013976549252142138,-0.0020437225905620797,0.30061584756724175,-0.16552826707441312,0.15694654096787608,0.07198045844556489,0.07264745727043415,0.01647205344656034,0.20809437469426406,-0.23473298511736493,-0.27341741020436267,-0.1785878413128649,-0.16698535789903596,0.23341437413419638,-0.017960356288579435,0.048079836892493226,0.4790472450094055,-0.27368397077324824,-0.10268242016986459,0.16309483176546957,0.19371903589306091,-0.0683400322315077,-0.1595773823930642,-0.00655806514955661,0.0559985393972622,-0.1509773954161627,0.027084303447875463,-0.10492290764525067,-0.10649770178021843,-0.17961007226649353,-0.13546257725260574,0.19349911972816278,-0.08507021543828198,-0.1906361204747205,-0.15342277438235555,-0.33348223060134696,0.31436573932647655,0.3269398134132544,-0.16581565443119714,-0.07228669566642396,0.00021829046205835187,-0.11490502763743239,-0.037414802614458856,-0.1311215394339139,0.08819560620170337,0.025659178152214742,0.08822880262914706,-0.2342653655510952,0.12787753518746048,0.022180063298566505,-0.06737982391498133,-0.17886693519749725,0.10481079859028115,-0.1306281674193274,0.13197121477628618,0.14289387403240694,0.2917490050120176,-0.25618920591272587,-0.0322502129237793,-0.062416960291592896,-0.31308493751331123,-0.13369217094606342,0.007345677395455621,-0.053390059275313116,0.09670079892106918,0.24776883855721152,0.2457102407560546,-0.06505450837723246,-0.05513862316898203,-0.2025532652170062,-0.20427650442127848,0.04568630477926914,-0.2535663151009097,-0.024035794488631423,0.4889640893610101,-0.15278978469814156,0.31801215205328326,-0.14111494872864241,0.268668769263665,-0.3687983064932934,-0.24705590849594514,0.14026373997939398,-0.07481992614246863,0.290691976985259,-0.17948797734923497,0.11425127955096392,-0.20245713069684382,-0.16392993930546584,-0.13896581074176675,0.03304239449492212,-0.1757141309674974,-0.10627023010562177,-0.015285646347485691,0.4884257893236844,0.18331966646361772,-0.1559152907781758,-0.04569984535005451,0.45670918077844697,0.012114554242405302,-0.059899150335941224,0.25727421357913566,0.18672012220909312,-0.24276987533362768,-0.024443098440870756,0.08843306330281261,-0.14022295307476518,0.05722689667964263,0.04410321709060812,0.06641227060694628,0.0763432977828908,0.06771035615542118,-0.25071560577174185,-0.2811387972230019,0.11602868153481444,0.4890719421945227,-0.04370933716167624,0.22615817315735437,-0.0012057239350092624,-0.2263337892132465,0.000480422375459504,-0.25167276062426075,-0.10178262652714044,-0.03107433879212123,-0.13934350955046074,0.14840060790036272,0.20158982133380315,0.02888411319781282,0.137960128713238,0.12669436038637052,0.08820085120360599,-0.017369278479503817,-0.26177419471352564,0.04937029234418578,-0.15281538320535684,0.16599263128520542,0.2107043616186464,-0.20743430552053796,-0.17378335324009667,-0.10999292722864241,0.18272383878967666,0.03133213418033607,0.12147618392707224,0.18552481685575742,0.08407539812662441,-0.051151038986980986,-0.02616118662857465,-0.18820164972548659,0.06327548354989226,-0.10082001956495118,0.010107024257042994,0.12209037675687207,0.06549284485128658,0.004956953190534306,-0.036126967018359026,-0.14608222877530072,0.05736895442020243,-0.24474436415299655,-0.09997522436777094,0.091773466496734,-0.020841138477991326,0.06993456270726234,0.10272184584312245,0.4526238152804335,-0.1954957474507284,-0.163905685089244,-0.053880232993182266,0.37594290948965225,0.02018642804428347,-0.1650273842031659,-0.02672020154038594,-0.08315266394298482,0.2269951130784034,-0.051445440744759804,-0.20415316255281768,0.14699230100777944,-0.2754211882990421,0.4744529625553344,0.2421510910612784,0.09133209253692733,-0.2165997741506357,0.0741612576201358,-0.15036672989888533,0.00022171400302852285,-0.12494622518222136,0.05434843344794454,-0.1907228868175259,-0.1319931037545059,-0.18363696578079983,-0.1664616558246367,0.017466810920295738,-0.14096866806976485,0.18313319798287847,0.27425562080461463,0.0824088245206306,-0.19692653036979496,-0.1522804739707499,-0.007839928811056926,0.06497632738174576,-0.11508817569967689,-0.18955780947029444,-0.12334406137493263,0.4889640893610101,-0.07898720839368252,0.131404226293966,0.26547268854435374,-0.1271256597137749,-0.13085413038568972,-0.2446288137540973,-0.04433237451628711,0.04249574384306303,-0.1128717220097845,0.04013968267426265,0.012825716545498292,-0.36552086765126246,-0.07402061636115714,0.2862866473191814,0.039396199114528525,-0.21433342954774348,0.15100200594379126,-0.11284960583384267,0.11964876259259712,-0.16066643499289338,-0.1695005106679662,-0.3258153916801611,0.2444757186791658,-0.21621618717024696,-0.1479045416973337,-0.02822626898504108,0.02185149436626381,-0.2321884316665314,0.06342439332428533,0.026968403027891538,0.19730008645296715,-0.10424592289813392,-0.004154759045849013,-0.16989196749987975,0.08384245723036945,-0.10092042167524976,0.029749555425931978,0.24015370344819809,-0.10393124391131237,-0.01993068404971937,0.27013725913410147,-0.34593354237590845,-0.08920816948861822,-0.04793938596426037,-0.21887837089429935,0.1608555215762732,-0.1470856613336904,-0.2332233138873157,-0.003249499370409608,-0.1737420703114651,0.3392044291433543,0.03322770760184491,0.08399778829365505,0.11296795488982068,-0.008354968605856772,-0.2341855635365121,0.03481424411407025,0.36540341203733046,0.06100192966412066,-0.07645018199105927,-0.1632141059840955,-0.12472469174703966,0.031429769613353975,0.24467980709315468,0.19722736480466718,0.1265644592838676,-0.031589325754429104,0.028596863942805178,-0.027872513789796608,0.23807528096421343,0.10418947837041853,-0.15303227712811562,-0.2698266818693481,-0.06722186201769743,0.03532786219069864,-0.0157825216646333,-0.03236460568811396,0.3167646288262079,0.048753750224941876,-0.19714008140988537,-0.030862998988416105,0.12602101461277962,0.13712690825016036,-0.2083255811120855,-0.1902695868652609,0.06718170207175625,0.095228616597063,-0.18377426918614256,0.346352224565772,-0.014422055334979244,0.14498544559843363,0.12931687628500452,-0.04989362835024359,0.18041298607139483,0.0174109697333533,-0.2515050349268205,0.28176618751609983,-0.23718452540964205,0.08726553211070827,0.2661009653566327,0.12078053622054082,-0.27800534811731736,-0.011001405045643068,0.1934474908945275,-0.07709279443074929,0.08627052899488848,-0.12006165705915962,-0.11389943873890736,0.01651359608375942,-0.15839652452841096,0.12155572995297316,0.031660554211565245,-0.13679676885673758,0.22239375018563723,-0.0064248262578135715,-0.013448492431573344,-0.344753956044705,-0.3121919681887373,0.07598648199740027,-0.14026098412294816,-0.07168833161477707,-0.13490569148064718,-0.28230360903177154,0.2712812915379805,0.008051541160498773,0.11672990376827906,-0.09258348474326551,0.02052443813815752,-0.07312047115019539,-0.09066207953576448,-0.2965824119614456,0.05615808701979796,-0.08293206880137319,-0.01760618953064191,-0.32067659246660146,0.057192087212037813,0.02354949677414992,-0.07382485071734649,0.11060950124196683,-0.02217006415368164,-0.06256370866126343,0.0004450229169442329,-0.17528131259927637,-0.01124303550077642,0.012614353266949414,-0.04989321723357312,-0.05679946225720536,0.08210039598096155,-0.0005332071936640314,0.16183007657980578,-0.14131337271109565,-0.29643377718326447,-0.09964259953404915,0.11821250285735686,0.2655059277464495,-0.2225808032672577,-0.26841010593985054,0.01564153604176523,-0.060059763417889,0.22528412794424796,-0.27824355919593585,0.038749737845078965,-0.1941142792783927,-0.13699811526867509,0.2982197988840224,-0.10129280205299161,-0.11485372798209346,-0.2298685792069544,-0.12824371195899054,0.16641981098741743,-0.08925510089869054,0.23915806458163813,0.09357315404891138,-0.04591417712064901,-0.11475896206886285,0.14713680236995155,-0.15504606984903105,-0.04353875378876254,-0.29403127356836184,0.08297026453522675,-0.26748030306697296,-0.0683872024904418,-0.28386720488625045,-0.15442878308496105,0.2578064121478173,-0.01796578857740866,-0.032379612521672374,-0.14338626954244285,0.15709221367997211,0.2935734900216705,0.19221377074645807,-0.24714467215217112,0.1872311417378026,0.10096114934438884,0.039325099451619674,0.036257950359165965,-0.15813372351121366,0.1696078980062524,0.09147211154595053,0.04528178449054643,0.09639423786466779,-0.18747472294210835,0.13179838975191135,0.1869098249090825,-0.3518991689644648,-0.3409100209023256,-0.3085471324325442,0.23569709838198544,-0.08409324904429033,0.392798766577993,0.0523642799278301,-0.09396013308364433,-0.17748735751900374,-0.1356846721772232,0.11146339443599527,0.015711449086116536,-0.04911642335465759,0.1947422657746616,-0.12044144321610593,-0.03895850314825128,0.19387483544499784,-0.006643969606384588,0.10960722159728438,0.16024920591767483,-0.09032153430592964,0.038716739230680856,0.09146673619879322,0.014310391388022009,0.05272637621945966,-0.10175778311104794,0.2181581382179919,0.06455192918559202,-0.0888603234779511,-0.1543210893628001,-0.029134585027361222,-0.20565149624766837,-0.36992107494699716,0.12862274772355098,-0.1424745779821945,-0.2517983778287361,-0.1266783418054699,0.022679823587551336,-0.09696214264273376,-0.030730664968896756,-0.008395914786575621,-0.25076753919672634,0.28163343204650665,0.10547659670692,0.0009172962312215195,-0.22492056198418584,0.25428918391297856,-0.1527388782362245,0.03963801821244896,0.03248892418415651,0.26033468818327316,0.2355079741562934,-0.008787871224036747,0.00013594851521172029,-0.24786850715568964,0.17802082099192884,-0.0160020675261879,-0.1412312848346192,0.12365676205688941,-0.2004543677595029,0.028643059510578394,-0.10443315417241841,-0.12043342993078852,-0.011660450648899558,-0.2127921966843148,-0.10482006958541251,-0.16235317489577272,-0.099948097417087,0.2565311663853215,0.3409331042142502,-0.22532118525599665,-0.00966614248278336,-0.06534257855693221,0.09547126721356955,-0.2242068751534024,-0.174346245014366,0.008872640586411351,-0.0032716437996868197,0.059282073795213316,-0.14187522930093974,-0.019209612771602946,0.025305369594652946,0.06573868788213402,0.2183032895288258,0.20785182483076084,-0.16822254450102864,0.16567569785275765,-0.355868733646284,0.2714373267106535,-0.049579402395195864,-0.10921876141020108,0.03542562338091525,0.006520137393098988,-0.0351477040174693,0.1202367917818835,0.03163841318160933,0.06408742323739205,0.012084389343036194,0.18428163844884535,0.17830913520258027,-0.240343929493862,0.25842208860275045,-0.2089697720964806,0.031894215270316724,-0.20075579972018584,0.13997890836740318,0.052404651549340237,-0.1301534161991272,-0.000599918420327799,-0.3119396045047557,-0.09367374234495963,-0.1680913380737562,-0.030534175528514014,-0.1520543046882068,-0.11829613060430574,0.1595478407220484,0.006263422796298683,-0.04504443620266826,0.051998368147713234,0.004513833433093472,-0.0005627097738908147,0.1978112998213039,0.37259197484535717,-0.3206686371256261,-0.09511802607020416,0.3050714065728546,0.20722747346417658,-0.00046343172073898487,-0.1785492843070466,-0.24432911603754473,-0.2930349277736089,-0.28236867033901814,-0.060345441709356885,0.12544428315406173,0.015068743671897514,-0.25450155572425326,0.4488042823572891,-0.15600829665530666,0.000972930511321689,-0.007692112463785384,0.12822776698486407,-0.34668545221842023,0.3603607130302272,0.0993212852183199,0.07541384130737686,0.03910641198120621,-0.12215525188265502,0.08171786430690131,0.08096343080215361,-0.10485507489823931,0.01999353987151699,-0.12742664752782581,0.06681571543457338,0.1222562435029755,-0.11496544197308609,0.012155203663835452,0.1051913484000789,-0.27138118263793726,-0.1649849563216765,0.15440082966736726,-0.09924374807047205,0.1673386430034953,-0.12527284119926724,-0.08962810512233493,-0.013179988475804375,-0.07681393938090184,-0.21854851209410536,-0.03895411596496544,-0.15325195374974196,-0.3437252511983013,0.25078689170859464,0.1247058354991881,-0.2215181738518346,-0.05245468600176896,-0.032913008451437534,-0.0009147582439503443,0.005411343026989446,-0.16480033136765132,0.09798391345633793,-0.31475244441092015,0.0027209794129383915,-0.031993537773151316,-0.18802412970379104,-0.27393482249730505,0.18717900619244532,-0.2279263521664254,-0.08951575553476507,0.14849111414036545,-0.23129155372264867,0.06842165748633018,0.05629637361063571,-0.18652859258525845,0.19794876636574338,0.09355577920792726,0.2014799460440114,-0.023662435030284007,0.02031516982278203,-0.17923318533240848,0.07634130145198975,0.23329582867294427,-0.21284519000483526,-0.28655752282250424,0.0588129649656179,-0.1089979991717278,0.006241002251179309,-0.008527842039875663,-0.16549001138028083,0.17173642483956775,-0.07468835912517388,-0.07118717346633475,-0.22223807432911113,0.24124115908525934,0.06532684159348072,-0.24164375405707084,-0.35503519036210224,0.3473922047561788,0.06296742514885961,-0.21558996325521548,-0.16002592439298094,-0.2027362995105275,0.051268469696727985,-0.30479353449425456,0.10607849347144951,0.06138860866672184,0.25082254429216977,0.10588844942245455,-0.061339899067690835,-0.20884805280358432,3.797953412439274e-05,-0.1766327185215022,0.16669997405772916,-0.24830435640346527,-0.24647629410161231,0.015617466874350776,-0.06993671277676675,0.07568693989455107,0.03218371713939805,-0.20044699038125244,0.3265228346377875,-0.22637172769102037,-0.34802107475871386,0.021777873925249824,-0.17544642675559197,0.08692850955147294,-0.13153434643849263,-0.04906596796757402,0.28450649894040886,0.060041129676419266,-0.3217942029537121,0.35209980284800796,0.03776824866572442,-0.011597020929079627,0.0061273415988989,-0.29998378241296825,0.1513499795556643,-0.06547317903371001,-0.13458850899685626,0.24333365408321364,0.17884885889967891,0.030811397572931495,-0.3539094545721399,-0.08150450747681498,-0.2708977586064054,-0.00407619407073734,-0.2783496947792572,0.0009827430711490546,-0.14227848287397982,-0.00222789829532075,0.001505629296712757,-0.15637704028398824,-0.10757955140434321,0.0631919031281061,0.07193531685132833,-0.2409497615162151,-0.23685416236703485,0.143336196938947,-0.19672871047751017,0.27279982147995174,0.24435663579151534,-0.12895578977251174,0.13107784721821358,0.4811521584922534,-0.1107001946476783,-0.02735859596607145,-0.23898000293701493,0.2509584914320436,0.16371309620577992,0.03326481207496103,-0.1963106606646606,-0.18822649350879495,-0.00031347655487080493,0.17855050921316654,0.08797987763591078,-0.3050304748129555,0.03292528365679054,0.4415984205294073,0.14795614874131358,0.292145064722678,-0.25088369410801575,0.10364654218989244,-0.0471767216723488,0.32536111109116833,-0.036653781006467594,-0.005567759233202876,0.39196653822077815,0.19777993622585563,0.48308305498362536,-0.1728001663553977,-0.11189246911543006,-0.06459518862464364,-0.08531616766757974,0.07581365098608925,-0.18814452208033372,-0.11154412523958805,0.15012957666234145,0.4588668126940894,-0.1648127608175237,0.17133278303415117,0.054226829465752954,-0.30088865223657324,-0.15560125037419537,0.10899808818028196,0.13051914311144489,-0.06993780812101982,-0.18819013733515624,-0.044588484850940656,-0.1494873608460932,0.15699441202265452,0.22519589071705512,-0.2761015085696851,-0.006494141052118716,0.09747308785040237,-0.06134564774503193,-0.24943795760546528,0.10845116168191468,0.04392784329823435,-0.04779699294957637,-0.13800059325141106,0.050338905075056194,-0.1174289623288074,-0.2552907620649014,0.027330723677386485,-0.1377727287587297,-0.12861339448073764,0.12729068940701194,0.2248959099461526,-0.2180345419856073,0.04435765240515769,0.10796801146408631,0.06376975106311449,-0.04582710804399048,0.0963139923306675,-0.02584166157230443,-0.02209482640377912,0.014759708076114668,0.3256357649888963,-0.027848029269586124,-0.07218679001440262,0.08931246401438332,-0.0846487879290888,0.06720180429984296,-0.2987663163361778,0.11121721849765945,-0.11993754929508103,0.22990214419009652,0.19436804411503883,0.15896811890599846,-0.15305372959809313,-0.051804978544250466,-0.12066227854095975,-0.2225542410198638,0.09222171265707155,-0.22007129503555867,0.3265570328369669,0.2238575082421558,-0.24297311947257857,-0.1230739352447955,-0.08069380760975443,0.10919822060157856,0.21428245559837464,-0.05292306734336612,0.2018829228419671,-0.0726699579561625,-0.15732979341748107,-0.22050887747351344,-0.11087073123783969,0.014764397094264234,-0.30089117868411625,-0.34722211262652836,0.03840908093003461,-0.11401455291051923,0.13407181876422897,-0.10566822761586729,-0.09476516375942343,-0.001481576985174807,-0.06991465191248669,-0.34023542879281865,-0.019127319380282743,-0.07635828876859078,-0.2257322442758267,-0.16717321028933138,-0.0511890768000046,-0.18767362773897406,-0.14631300073262615,-0.10050053585894898,-0.0015851176032368544,-0.08604310365706776,0.3632331038534173,-0.12858394187981,-0.18505373500960884,-0.10792679304652113,-0.25073013024390317,-0.03735102211307808,-0.11220555487347976,-0.021821115451927237,0.04367197409993799,0.12443430764057478,-0.008043450903885772,-0.10949879514666971,0.03683529009377792,0.0454093509611751,0.440537295831609,-0.08160944767077802,-0.3541144098883522,-0.1853910108937536,0.042968502713053004,0.13993360089724866,-0.06379645622532343,0.0567127786579714,0.034191431534666375,-0.0634993428014552,-0.046669953570131054,-0.03724701642188412,0.3723824358979778,0.28533067313431193,0.3730118624246159,-0.1393118888356153,0.18284037114049329,0.019429590943138692,-0.25469157610877147,-0.16812070678636237,-0.03161644581351348,-0.21078188312907448,-0.04571321004159866,0.13243957154386699,-0.10039317873385067,-0.2132184044490445,0.09886715218806578,0.41765406683361067,0.0745514417481858,-0.1260514147558899,-0.2669797716454829,-0.3183263887384578,-0.1464700254597662,0.15574696828997459,-0.11121418335541397,0.007276031292369871,0.1910896004812008,-0.2084875003899246,-0.12316503978957465,-0.015590653792076359,0.373702626698793,-0.08768709140828168,0.08376907530319819,-0.2962955261699783,-0.04207336876036054,-0.14370080191479492,-0.08603422723270425,0.06002615623613595,-0.23080128512627565,0.197998551795566,0.011968947200152543,-0.14233061603890518,0.488189271198127,0.04232930975214657,-0.21388672889671417,0.2411424954925956,0.09254104900599712,-0.16103117231643807,0.3069148555985585,-0.1633038670763243,-0.303432212090132,-0.2400101133079755,0.21055513553418756,0.23341747222796863,-0.11335195067658138,-0.10407580308661156,0.025286485704672988,0.11754472783074862,-0.1223353232501956,0.1633208687887518,-0.16859327785963532,0.11042988562907545,0.13834986056376827,-0.13354493561280487,-0.324476539894821,0.19275901375568238,-0.2774463546602838,0.139890163733135,0.026524350673059024,0.1750111948650306,0.24829918682531316,0.07667101284392273,0.0418848889074398,0.21276590258654338,-0.041214926718216936,0.047457469572857736,0.23023776939795065,-0.10586228199481386,-0.04994445283400566,-0.13194475005208398,0.005580027944423917,-0.14169758146387199,-0.22047021306580336,-0.12187803267768194,-0.2470591873842081,0.009658643657123704,0.07333818803387387,-0.13628958431603905,-0.2017446823070901,-0.267133290076601,0.06412582168371238,-0.04626707450558294,-0.1936850566741536,-0.17732433144777693,0.3640768735130628,0.0439400636087438,-0.06175749431111922,-0.1531194851856469,0.27007189964460226,-0.2779625085703815,0.045154891601273416,-0.023721620140478468,0.12255891952022958,0.055704023297693944,0.09376004071451116,0.08698322153690216,0.14032592859076679,-0.22777687860863047,0.036671660904552955,-0.02129717393325263,-0.02484105023661624,-0.021748866270566103,0.1713457792460425,-0.13620283796172128,-0.09567042073451226,-0.04631929534465558,-0.18959219922827691,0.22636373627321077,-0.0835037235587701,0.3324746266384642,-0.2330021065239138,-0.020527484796004972,0.05069017919631093,0.18915518451547786,-0.1900909211714687,0.062266642845874576,-0.0782525724728153,0.3417610935455207,0.3295874734016506,-0.015723977577818363,-0.17100472031725597,0.07207333301464783,0.13281938830104986,0.34820379947374785,0.40021524481756277,-0.1406338806626132,0.02295873392174871,0.019717415779399807,-0.1316784461800193,0.11965941358082899,0.0435860876054047,0.006228420040221605,-0.0713176280487487,0.10775367798807563,-0.07681813270221473,0.10397984556446505,0.12434564431675818,-0.21232849544194815,0.16790905409755044,0.030557352660096394,0.3463620937940733,0.1073249013158129,0.14497599223569685,0.015624818612275765,0.2606427260439455,-0.09855902777764969,0.10517854781497225,0.1461107832330653,-0.2319391456009612,0.013231326875917602,0.006886152166718029,0.02377401555804331,-0.12458581807731962,-0.1520846667581809,0.02957914694084999,0.21119707807936466,-0.10416835514629272,0.007626170389066871,0.0505024575112856,-0.01130825451753086,-0.01575369434381924,-0.3235048593972662,-0.26131751811972165,0.122158485557303,-0.18709730698344024,0.36163737842774496,0.05479151581609651,0.0804576570823945,0.4891208262002621,-0.21242994739740179,0.236004971579753,0.10202207859849058,-0.024188029088901327,0.01321107867184805,-0.05973864666414025,0.06689249759389745,0.11000615401114794,-0.15722134260275153,0.13113774395275915,-0.0009156250647713463,0.17191114081109288,0.2391873569257175,-0.03565073752502893,-0.06437742410079904,0.19333513030977215,0.020190947172801777,0.05797110268914795,0.009875754243576116,0.11027953764966156,-0.3209564340132725,-0.20178295430895354,-0.012921922581265356,0.16360399772578318,0.015554526077402875,0.25167750773665476,0.07175497902182526,-0.010446540422577446,0.2458722530870186,0.06373014411922333,-0.11106923825344331,0.10829543065328624,-0.11189125248537908,0.27546907595760245,-0.005222912270872203,0.16958286220084076,0.2091574679075899,-0.002857913875784475,-0.20993646960109358,0.08519923954278544,-0.22410473294289238,-0.045442556584398344,-0.03528338353741229,-0.23498332786262757,-0.1085261398245524,-0.1268721948923076,-0.08078114782955659,-0.10909685138777547,0.0753511364094699,0.34341801719476195,-0.18499256227450692,-0.15154529048541163,-0.028459907044311217,-0.07142391026811802,-0.2896849290865583,-0.0007353797644342803,0.1890983085757769,0.020949326537537387,0.34648184586454933,-0.04687965473959177,-0.06844714219705739,0.051260693521905695,-0.06472604205305783,-0.19207265363294898,0.13326587031482262,0.1275148732811538,-0.12627109394631877,-0.01225153861025777,-0.1293548050690973,0.1865534171519088,-0.1230268337213846,-0.12732543222403178,0.17466944170010967,0.39893920759843204,-0.037723167430972496,-0.17291182503706398,0.04747902157114536,-0.14591767265596672,-0.11635641874318843,0.2757213475536985,-0.043117234382961285,0.018581061601331477,-0.08479209906735274,0.311105250346516,-0.12020204061056461,0.05209977878342284,-0.06937963060935162,0.02175700311703238,0.11368911954814559,0.13079839840969731,0.09717164433446727,-0.031917585210340486,-0.15911028775067892,-0.03229937513684725,0.06104059731452021,-0.23029156895036232,-0.2717781087320957,0.12226734537379752,-0.14336640648234755,-0.17894687403624285,0.07889432757257794,-0.14320539861419015,0.10774048892142052,0.013806506540256369,0.006489530422949496,0.2767646507045208,0.20373389781601953,-0.06798874650341762,0.018647633260275204,-0.21756198902808083,-0.10668984817037246,0.16591083737380213,-0.003831118919203748,0.35868522091363036,-0.025054828226298682,-0.030312469788589286,-0.11590548169737498,-0.33465457566932916,-0.30684704763663284,0.07498417526268616,0.3412417117476136,-0.002532679309929695,-0.25929405782164544,-0.19522536787590322,-0.34858766701450083,0.22417953129988072,-0.1258781023419267,-0.21387353185968822,0.06904396966077168,0.2204051455047178,0.09249584456571668,0.2531985928842585,-0.18761531924904726,-0.19418506858712145,0.14723489495629127,-0.0861415353666742,0.12144671789975336,0.09026046795187646,0.12572311103894387,0.2518027128434673,-0.138043560480587,-0.04231616604546821,-0.1443462483470193,0.03016685711964484,-0.1829034892608332,0.48578049055613204,-0.16435044406148627,0.07589755064274352,0.1781877773798822,-0.001546709753913407,-0.11080581208673368,0.26091124496758306,-0.06313906627411753,-0.050403128497650454,0.004175956303078233,0.05758901645936049,-0.17542244731595322,-0.0405720760733905,-0.13269901845913637,-0.17441374308681898,0.012395061934384291,-0.03574054922010687,0.024677859136430544,-0.04159003884897329,-0.08491074188586555,-0.07464009726933905,-0.3214104736971253,-0.06032065814226791,-0.13264042497636822,0.3888195429886659,-0.08469663347774009,0.07936997850278484,-0.0027440063006953434,-0.01764800351271955,-0.04140109121833705,-0.3555822082130176,0.19696754081769313,-0.15955551963373285,-0.17770301702932464,-0.006448427494671243,0.04888983983184811,0.023866043602425873,0.006494321438768517,0.03022096374212853,0.18309974258358966,0.20452087511892642,0.22259508974139353,0.18326949932790032,0.013697230297518705,0.03448413382084766,-0.2296507185071143,0.2811694499828925,-0.32883541377179204,-0.30947416501321573,-0.05950343560126129,-0.1379633622995231,-0.1258542876966788,-0.11903324491170633,0.14352383133342025,-0.3118485478090127,-0.048049581308129155,0.18350882157271456,0.09046268635960138,-0.025957120933563182,0.06728155387933903,-0.08254578677268445,0.052953397427850504,-0.014957260560038543,-0.060216983356112336,-0.28274205029634497,-0.09356633494539565,0.08125061737407716,-0.017328095579486426,-0.21255354812475072,-0.11437365749301982,0.04711668442544478,-0.15410350790978902,0.07734137420535335,0.030685451383158358,-0.16007448383904055,-0.016563224227801078,0.35923506876255007,-0.10804862146241545,-0.018121653297534936,0.44312032246251337,-0.20006087137379663,-0.015175246276386204,0.221430852700459,-0.18580860284749698,0.16395910688015833,0.038358528463285246,0.13081360959043303,-0.05540848394871439,0.25054347023791657,-0.3213012587841549,-0.061846700670006,0.008626803595904608,-0.07821785214264235,-0.12083998159493615,-0.09618188797520394,0.07474182203971973,-0.0956056887693041,0.39565111240393575,-0.0002218231294324305,0.23741753834781976,0.06537827771220299,0.33558884417269697,0.22946785957612548,-0.030415945964054576,-0.32385715025777917,0.18198589238626892,0.20748121577011167,-0.2958421876817476,0.2734032804427919,-0.12684130295778007,-0.13098222255320793,0.10295610350776326,0.08614002554051267,-0.3011792894911817,0.03864363085822052,0.14127827730697032,-0.04348425503701479,0.1494001674964778,-0.08254676246970212,0.06786641706310081,-0.01700570839381716,-0.004868131076064926,-0.008914010992619067,0.12241133174993056,-0.26861823951751657,0.16367001468602646,0.0945370252921228,0.14516671578501647,-0.014043681287909221,0.07052182912938779,0.1804330203748174,0.08408538252159632,-0.14089657417176257,-0.07716423476462007,-0.06986899801532491,-0.04315627416815098,-0.06996723064306001,-0.09692591987926183,0.0258617847728287,-0.17167098792502858,-0.09252240240512324,-0.0314196133091087,-0.07669443001022694,-0.07943450895091635,0.21237924153097187,0.08852438017403959,-0.011239335453488923,0.14622130589385524,-0.13154751569611048,0.17945505352430838,0.36141338249463045,-0.15792076419533732,-0.232553011182512,-0.04471775263035472,-0.12908166010643923,-0.10196455691724103,-0.12220395422217799,0.220235059698119,-0.2857300453531771,-0.07154911816400045,0.08703808737893097,-0.06456916319533203,0.038933391405216115,0.026736377024490058,0.13587863366731034,-0.08775139574593807,0.37073604883929373,0.10729923349502837,0.15486667891721995,-0.0030004124843400217,-0.07981340853912001,0.14816289035062188,-0.12584239223934027,0.12800668957046504,0.0361037407787759,-0.16518213337938237,-0.013436608499932811,-0.14421081244645764,-0.14974861351593952,0.0635510475510048,0.058061170004826544,0.02579462942242347,-0.17930192939610104,0.2050294362952606,0.12005971140521024,-0.04750451761358237,-0.20269187467404373,0.2055343319230071,0.003603931631865211,-0.22835072377040794,-0.23513217277360432,-0.097343846588858,0.46297552435931655,-0.20434557484600763,0.4887006595803765,-0.15171431069953248,0.021610121921438186,0.09246429129737704,0.06731299408366047,-0.11875262412491196,0.020170117314407727,0.28789766523036237,0.40200193475188994,0.10510158462000264,-0.1281659089827007,-0.22177250108205843,0.047765506709757204,0.24255154697395123,-0.17054684479120988,-0.2108142198117595,0.2755020482924293,0.07380136056477067,0.1440559548446843,-0.02494179778298962,-0.04634857513997144,-0.1260589954335071,0.012309517960915327,-0.26454699542357585,-0.06359869177631106,0.09721268606709503,-0.07171786108832912,-0.17969992339909618,-0.13526462916663587,-0.008595501833063417,0.2940613525340307,-0.17434655863793203,-0.13798899280032523,-0.09730510983309609,-0.14672397151838318,-0.17878809626810416,0.048761899050191466,0.0252823574598791,-0.20431188626510194,-0.022401587946007346,0.048277086883375994,0.2488381640500604,-0.06972438760944792,0.00218623698959996,0.09032947066387592,0.20229284416036936,0.027463900812123545,-0.02676651074411543,0.25916324252622874,0.02977727475941878,0.28562852239976444,0.15300064426812662,0.03016155110105856,0.08613552299053326,0.0654492091708,0.3252691358758827,0.2820604657207182,-0.08214881357297556,-0.10905879107535578,-0.2506811450641024,-0.20637968876286483,-0.12074336374536387,0.2996941176204255,-0.13574321507152334,-0.00863018754377686,0.09042240201316333,-0.04876362816850545,0.0630231786756061,0.1666979009194113,0.18355900737793981,-0.12015908162146993,0.23996619511552655,-0.07028911582868394,-0.13208280290629348,-0.16999144765141633,-0.23360458329657194,-0.0016781187483211192,-0.11886993930868114,-0.28119458638485245,-0.09670101415680053,0.05073428017026585,-0.1313212871969551,0.11372152842015171,0.16770611641995395,0.26183676661971017,-0.09413076848879466,-0.2925353622656803,-0.10460757372054043,0.15133370272702587,0.03844668688607612,-0.10884989383750322,0.47063523627010156,0.010724920649982877,0.4456379363233023,-0.3356696358266775,0.36287515474785326,0.13086720444882402,-0.10413359879612936,0.02977017541991997,0.09331873030835822,-0.025996470079697947,-0.02606014552016245,-0.2667127682227537,-0.21636735352290815,0.03686344404745263,-0.008079553486594334,-0.07678635466242231,-0.02170702003957822,-0.26576480015547566,-0.1873590992668965,-0.12050455960414838,-0.08086260148262997,-0.158009727839928,-0.15496646319427376,0.013765172552671276,-0.05001354975917471,-0.03388913130576936,-0.05358455537467104,-0.12988959918078724,-0.2398942875573493,-0.09804390368940245,0.09980488166104153,-0.22088800403992845,0.2486033967123237,0.3698002701796916,-0.03958087127468513,-0.26400456200887396,0.09017690923040865,0.07351156187964474,-0.0024492452904942056,0.03614467766456719,0.39967592525619067,-0.01949381282525524,0.46702974168186095,-0.07259662320291774,-0.017092381767754305,0.09435808630191035,-0.09681227009201865,0.01349795390487067,0.11489970842003536,-0.038582397722506025,0.1330790061322354,-0.28506802407917137,0.12316587883983049,0.4871986038032522,0.09605865954204909,0.24895612496034278,-0.32272225081647193,-0.07785225806664067,0.12994320987833083,0.1310348884643275,0.010791819655700651,0.02242876766717299,0.08929162653880968,0.04346435851516023,-0.0706533146167131,0.07245224223165109,-0.057864219500221296,0.41092763998317383,-0.07333434047472298,0.33633279400176475,-0.07497508920294708,-0.131370044812848,0.012831334419920558,-0.11274357287274259,-0.17350731617392645,-0.32049424367654483,-0.12898525531658905,-0.21065305537174664,0.02405065207408674,0.10634232028837774,0.0963114305119091,-0.02186914936533031,-0.21077176559763466,-0.08115670563965863,-0.042337078110356204,0.09844119424854213,-0.34558631767356307,0.3470484670613643,-0.19383071488906228,0.024367881984415293,-0.01714008583182397,0.1444310102038253,-0.20760299869663676,0.2648132460098486,0.35100694015961476,0.3023628064757646,-0.019303259814840348,-0.3338271058904636,0.19880642441940816,-0.11480588598771337,-0.1164616559391506,0.06520240183503913,0.1086093952183861,0.23927804258001556,-0.14480764818192327,-0.13978256696251704,-0.16617084283604772,-0.35542416064105586,-0.09611202755418224,0.015247871172947474,0.04918066069009646,0.04201611705416226,-0.03737317277291225,-0.11987474501047261,0.15986614126787943,-0.3193987516623098,0.35796447474744447,-0.07308268017843446,-0.01949299236121771,-0.13541495115363078,0.28808259085426,0.13553485606921187,-0.10770230579099278,-0.14494361030323272,0.19945263466020294,-0.17016723142759477,-0.37065324616707757,-0.014544716037738525,-0.23079947383354366,-0.06348922347408545,0.3798820801994582,0.18690894638487998,0.442250735829779,-0.04366143006871184,-0.21855326464951944,-0.038064357978967914,0.03968097472690786,-0.04485428980766079,-0.051767956647754326,0.09735961244898923,0.1309651324413578,0.19530104740898951,0.0704461054040977,0.110961302097543,-0.03302451990444489,-0.2894653991423526,-0.0029273760695823054,-0.26027052107238063,-0.15632624434076478,-0.02974877518254308,0.2327546331656402,-0.11550614232955837,-0.14221197144927974,0.1597199981450665,-0.0539377477319486,-0.015364416002122952,-0.03818690575377679,-0.05819551278677374,-0.3396573002729153,0.06137744023978568,0.2297916558574602,0.012684930732477563,0.02543410221382111,0.08644282139757449,-0.031223680609123106,0.24471021584584227,0.008802948131490587,0.009487972886432982,-0.005899058022472343,-0.06449754231180767,-0.025568956432804543,-0.19595950121128153,-0.12847837152980732,-0.17358330041808345,-0.04908274591647552,0.028816352722482914,-0.20041526242930097,-0.16212141871859498,-0.3046478594325334,0.051416766126923995,-0.323702929269462,-0.18087345263563093,0.1240887128856444,0.37582607528298845,-0.10772337654027528,-0.1577370473577392,0.17129621252266827,0.2597367543662589,-0.08395960852472897,-0.0894114932594975,-0.06487438634115845,-0.1821956643679357,-0.016217960477753455,0.25428693579278744,-0.157453754310736,-0.07039036541276408,0.11449536397720306,-0.23468050650402097,-0.07301082091659664,-0.01932426622580299,-0.20439720000306286,0.21689295483925808,-0.05650510615200389,-0.03765884930859337,0.07876222908664547,0.10897915162568707,0.1919617243465892,0.07387856706975159,0.28619642100900183,0.05642485767533976,-0.2484697021767125,0.23081522911040245,0.08739131242161621,0.487368607607533,0.24192450220786552,-0.18798420242560007,0.19679966274823327,-0.04369381839503055,0.06274409567985083,-0.213657521667713,0.030170499615402534,-0.23168555030569501,0.005332777009842868,-0.30563039477150006,0.14659342742051215,0.48872383052700347,-0.029326497691755417,0.21808659714135803,0.12692339090554478,-0.19610333927265255,-0.07892033601107955,-0.01235841316960596,0.029901484313088908,0.18282492157948785,0.16818791255674945,0.20804380025245311,0.3572865137556951,-0.261511298775975,-0.23422665864781847,0.1286930254463069,-0.12278538068095247,-0.1602351821723791,0.07044601013914203,-0.03720320701837074,0.04198582306757099,-0.03529336683590983,0.029947338897480958,-0.13535657810507323,-0.03178029389831667,0.21052456923283103,-0.09858366059164463,-0.24560570599467305,0.010577461980325758,0.15311273323736516,0.48195965428157383,-0.11508539134510797,-0.026920047195067486,-0.27569184137873703,0.21681035662288095,-0.07536183813798093,0.055652014147452675,0.05106718224491121,0.0495060202620171,0.08204382072618266,0.05070762625426556,0.03234657926858028,0.18794012162458987,0.2001166392358285,0.16024403745976495,0.4312412381601515,0.011840776012122313,0.03402837945766753,-0.05055500299406693,0.34662810725535126,-0.164104127047711,-0.24509894869846233,0.12474030068554628,-0.008933449973909513,0.09791765558203899,-0.2990887928939962,-0.15743844551063127,-0.2193684641162814,-0.0608267888690046,-0.23852117767204015,-0.18177143398082715,-0.27832709679303247,0.16561629050644103,0.29938876453158214,-0.20691698895584232,-0.044324284004378836,-0.05526014666470777,-0.27356237012440304,0.23567476729172238,0.23616765289221797,-0.03370142506851447,-0.1906383637483909,0.06036344660030947,-0.017029023394351653,0.16391521818248497,-0.021238810474944015,0.12099259199288001,-0.025514277828133376,-0.3018669976770966,-0.12602752171678672,-0.18997848625506952,0.004476897926850069,0.027398552736873243,0.13720871871984977,-0.06470488840281244,0.23067706177035754,0.20422035271966965,-0.01335985362627521,-0.13667356163419264,0.2823165965252378,-0.12537775521966424,0.16549831888646827,-0.025329478108889424,0.1398332783742318,0.24471092726652774,0.10961473958441426,0.09256227580313806,-0.2180590161755381,-0.22455778108997138,-0.1877392263070874,0.06296652242532327,-0.27117663657521296,0.22189158080924026,0.08773952215022648,-0.12768474212148115,-0.37100854087452423,-0.24976825429980928,0.033068739887749846,-0.23384900694721972,0.03181990010793545,-0.34802107475871386,0.053498413515722244,-0.0012691698838734878,-0.15697607207485154,-0.02964234650707718,0.00041795576379774816,-0.006419699147279462,-0.0496421445755318,0.046973927712243575,-0.13923721403034733,-0.2670731708851863,-0.08264802600768169,-0.1622305633916149,-0.18603179955388072,0.09883700909227588,0.15460354157039058,-0.2666376792363998,-0.2330385287321598,-0.1933354920372814,0.39062959775431155,-0.12441108219234139,0.1406267992717727,0.20367827761988702,-0.08429150206041346,0.02932183550887819,0.016847932099307572,0.09828495755833996,0.022528762929970917,0.2321721862154547,-0.011998118006243407,0.06130156916337741,-0.1419163049310967,0.06994400725970487,0.14248349979131295,-0.026508001107827767,-0.10012278366349481,-0.08210933725307712,-0.09450116028810529,0.09500386773375459,-0.1177922522886895,-0.0744372416482329,0.1653566091296278,0.1984682716523678,-0.018868807063182853,-0.003640497117593293,0.284659827294767,-0.04270820772333485,-0.15170434564765173,-0.07550493086472067,0.4183320282566738,-0.030510212205083927,-0.08098263154997673,-0.1859543095381802,0.17367387418786387,0.20813869102665428,-0.16243843560875137,0.12654262659189014,-0.18948306020919567,-0.17028864210831313,0.16453475718980376,-0.15366756619058208,-0.07597820712077082,-0.18759150269227415,-0.19887952836474668,0.005295114950278327,-0.07094500872515713,0.027992352277853814,0.028356691669893595,0.45105082970938215,0.17153970345436687,0.09555804074487886,0.3081526425334931,-0.041348152660319215,-0.12139087597048566,0.30461431586567833,-0.06569819325041093,0.3573456841457392,0.18579325454361661,-0.2474469640491923,0.08216009046454198,0.008182633807745997,0.025987604510963644,-0.13032231343125064,0.2184108312516283,0.17279330815706667,-0.019182635442530998,-0.08483377563075124,-0.03643857133464509,-0.2726203475691861,0.07781430158740792,0.05326193666447016,-0.26827476792416277,-0.04351168402955837,0.08618627740695384,0.2099194341879955,-0.01104615800255552,0.0776206840061818,0.0037786148623247954,-0.1609884889146704,0.03052336968350005,-0.047576597175504165,0.10914149663368833,-0.0009582227409289099,0.17643518552012605,0.0535685977216443,-0.1590409784379577,0.107209486158971,0.22051078734157212,0.10275710934412918,-0.09823020467182525,-0.06009342394239082,-0.15312489040002863,-0.274343721105987,0.13238992339807887,-0.052411065910105574,0.25727421357913566,0.25829824286279524,0.021065485954283425,0.014473187449175896,-0.27666243674257307,0.05373374901726671,-0.05763630710348958,-0.16239447421255945,0.13470050760915558,-0.019433525150243134,0.3033050193198137,-0.06701488883080756,-0.10665419819663152,-0.22451061779116938,-0.06647810089396362,-0.046878507464992954,0.23450976147652425,-0.1341553258356764,-0.08393628122806336,-0.08180320860531058,-0.10957046567842493,0.21135226426098394,-0.17147718477415988,0.2415946683730607,-0.1588714841495546,-0.0866435735102444,-0.24068040771058016,0.0523077227077896,0.17267148015397563,-0.22633452438469984,0.011270689571982583,0.23228362723016804,-0.03546547145967694,0.23559556300848158,0.1615649859765697,0.11269756733069561,-0.18415714745675388,-0.15150465345915765,-0.020791624662462725,0.10978700295926924,0.013625446298038489,-0.04967455999137712,-0.06697206864784981,-0.027726107729069936,0.00045046083146049916,0.20508032610179738,-0.04947956375808311,0.18647241196757647,0.04796647568885353,0.10771696930581126,0.2029270456621941,0.37489612462122207,-0.07613543102587214,0.29662716676110745,-0.1410065702034105,-0.01908039450140953,-0.0225012435811341,0.2649621198966253,0.08816134302094308,0.22023325396009077,0.15054539215526064,-0.2351114404349588,-0.15553897060068497,-0.26816938800142964,0.17191898553620885,0.17166566369601102,-0.09726611086275348,0.11052465388714576,-0.038725421151371044,0.0020896492576184533,0.03891787602051174,-0.09087748406572038,-0.024077475165592534,0.10697411511252909,-0.045038873816913974,-0.10463010094095523,0.3702522499031667,-0.23041604063758278,-0.1773328730976725,-0.14867115024316563,0.2050771707414284,-0.21110319165993127,0.3826420970231575,0.21644419887928085,-0.06007488687237477,-0.051996258686308666,-0.04862564576085238,0.05211374788781884,-0.0661094538462841,0.02479122728382587,-0.1379797839478596,-0.010956465967523643,0.012674311289198825,0.36927064858349623,-0.1994927752715988,0.02087809204060454,0.046222484081748484,0.28645657232918614,0.20583602234901882,0.09480153639164762,0.2435139387579,-0.05873094024105945,-0.1758095977054834,-0.1404920886494887,-0.07819256358012093,-0.1358335130726279,-0.04496247330256839,-0.15617976383741622,-0.20456996848790057,-0.030908609052352792,-0.2813782937988271,-0.32252261413538624,-0.04410050599717474,0.011071457777113684,-0.06548607728799688,-0.054586443564770264,-0.04877487359146863,0.043341555238495456,0.07306720214916222,-0.04461023461696635,0.13520155751598434,0.110205938522188,0.06449283679240556,-0.0399250058642479,-0.045588157527746524,-0.2067625432351973,0.007757913670876352,-0.0021708034907789654,-0.19691536239072704,-0.012176605172320104,0.2532469384765859,0.04097798581741613,-0.19953763572761005,0.049652311787233894,-0.17350255998926797,0.07280348776391153,0.13290725968038042,-0.1251100564949091,0.11938053892611798,0.06740211021877791,-0.20826608997321952,-0.22095618526945196,-0.2867181403920464,-0.07867624405404741,0.03255926664522183,-0.04241357532168821,0.02867552653917878,0.14232623500292108,-0.0751234724549056,-0.1649781768026962,-0.09365214038795344,-0.005222316812415894,0.07675890600109482,-0.056443958446056874,-0.040701961094138074,0.08237660402760577,-0.05670249179449589,-0.197043596269545,-0.040418302082576886,0.10716873530383392,-0.10341782793179784,-0.1377704676949081,0.14267875452461512,0.3689685731420588,0.0819878345697164,0.01370452741818474,0.028011333671360824,0.102244079037033,0.010503422581612283,-0.16155599654927513,0.033201102946581136,0.1671346862566236,-0.15475434508803113,0.4211155895314882,0.47210876019083725,-0.045230762921013135,0.23560095181253174,-0.1665948076886877,-0.13996756974315247,-0.10480633666752953,-0.07679117633469872,-0.19090876781779373,-0.12293176398920001,-0.084752850745237,-0.0017705508844842821,-0.1960025927305138,-0.1028960066479365,0.060620518295712744,0.1996939400603167,-0.09297140757880529,0.2613195136375801,-0.14226166555654857,-0.10061150065819666,0.2022579083898968,0.09612889582893114,0.05039737106847803,-0.3589195846745657,-0.3593657692744941,0.1434553216140675,-0.17005759027117126,-0.06219414457487384,-0.031831106541911874,-0.251605361776448,0.11501484109240999,0.029424325409429943,-0.05644727409007727,0.08387871383293256,-0.08814698858489439,-0.24334286279888342,-0.17224612258439867,-0.1566380785761441,-0.10202938893143106,0.07636606407578038,0.31271306421147604,0.20858172362957467,0.21295760379095485,0.0011876597665334593,-0.08365508630253973,0.09755836341049264,0.06132922793917744,-0.21437489253880668,0.1307080152742428,-0.02598969382838191,0.0649311939135618,-0.1802219852828256,0.1468036144531024,-0.12766351861401334,0.09805251393696379,0.48511435928864005,-0.029585692304128174,-0.24440930756757462,0.008292209548713512,-0.34802107475871386,-0.27249189914516725,-0.07466590216966487,0.16267075438151363,-0.06535308402937041,0.10728373217079491,0.01438550181976345,-0.1851152540199388,-0.22593727274809955,-0.050397418430819914,0.37146290878191673,0.4532032699484919,-0.20459642554594834,-0.08945784597547707,-0.17756543536989566,-0.17900951532866885,0.22686932052379608,-0.05146657744865032,0.14511053483927985,-0.13545103874475084,-0.10464151670810454,0.20179671155691498,-0.11761908462620517,0.166164924166806,-0.16225546293836512,0.2648329457187582,-0.1346199557721303,0.13119129058706147,-0.10161832996492697,-0.0029449104422128594,-0.17314238307127716,-0.14865293803595797,-0.09160696462556972,-0.2586495736764769,0.15456074743871775,-0.03150420021932203,-0.19803152621279943,0.051579673385555795,0.0019115201769675314,-0.2475072391939529,0.12182394079053502,0.1519383278291809,-0.04874427381618411,-0.022758647988775013,-0.21244945804779117,0.11284769483572915,-0.02888905782610249,0.22743162391792654,0.40087834751824314,0.28777903061398347,0.12737520950145947,0.08031453732493583,0.04960680909917373,-0.07988445628709588,0.31075322465360744,0.19192704619416687,-0.3206578966709297,-0.04957208076144451,0.011844108593076223,-0.11240629074032006,-0.020796984403175706,-0.30739296268696703,0.16062392938566325,0.1683564710861971,-0.20084085331659404,0.003163215146933183,0.012887260363244108,0.11012652549267302,0.050808904592708516,-0.23502536512234098,0.13594478355040907,-0.00566562249827902,-0.14972249037264748,0.3065128835290466,-0.05986559810936345,-0.1330100555838477,-0.03561351927300567,-0.06939394241449082,-0.024448240844211506,-0.01895617339542054,-0.17605011451575053,-0.03196960778119504,-0.12201416053912761,0.43307580870506757,0.11597581007012832,0.1449145712313954,0.12660484953698925,0.033773495595164744,0.04058840312100464,0.06353222055673154,-0.06351337932017434,-0.1438407750740174,-0.19049864698748725,0.23666305211532981,-0.00539448422531356,-0.08518011525616542,0.2654294835784406,-0.14011781105213875,0.16038887173384236,-0.005152344334604442,-0.0837377858011168,-0.0632756828905218,-0.12473448260768875,-0.30072666735098724,0.06736040966952225,-0.023528568158152744,-0.018260380983798598,0.18196821171384778,-0.030884713090327896,-0.03810950188092836,-0.31147184358920227,-0.2879548080939589,-0.2129225083321544,-0.08843515990828081,0.002513851278513906,-0.11749342450089631,0.16663140169825383,0.1694818023740164,0.029561968302066,-0.11323338543495026,0.24463127644579696,-0.12448019386846126,0.4863982772029714,-0.13175384300122864,0.10725413281821612,0.26028551796760874,-0.14036205646753006,-0.15927591768587432,-0.04571632237806251,-0.343931273889434,0.05438434616926249,0.1829147502276185,0.22408979220137898,-0.14132964906669593,-0.22047593686622396,-0.0019294888738322256,-0.010795747640804708,0.23935295742261414,0.22952954975919024,0.2023281574225303,-0.04611109373331732,0.34737273485003867,-0.11996757748782068,-0.16340060802288575,-0.13754580256211668,0.23672475531289378,0.0647632773673479,-0.10422161372750006,0.06433191126988438,-0.16690425874760723,-0.15229033525704277,-0.018021022600329416,-0.011016613694712617,0.16253300158669903,-0.10531113268300624,0.1635802079917521,0.165482894475851,-0.12140578343502094,-0.047650950625116484,0.01677658438361915,-0.10678435104193187,0.008459601713532319,0.05300841934683109,-0.006747264832251479,-0.037483582435849885,-0.21251632581481986,-0.22312619988187785,-0.21399062455135237,0.12235850930580305,-0.16745515043688775,-0.0315298097059902,0.0763775794675444,-0.028703531895665168,-0.08790360840471616,-0.09706416327096427,-0.12612142151547745,-0.09232744470091543,-0.19990748136075884,-0.07560867267675826,0.049187449732015255,0.007885299335864179,-0.0748937270925602,-0.3292425485370258,-0.08492938853393989,-0.3152630644787029,0.2141371377754992,0.10437348166213288,-0.04283087494845776,-0.16102888525638542,-0.03754627760312304,-0.17258104067143737,-0.037943861527524846,-0.041057400163610404,0.2228771483426434,-0.22005762892288036,0.008025369861354338,0.011786994016999718,0.02198153350109781,-0.16506035536967203,-0.3186378081914352,-0.21709087199596336,0.1075101310106617,-0.06937661548907102,-0.3480305622772761,0.4878452046700101,0.15883400491270985,-0.17919482323204922,-0.19845225176084363,-0.11156102029055681,0.019429622584755744,-0.09515989513824732,0.0006608028813951046,-0.1177398400117611,-0.08349311042603404,-0.1567589016320555,0.12661690533078826,0.006354600690127077,-0.05549781536282278,-0.04199897788910641,-0.14542093156736086,-0.3453385265969935,0.2267364986702047,0.007243029636359056,-0.09235320956280631,-0.15641188138446702,-0.09532450913975518,-0.1823623829754825,-0.1525366999559243,0.14652192767620556,0.23920420898123534,-0.11032222210903046,-0.04938053756601132,-0.04397068442056124,-0.02574564005936892,0.19954864811898324,0.004011338182101787,-0.0840145762573652,-0.18590750582273088,-0.05450374055976881,-0.06293710201643157,-0.12665865598082626,0.20073673627408473,0.18936115750018345,-0.0022566738826344723,-0.17944334307511134,0.20075109034490796,-0.009514681290139508,0.18510622665621076,-0.060400106649201535,0.03356616107937872,-0.18404140347666195,-0.3105183965923917,-0.021646482757808515,-0.06614682991910494,0.021969728296311452,0.09114154251193375,0.1269018391134147,0.1329597659615861,0.04976271256274879,-0.2884812130299702,-0.28661321754672503,-0.30040355604085633,0.356786065834218,0.005686220989358782,0.0919180947848072,0.24956703664347854,-0.0709509650487062,-0.21225449765587168,-0.2082574417969569,-0.1912980730598191,0.28517642775350555,-0.09950856458257318,-0.019318491441859216,0.02266716502671612,0.10286248844337668,0.012678699377855079,-0.048333863718121844,-0.061153547061249855,-0.13755593762728213,0.2248267089016808,0.02071885971452064,0.08285600939460551,0.012965358021633218,0.14394162843212846,-0.14711837902054595,0.30683807859327145,-0.2256174763411831,-0.05821991619959146,-0.05302430667530196,-0.18323217094750646,0.04460603080415783,-0.0703719091798166,0.1058862181935745,-0.014911596245982671,0.03674558235438178,0.12115301559147765,0.03010282219510829,0.08175703033145561,-0.025613622131346905,0.0879702616523512,0.4737436175216216,-0.04399629610545162,-0.11989981913774579,0.256414672865093,-0.06198588619433739,-0.10534570082738841,-0.014909188515905938,-0.009101138738615275,-0.26230912587282196,-0.26203404983899525,0.192651225913072,-0.07156917185090558,-0.0915148926319815,0.3527113018451089,0.22164777171335148,-0.16378985492845122,0.40870124422121257,-0.12392579702955163,0.1868241776075368,0.20315192724612063,-0.14058117301418482,0.128336249042253,-0.07195805778108154,-0.08440176082578718,-0.12327442154912853,0.03347767389056836,-0.0944550111149786,-0.13918970469533062,-0.05842538759346768,0.05557328021859694,-0.27795942985366523,0.3891753085156638,-0.024143520235129178,0.000257311248118611,0.050904220399700495,0.03907190999690901,-0.24170712928974988,-0.09045460287150137,-0.08308533103544322,0.06836211879220083,0.17179446296871637,0.025023479857796058,-0.1150278106905009,-0.09873431131514124,0.024153167763311886,0.00858603751089638,-0.061069531289901946,-0.011456794745011526,0.10332828102806373,0.0433385052206144,-0.17212171352120179,0.0940906192624437,-0.08410533760501045,-0.19217128076254283,-0.0070652642426719025,0.04510339988333259,-0.23025809076312898,0.08757117346587785,-0.3541604309197967,-0.1292798219518596,0.0696396857304156,-0.05460612293705259,0.12294361911852411,0.061000246080696,-0.07399986159891203,-0.1925104524097647,0.3646650142925663,-0.1700433320256179,-0.2147554864597258,-0.030640617266616387,-0.2738068112344538,0.08507285428968424,-0.0429534543849886,0.06298905848324501,-0.01592477748461793,-0.16073568354461004,0.16665223249688058,-0.028791591824885086,0.13675920689113982,-0.13036961929626067,0.31877208648419386,-0.27584125712081203,-0.07941586051855572,-0.2579563571826539,-0.04911425083648817,-0.00835881132458232,0.04983447402565551,0.04351206147902189,-0.037701161289573466,-0.08043821631230677,0.28807810902548264,0.2632649027877633,0.3913147569633303,0.15300109752622454,-0.1747383006381852,-0.14693555302562475,-0.10780656035567042,0.16640139052135006,0.09065659003161106,0.06993939236013333,0.268839555661643,0.009219834103743628,0.197706486814287,0.08279151771635808,0.049415369305006986,-0.13671137113054466,-0.3004346287809083,0.24024677993474677,0.06252312665442633,0.05430791838493823,0.17057770632855812,-0.10822901362137771,-0.11739826405562112,-0.17101573324942052,0.17176163129137045,0.1656309112275643,-0.07217597449939117,-0.21279529871003325,0.080375856832203,-0.07939269125474771,-0.14459395524338509,0.10583539316060903,0.1435085402312159,-0.06882060557155593,-0.3703362072130992,-0.033073394866869416,0.07075599343710863,0.4887834054631655,-0.3120606894137401,-0.0052622260591482835,-0.07019163077420436,-0.06080448976082282,-0.11723708856175832,-0.22607302282155733,0.26124309735347734,-0.023487779708094663,0.20778917096124822,-0.048084419353803275,-0.3689983041381136,-0.04573939826427318,0.027248382435598344,-0.22901676437982435,-0.04678929053976532,-0.0997502791291395,-0.32575869269113894,0.12138154006312944,0.19611826376261432,-0.02336255647464417,0.20180814772919264,0.0700478390373463,-0.15936044290161122,0.0211520694267447,0.21421306286764846,-0.049974579170697093,0.039730775474876065,0.0398534314969005,0.136866226008919,0.30293563643246757,0.016294522837533725,0.1088790281557163,-0.1415942736489614,0.082160022074901,-0.05439522442891072,0.03416010648635153,0.047319370177795694,0.4235302719116637,0.07040112686288401,0.021245471005916845,-0.13753299841747793,0.07354229574516372,0.3050970220438224,-0.020517800000537708,-0.18289858456024943,-0.022402100726838632,-0.1749467547731654,0.15128704796936104,0.30767587331270524,-0.034762249211185405,0.033244995799258756,-0.12017286090431392,-0.043368215996754694,-0.1946043264371115,-0.10023482987737116,-0.276069068354842,0.20799083272378371,-0.20124757154677933,0.4651100027233984,-0.19226656918859075,0.004818378962692983,-0.13692103184306284,0.04415585921206278,-0.17474410745437613,-0.12418552277236196,0.09058612469103249,0.1611348259471568,-0.037676041876998696,0.10307212807482391,-0.1987585581754657,0.14512437726102306,-0.06916931672470832,0.40547514295826464,0.0643138871413767,-0.3032065913920566,0.11193979831655354,-0.19819645852652004,0.14209379821478957,0.19738765335643677,-0.0237575549055406,-0.20061686482135896,0.0722957343332316,-0.15829343256386247,-0.09992680010786938,0.17518097747824932,-0.3072031020444907,0.04802792937548971,-0.10682108540613651,-0.1894581167061893,0.3552055597652641,-0.12717539605412903,-0.09855902777764969,0.29113135105145643,-0.1410878670307196,-0.18011625508120316,-0.2519973551309896,0.3627005201223942,-0.09953401565590851,0.08070466954916812,-0.3034164810170296,-0.003397057020564482,0.03567725884704894,-0.04214186398353779,0.10642499875819635,-0.21208130614688397,-0.05705925195427224,0.0675027400486943,-0.2250067668976716,0.18935776509606667,0.10853270943602956,0.05503429979784708,0.1896624405614964,0.02383038377783057,-0.0008867589226360112,0.017986506536008928,0.2586891431546451,-0.06475169512179249,0.09204934799562604,-0.08101137204231765,0.215630274381607,-0.013623915960854088,0.015278088734199426,0.017646044260752412,0.12087978388238108,0.2970010430034892,0.0006791529201296339,0.0043338571659741415,0.13894412890925675,-0.3131697650552415,-0.20443606398800976,0.30947271164410456,-0.07162245795773078,-0.046698102094634465,-0.15325007980861843,0.026295541558101446,-0.11102628343826675,-0.039994151637137,0.07571820375855216,0.2698094245248015,-0.10089044104397293,-0.3326652475111846,-0.12481481648372826,-0.02938284243645676,-0.11637797036771884,-0.08107271501508184,0.1364890049525826,0.08260333185301108,-0.03830387011451318,0.05239320019133917,-0.09529979439797659,-0.030728538293837006,0.25588075328272153,-0.097685176680131,-0.1174662939574169,-0.12131826391450648,0.10118124993898688,-0.017048270258590258,-0.2531230550205882,0.16233725379186947,-0.07054487383652039,0.21865488454405987,-0.09677895666449991,0.18854606809552554,0.11733958010696668,0.3185909715062551,-0.13139239268434139,-0.0657665290264936,0.13370476679130475,-0.10745457778056325,-0.21442195404932757,-0.35911599484290774,0.1754479004987777,-0.11079840896598732,0.36832331740821206,0.25437068501294635,0.025194513452226893,0.1391826626087228,-0.16847635976276282,-0.1126406513908227,0.09100771610222302,-0.025522088810449112,0.11219155522701316,-0.13200902058826638,0.2568469238555602,0.0567641918711918,-0.22380539273091407,-0.1293548050690973,0.11441754719664154,0.02057246058764429,-0.18589975898882408,0.4878404290152307,-0.05178862059049833,-0.0434033177954827,0.0400692803044732,-0.21330026894393506,0.30012146705592446,0.07634726228922872,0.1308182393760245,-0.23223699815167076,0.033157708338386695,-0.2329584793056325,-0.2071513994251262,0.2509939945455801,-0.11618654311781329,-0.18332352484259454,-0.35576192728742534,-0.1382614134465307,-0.0989211840511148,-0.016919418798593263,-0.1317289647904637,-0.015668279043687717,-0.30214982259592454,-0.11915455426236879,-0.17494410701261084,-0.11559678460157527,-0.049874820468065566,0.176947382911727,0.2117353829989567,-0.13476448124414392,-0.1221614440635217,0.04346016844101416,0.03189558137189817,0.2020116824979428,-0.28749904655313474,-0.003562305222099603,-0.1256713302486265,0.010028061448007058,0.014945438816234374,-0.22973935727759248,0.008585955566016802,0.3622715589070296,0.12577352907845485,0.04553677320918925,-0.0550583271611705,-0.3543982588166713,0.278080957347063,-0.08282365176040124,-0.05660351719330532,0.140470809603494,-0.08073548484810526,-0.05663747928959298,-0.024240725163080092,0.010544621100932804,0.0865651391669432,0.06467360406013427,0.07884951798758329,-0.21540169588574426,0.4797824034366655,-0.03652724976248685,0.2363367506303202,-0.011838191870665482,-0.02603403166338884,-0.053350907409965884,-0.0618055904203077,-0.17780453901047047,0.046625940620749845,0.09985950459028008,0.004886819299955536,-0.03882655129584724,0.06464581273977654,0.200331724465645,0.303161374725977,-0.1168000183013486,0.11721605702940027,0.3514792836743346,-0.1504721817510639,0.05249825273695425,-0.14081693184632052,0.20993738978157403,-0.20168322514063833,0.2089118288689097,0.08123708463375597,-0.02178829384596326,-0.30676847536329116,-0.034134578623322955,0.09509906192863081,-0.3368598033597963,-0.3189438437002289,0.12160721905413598,0.2082199929898646,-0.15152214710445228,-0.0495276538072134,0.1383823848623187,-0.1163470662017671,0.21259273414844312,-0.12942657882135508,0.21169252364725755,-0.05455136948139313,-0.216020624957825,0.34764198927622236,0.22552929227323826,0.15727293494062788,0.05135407820487159,-0.3313282802564078,0.17285542943097204,0.026102298835032695,-0.1915027758632289,-0.28148952434786567,0.2584054124994483,-0.18780742126546543,-0.033357044572686165,-0.2726254310456,-0.021284279536287656,0.4879363192251012,-0.16145353863299985,-0.20313442601420978,0.272397559267978,-0.04475241023642859,-0.13423451334499728,-0.13096113091566638,0.2581723538933578,-0.13881792402215443,-0.10566544127821292,-0.048034918631004266,0.19954633402765112,-0.2557002764879828,-0.22418486978429972,0.2709505151093541,-0.20782057642422272,-0.012899702137265488,0.09214604483484011,0.12463255573273133,0.008952965244653851,0.04173945412056341,0.1288126716043222,-0.018066366454852132,0.3603089406432112,0.07070263956371549,-0.10485341073294867,-0.3190033825067728,-0.02861023541356884,-0.21897033727613086,0.05578863694444633,0.065818631073951,-0.2580248065366945,0.06980755764460525,0.22177608366789495,0.24018933665358375,0.13841725901438115,-0.22014305452394184,0.011961282864278654,-0.14474220061619228,0.08267516650597405,0.2863864322901939,-0.23065005957557272,-0.21216572338181564,-0.13912673572691042,-0.10051918727186902,0.08831312162471616,0.045347190416987385,0.012801768959033632,-0.023597123692702254,0.308550380909803,-0.038640027858440874,0.02466728103298412,0.25310703464062617,0.053975074474464826,-0.054672157058292495,-0.035174976470308206,0.192130083336861,-0.002574833772918592,-0.018867123752919754,0.06412573792311675,-0.03797969620866556,0.186887928847278,0.2039810979450805,-0.006978107215326988,0.3070931110134607,-0.08856918978676828,-0.13442982432046321,-0.3201518336606206,-0.0160386007330343,0.4192157113301046,-0.05336959731102383,0.004050339876539892,0.03443861561923691,-0.0018419720802758254,-0.35947133955333893,0.03557792625869099,-0.17666181117425833,0.09733354856804108,0.07036805775935935,0.1513215118588125,-0.29725051868724134,-0.1273260143455102,-0.11102616000452543,0.0643694123763453,-0.19529570042742356,0.0578844604226294,-0.04362596192770104,0.24470384985374505,-0.10252446946589328,-0.10637772048584322,0.058276398585340033,-0.21815203636292135,-0.22134159828263136,-0.1852021715721267,0.11227767199145974,-0.021646666329847283,-0.10037898029027821,0.1328073538897928,-0.07835032472566777,-0.15311400388440577,-0.031728246264509374,-0.07539314666708172,-0.04102062967695756,0.10932530354554386,0.0361417724543725,0.0332118656687249,-0.23660115755370723,-0.18386635832428136,-0.3463355698067524,0.36659873819080463,0.23871029845165145,0.48511435928864005,-0.16559392792171967,-0.2747112047367612,-0.02063165918688473,0.07903692079955188,0.3893246868207842,0.09490724249519862,-0.09767530656561765,0.3129124017404179,0.1300421371356402,-0.04566124205107112,-0.057819311107837834,-0.36815619808347944,-0.20030080389552754,-0.1329766276499329,0.03871337798215891,0.24451987265753325,-0.37146102520674446,-0.3431215387371681,-0.2567486811971664,0.10441436491193357,-0.05686420013496776,-0.10906515882168868,0.09778549081246699,0.25401292506673945,-0.10447690220987042,-0.033002595807458276,-0.1016878590609718,-0.07643537875437978,0.0995582504340409,-0.3079769680414733,0.1585809607602276,0.016683372945497457,-0.1304896456333316,0.23640021407780573,0.05072971598036893,0.15329331923321732,0.18525867825453352,-0.21610443281054859,-0.20244060881812462,0.013841855395661603,-0.11589210743825588,0.19886574281596983,-0.21739120418005914,-0.25747586000892647,-0.08607175752396527,-0.31799908085382594,0.3173559384500115,-0.3706791164542083,0.11165682545894798,-0.24268074756395464,0.34437500899048207,0.03995045344264925,0.17406646217749888,0.06672623036326028,0.416577704944415,0.08597202171945957,0.07936055173106994,0.12606763659479678,0.2948759561223661,0.35162094863303905,0.0819331037978724,0.14967824621036396,-0.07157979816046452,-0.21260184266878124,0.27607712687158775,-0.0722470110564261,0.015362405821949032,0.08164915094404968,-0.09845932010913423,-0.23218713635040222,-0.13331196905753104,-0.03973549992178864,0.017403658780397933,0.06745340102684576,-0.027805325298738026,-0.10542730998932034,-0.08196100325037305,-0.0813494199866877,-0.040107386000795255,-0.12839755550293627,0.040668783762274625,-0.15541915234704726,0.16346193299013237,-0.21813744125364057,-0.03354474896841896,-0.20804342682777774,-0.2733431791921103,0.1101154381461423,0.06587375447972617,-0.050256735997804544,-0.11038795666049571,-0.09127268362218001,-0.27229562912525435,-0.23692493266141784,-0.09203405554057383,0.04648895940734972,-0.2607898322622357,-0.004318062894913294,-0.10274309930517658,0.2424898878710748,-0.043974666118757985,0.013011750834945017,0.12603381233709812,-0.36414833223232473,0.0012365287132523922,-0.18514648797989994,-0.12777291713130173,0.00899387171219289,-0.23862719360923038,-0.05169243235388914,0.12671026220177867,-0.058707599229057585,-0.27593454038439136,0.08114569693193423,-0.07833950402585746,-0.0566380127444609,-0.06612688629746001,0.3231435208036219,0.0530355910530824,-0.21291385016105013,0.4887764433238813,0.06894030374352415,-0.14280156902568253,-0.08746311365844339,-0.027913577345991248,-0.005284868406910459,0.3479005538291081,-0.08167438585967475,-0.14698873522586772,-0.16665070728878353,-0.05324310390116827,-0.16806306314165378,0.05475231268469827,0.006820780133702678,0.09887827130999949,0.004805989314558626,-0.16820934195994564,0.004407292253741208,-0.008975742319449026,-0.2448761598810024,-0.20102834924848445,-0.13021605578125692,-0.3368464531300401,0.46545770000252046,0.07833818287529014,0.09288196547457593,0.04945961750116767,-0.06795003209745416,0.07166302872192463,0.19369885173216414,-0.20762593596552276,0.4662536457575872,-0.17262795692371738,-0.16205343068427175,0.241890950303864,-0.02505916909605887,0.10918549409208256,-0.04682884716395836,0.3187404664800954,-0.032583048105053004,-0.06719742653433661,-0.2943685998366867,0.0880195596930171,-0.10289861306028779,-0.04354242026195552,0.14260009284496683,-0.10090611240390275,-0.055773053072007854,-0.11124232985850903,0.185551144137394,-0.016903708721678692,-0.16601297667358555,-0.11979499708479673,-0.011735490612879748,-0.1943157126356999,-0.13330747493795697,-0.07600291737774513,-0.00014482507084219855,-0.30049322979688653,-0.0798202554271509,0.08933001322159854,-0.07038721456216597,-0.19439009599476606,0.17307145857580306,-0.1112680503846235,0.3296546564492608,-0.14702969004711616,-0.04384504922274204,0.01681873290329117,-0.023980300634072298,-0.21421541239022437,-0.024943174877538738,-0.15458261178163812,0.13265227694295484,0.19650755808037465,-0.34783139322046663,-0.1399823735139787,0.302107897922285,-0.05533914972397531,-0.18053429433388465,-0.10449873383485984,-0.022495734159645668,-0.13094403418082787,-0.09018796433852053,0.05070084907653532,0.04028325464857303,-0.1315928980576284,0.27515595176243135,-0.10126986973982882,-0.021684624036984766,0.09310755208299161,0.09626871633187822,-0.04649886972571645,0.04911644527585152,0.07874579786554617,-0.09793928351147481,0.13995485732081767,0.05068904406864482,-0.08021732890312527,0.16675349652224708,-0.06093255460473497,0.055430142876394306,-0.06337258077312688,0.07399879335619314,-0.11093751330917528,0.019177723428166486,-0.0626627779623149,0.19807407296653692,0.3594135196047101,0.2591419653942036,0.05929472947179919,-0.10456952579915209,-0.19683155397672947,-0.038387253064667476,0.011391783806112669,0.048826860579148805,0.1099188888370159,-0.03970662772780545,-0.15527899368298892,0.03864074701844043,0.08070449999973175,0.04977123627014442,-0.12049839109062435,-0.33493288042176644,0.0449553610001429,0.2827372656117431,-0.16966285813874862,-0.07505625007689315,0.028755357682011994,0.030989843354002383,-0.13439088222576112,-0.3082858968185468,0.04365323590945022,-0.19222710070085886,-0.1472505653980716,-0.16202326972574913,0.18196266451030216,0.0749158563738074,-0.25985679801320055,-0.06473611110351553,-0.07496845209290393,0.037480736643712445,0.18848389515286126,-0.07662540378207704,-0.23121097244606062,-0.2384471459139915,0.19382798411289712,-0.2284565466242627,-0.07509610881029817,0.24935223575420543,-0.29256649264058215,0.0083597371066027,0.17919809859501698,-0.18172231218853513,-0.2003971803947002,0.24842043342634376,0.00014213945243195502,0.13629624776220356,-0.23675596928405768,0.2750190890382494,0.37717165034538847,-0.0783655436358611,0.1363499046695389,0.08893737965702211,-0.3697072281373615,-0.1927424775317507,-0.11763669778323456,0.23054259979219832,0.10784490892355961,0.1525539919093095,0.4813648531254862,0.22115503361619246,-0.17429120586388455,0.011513422387726227,-0.05484123679712494,-0.00471259696751865,0.09881792223017684,0.14458121419334058,-0.12685237104194175,-0.02927944640717245,-0.2066017694528331,0.007604296794171726,-0.1326026519675765,0.06336067076584993,0.24055528399702733,0.2068986619690287,-0.16037993821314991,0.2135414881592729,0.23116890349421065,0.13183945975846062,-0.23598101090989032,-0.3501687343628411,-0.10781704072155232,0.01312312903719958,-0.014440609832351018,-0.07996062709794162,-0.1507491957164105,0.28869688116768627,-0.17081225758319518,0.04458045725498022,0.12181424060268059,0.03224481426068444,0.058379267691607645,0.05989198368746721,0.23179176823579747,-0.09804120489330363,0.25046622250927536,0.13181433359666123,-0.1883697789321742,-0.1331422683699537,-0.07495628067050751,0.3881933225796622,-0.09736663335418864,-0.1384479484044176,-0.0846494025144217,-0.06492652940535482,0.3390667673993748,0.039635717205758474,-0.03294212175876271,-0.0073841034113559575,0.11676272442000253,-0.17183449920995683,0.34349667829777175,-0.07994583938450131,-0.056998122116036394,-0.20217475662637685,0.004097743720817349,0.07140152359729766,-0.33454046184439223,0.13347575674834702,0.03928613503217916,0.054637958393195823,0.14801372173317714,-0.18850267388577702,0.3348723790201837,0.24286090503772267,-0.2449094758338592,0.0044070106399090704,0.21069637803121324,-0.09855902777764969,-0.06364720029179184,-0.17902817395633036,-0.3449972359347937,0.12518429401028572,0.009426812518255733,-0.25732287287855116,0.04278865849575106,0.0829223108490888,0.489116886190595,0.13150552121298067,0.3346034831917307,0.2680739818719054,0.04081042660314058,0.2691970183171251,-0.1728606378028599,0.023587703647068844,0.22167747839825855,0.08204728729537739,0.20325877132848547,0.12874282666284814,-0.0955499726256582,0.35942088203089895,-0.20812211587620366,0.006633244998909405,-0.05423255936714146,-0.18188732127163337,0.0832358553389633,0.2666443003066527,-0.02314562566219269,0.041804940276082135,-0.1091900033772791,0.2859669503771107,0.14407283082004832,-0.1635598090443826,0.19455161441426405,-0.21070361679256042,0.40843396785895675,0.10906855724099065,-0.19270367119146242,-0.03612178068790524,-0.05650994579300092,0.1565632412663609,0.04985054911442951,0.09819103997905629,-0.09856700067055642,-0.21443604349911657,-0.062131182848093956,-0.06291312570008711,0.212625970390809,-0.09681638392062092,-0.3512811232053037,0.12823780329575926,-0.10788776232591171,-0.09005148768323219,0.03652308050701796,0.3086994008717932,0.2355005489224484,-0.19150690982364038,0.16883037449842414,0.0009953364714769299,-0.0886295368853941,0.14098242145628792,-0.032977004631024764,-0.10006780836558883,-0.08670223503241306,0.05847417697401235,0.025211454510898203,-0.09453359119913692,0.4863982772029714,0.12385934172085339,0.48726455657782386,-0.021007131842503732,0.2293763082712161,0.35112297106262136,0.07607461001759301,-0.06666412024519204,-0.08890170885632101,-0.10161029593777986,0.0044167290485992465,-0.16593495582718476,0.1941916762718861,0.03638061073145351,-0.29523415878182424,-0.3694758053615497,-0.1250963676912733,0.13873230569065376,-0.3220114030221386,-0.13112372973497682,0.1506239031345395,0.11994820361151819,-0.12764736835614715,-0.17122388134522826,0.03546632051548844,0.2293067709100349,-0.1324105194417493,0.28440804128918246,-0.08411360048339188,0.11460490008175997,-0.07189291537955295,0.04998482929539328,0.0032132148702493893,-0.20080626006779292,-0.06923170441761775,-0.07525448257422201,0.03660505172236939,-0.0956772181544236,0.33266329791213006,0.23988680002289403,-0.08704403460695115,0.15707646031774833,0.23208660400764558,0.12734895177926941,-0.21273251419079858,-0.05344661086025385,-0.2607798613988339,0.1673569156990363,0.3911341825214572,-0.07054487383652039,-0.31974527655518564,0.2858560506102826,0.0529457517205775,0.041791805664949354,-0.226936985417579,-0.1951358662855712,-0.0805437988183572,0.12123877719979258,-0.0739394680927321,0.01535220685483278,0.22523868369743544,-0.0434626264572966,-0.32528440946154613,0.16430293215898079,0.1336476402455331,-0.01801297946998328,-0.26156282933093317,-0.04217421425179355,-0.14001962724570105,0.06194722666187574,-0.13152683462673037,0.28225968928349665,0.3843701719085452,-0.14351105347049378,-0.2822366249895272,0.18304042510014876,0.14644777711809367,-0.2596657518199165,-0.009814159726477667,-0.23474465950721016,-0.23880669240951663,-0.16784380751308828,0.16230404048278424,-0.07966470101319242,-0.14590288028373513,0.37527236535374536,0.07660494925316563,0.15052432804001578,0.08514171792386212,-0.15154901328622356,0.2807988827699857,-0.25655553965622974,0.1601049186168539,-0.1939702723616542,-0.13526550731545706,-0.18432819502730197,0.18407040133176128,0.0705762964294323,-0.1342424334211567,0.03373340577793162,0.2370612740341653,0.039116341704770866,0.47620827139888505,-0.07075813854406854,-0.1874087491012774,-0.019608751437163518,-0.31890069642139196,0.05895701729524894,-0.058888171171806454,-0.1451296602035888,-0.059378798423549435,-0.2564663445210496,-0.23633705132021798,0.20190059324457768,-0.10579711167691105,-0.2017034850771027,0.35205000920915885,-0.14566209392608473,-0.20993070333960584,-0.0694055638808584,0.2866244403368992,-0.10780827598087604,-0.12491433061758446,-0.03672448142460701,0.03587316236613321,-0.010186366845283913,-0.10993778112677008,0.017035505675134687,0.33733814281749985,0.12313758056116338,-0.143023491263208,0.14478700413343729,0.22811816026344403,0.2372154518332052,0.15420020770789175,0.19043429110220278,-0.34802107475871386,0.2223144140415964,0.009311133881119439,-0.262177528692577,0.005810347273510375,0.023545278023105378,0.19126594457965296,-0.2565796478219689,-0.04489702623059128,-0.3274693842951693,-0.0704066964576024,-0.11432112712368935,0.18425973580058305,-0.01915365458773684,-0.16402997058025276,-0.21268887294810393,0.04320380967448516,0.10999543200489385,0.10021104424783744,0.05821547578276887,-0.25913711707367443,-0.136334411945968,-0.04843096681625715,-0.18420147115176222,-0.19268591896784767,-0.20654973180736144,-0.29091563666045783,-0.03964893586305897,-0.09400388829068308,-0.1829584188478728,0.06704024390653897,0.18022276686339775,-0.2692629339276106,0.12111644764622977,0.03719294074129904,0.061110943241851894,0.20312007387711123,-0.07371943131149696,0.31185837907054664,0.07800455113314662,0.16874970450655202,-0.023532489402539016,0.16146607401822868,-0.017550710413379786,0.013369658379272136,-0.12562884778358066,-0.17912691247082932,-0.2340420447193435,-0.053588319683517814,-0.12973053766299766,-0.022988395264193724,0.17263855522429333,-0.05423778280989846,-0.18304042929058148,0.0900343121876877,-0.07452712430181202,-0.0685933050698435,-0.029069645174822113,0.005118979882757171,0.05819904140913012,0.1529940535914649,-0.24857464982761368,-0.0813229644771838,-0.23216670401312375,0.013338662000350325,0.07621202146612598,0.20293542643025153,-0.16563744128686386,0.20988047050706293,0.08250903657172541,0.01085840335916608,-0.15904709616810464,-0.038061851627008356,-0.30628304189034106,-0.07158843734043985,0.06162201815618367,-0.06611541732324709,-0.31053137068164705,0.04709986594320999,-0.3694610578626905,-0.0900242969901831,-0.13702211735394065,0.002129209593180605,0.09748882698631434,-0.15434500861173642,-0.18803441280535027,0.36384385555026755,0.3319081735397436,0.09928423695782351,0.011101838053269003,0.034904407524217376,0.055873596866611586,0.48872383052700347,-0.115781015067737,0.11542881675775186,-0.19232266990647418,-0.06407838212032929,0.2232894009054712,0.03262321560921555,0.0870343459823217,-0.005862553579721481,0.27977293691551997,0.15023994326246282,-0.035263934788139474,0.10462106462692966,-0.2325195222872209,0.2169895203826382,-0.31352351244264937,0.0015819510627565793,0.3352029831354518,-0.020903347087871733,0.05429367178887374,-0.3408538062676504,-0.35285584396224795,0.2910591771888534,0.3514792836743346,0.16207940551827665,0.032344887482636406,-0.0652082015259111,-0.12985372138850856,0.04777916537233026,-0.0572145822503604,0.24418636034742897,0.17932588912227784,0.07568364127464208,0.18588037777963354,-0.22091118681864735,-0.3042711418895972,0.019613034093887805,-0.20168531443653928,-0.1286261944758952,-0.21015078501989354,-0.10563754417679612,-0.11953698078370127,-0.16273978694299016,0.11541895113370045,-0.05708782088309984,-0.10455831955338211,-0.14433404708832046,-0.04528375402334706,0.47492220103880567,0.11989362288447876,-0.08539468017595621,0.01834969055140503,-0.12233069316293695,0.3315176907533284,0.1788113494215161,-0.17273585139182834,0.15699664652710577,0.10455600677139018,0.07011142048896875,0.2722257612813429,-0.1320315088204714,0.039117063365540015,0.13498056266365605,0.26359997885265835,-0.024110844442347963,0.2687062813907877,0.01640374514638338,0.2821120518009694,0.20453715756650825,0.11360784702015886,-0.1352586727287816,-0.37140790192296974,-0.00550775797428729,-0.3265314101825385,0.25615641460808763,0.10447088753691938],"y":[-0.02719132807179488,-0.2587041437714644,0.35487628349767936,0.04144207484854738,-0.023955474299664326,-0.04455685094601722,0.42189044027920836,-0.17420005114506731,-0.008294334023050055,-0.10599346300520537,-0.15536893823775633,-0.08560419691259387,-0.00945972253506599,-0.0748873933127731,-0.0602151542250706,-0.13139055204904376,-0.12496639546155973,-0.018880119623185396,0.0018794380830785573,0.17817233813497196,0.0071714582145121835,0.2326420247887645,-0.09462656172938563,-0.03974468055204426,-0.02898880397040616,0.014313586379933109,-0.1498005659678823,-0.022726783043904165,-0.1396676552663237,-0.0014943424924098083,-0.1236744661470785,-0.029411792914907527,-0.04813819849571755,0.06065607456729778,-0.05161869342329155,-0.061422709109182454,-0.10478163205584408,0.025857516495577357,-0.17686016618510825,-0.10798884867380316,-0.1595748417355717,-0.04381809833167334,-0.053538365902397515,-0.058458750091374634,-0.1353477029151089,0.005925918107009122,-0.10541267409537612,-0.13612949242646835,-0.06873673822941848,-0.05294160975482906,-0.16457192224129222,-0.12636892804352085,-0.17239005261462312,-0.03135573586319305,-0.09150593698869207,-0.1420624862600572,0.04306897800966923,-0.23508408637547779,0.07493503897956416,-0.1118229555282834,-0.2963232262358204,-0.09502278367218975,0.12434431605255433,-0.012983536777604013,-0.20997526475673375,-0.06828344771098288,-0.07124776480458449,-0.018786855696691303,-0.09328260992253462,-0.10563446557930058,0.023743469482947845,0.17077398046827666,0.05324238362915297,0.012683437081934696,-0.1711903699806209,-0.019512945299785662,0.09804558405904772,-0.0688548227498797,0.14007864293215644,-0.12327605909139108,-0.22582347360593946,0.009293249166299707,-0.10679924291516667,-0.14322313305139947,0.10864559619490229,-0.06544631374802092,0.21926739450539406,-0.023110120215727625,0.047921015400715646,-0.045930380043329576,-0.004790463779705384,0.017124637618172713,-0.13041830702950394,-0.030852858762800903,-0.10954249711054169,-0.04496245978893298,0.00415182878356876,-0.02904308614332084,-0.03221096396225857,-0.10270384003979784,-0.10006004098800847,-0.15152512440856847,-0.10734408612015811,-0.054891113576089036,-0.2848475605475558,0.0693382159311103,-0.08756612606923342,0.029311125061367228,-0.00036557040796246466,0.02149326973291425,-0.07585215470232758,0.1980794344708201,-0.015579897867744639,-0.13902324351040862,-0.006649857629751832,-0.10260758022059127,-0.25313694687672605,0.18863784370208772,-0.12120853096522226,-0.04210597626956333,-0.08135758400540119,0.013223400430554685,-0.0135974672060091,-0.09210928923494789,-0.10633391354682128,0.0607925357830294,0.208635252255682,-0.14413819612952625,-0.015623652048258638,-0.19880132832125696,0.03412960073534058,0.10752853258886903,-0.14883548486498877,-0.09887232259203997,-0.23823668353675206,-0.04224199786717473,-0.0982609876760709,-0.18536175435259536,-0.13260437369840652,0.2753571745283106,-0.13226138096141227,0.05888991371183852,-0.0710366952326966,0.06974942357537346,-0.06118077739224898,0.03224215323035958,-0.0037413633186202284,-0.047340768320565586,-0.09599333348212091,-0.00796706323582324,-0.017271888132793126,0.0402277685103232,0.04958459908771195,-0.034987690185988775,-0.17322032130950374,-0.04075354970372502,-0.12029310361846668,-0.07605199448961227,-0.24183574044759132,-0.16186003610691724,-0.005386133689402891,-0.024991379577317747,-0.0059032219525228915,0.06825753703478013,-0.04755825150698744,-0.16080632328525776,0.0421883226007688,-0.10997963244019687,-0.04688057044953138,0.16181657701003366,0.1445928190533759,0.04963108532983607,-0.04959897029941548,0.2872297193585829,0.19793133320993572,-0.09172780071894521,0.16227003862502723,0.08145792595423688,0.060146265609355314,0.131999144525593,-0.02006976657086177,-0.041660537339854156,0.10025523841157692,0.07500838547457714,-0.07511351777610366,-0.1384581222708556,-0.15464332628091562,-0.009350690191486719,0.19543062630180696,-0.058865264433380915,-0.17109326011754325,0.09466559755467406,0.11124796081965121,-0.09335769512797078,-0.037744100497160434,-0.1793075110607246,0.3156742787557678,0.22132947981421766,-0.1609355379349961,-0.0779870808256496,-0.07362829483549518,-0.19207741003801768,-0.23724551046965517,-0.0961535455399975,-0.15632630336846393,-0.06819981009956616,-0.16883817963233996,0.06498615919253946,-0.08773449672541019,0.10360433012024374,-0.07209463943544674,-0.1177714361531995,-0.0635945982917874,0.026440782536045333,0.1472100065403271,0.015959660207431604,-0.17523604893370956,-0.025370180660675506,-0.017179924756626417,0.03164895254364569,-0.2193440560334288,0.143893392471876,-0.09576805452779628,0.045438510519181276,0.2708992222708825,-0.04022737031829707,0.06998422738327442,-0.148655187373898,-0.17666445832571703,0.028611672249469495,0.06939476713963025,-0.06083293610634719,-0.18914946501065488,-0.02939002404180583,-0.2504455907557348,-0.022648202340258827,0.049619954503431575,-0.037524322359865754,-0.14431318949622457,-0.053747766176837575,-0.06350157607250219,0.0361594037040402,-0.08352435670589074,-0.03428561671797676,-0.08825943621361299,-0.10890695212976169,-0.08987274113827777,-0.0223437754108535,-0.0038721476084836474,-0.029424409114000936,-0.18782951273727821,0.015596454627798831,-0.0503938551305798,0.10351551336845201,-0.06402868569649639,-0.1864501294931927,-0.18922984870388096,-0.11003102071740715,-0.17600326330902868,-0.14963103171151357,-0.012366925206083143,-0.02136975929298933,0.01545671481277384,0.031177445679509738,-0.0929486145193794,-0.060802069348559876,-0.10426625671979536,-0.019791093287558556,-0.12258511255281843,-0.04633478064292724,0.016469192487860024,-0.12435457242491599,-0.06688441948074852,-0.12390963881424652,-0.06917327331871953,-0.10442743640187177,0.02982873418825315,-0.2642758435041802,-0.05456537400961369,-0.22621618979454555,0.3635399894713302,0.02092886691836948,0.027522881647974735,0.05295456023621188,-0.09052644114443069,0.0682824178153454,-0.09938977716008479,0.023610764502006033,0.029689903824833695,0.20974728540262888,-0.11475616029872776,0.1627100320822284,0.04109536491071271,-0.18448664242172322,0.07495227399781053,-0.07818144524790538,-0.04520586613113046,-0.20136645369610623,-0.0836850583151128,-0.03862431164330832,-0.14017051908462969,-0.18179099372991103,-0.12498895528403996,0.03890996729920985,-0.2449895996501259,0.05002899976317819,0.0808084972335651,-0.16200655203287068,0.010541669133356893,0.1655454532388683,0.09199865120761273,-0.013405945106888641,-0.06571704997483016,0.10343544356575296,0.05499113398857456,0.0826018559667016,0.06019332336754492,0.07153563864078374,-0.018542627571605436,-0.03801070204657052,0.10843423457404343,-0.018987420274345312,-0.05793695717617735,-0.21286984452835592,0.0026087428857377646,-0.0028643600806718063,-0.11585039288209506,0.26479347852595586,0.060442495352196625,-0.15383465781577466,0.08574108436030553,-0.2896364343836151,-0.06053473652097318,0.012725220346915428,0.0035777516122604412,-0.09313688446376336,0.0007705702121166208,0.03554425230997955,0.01716670172662244,-0.04545322077996613,-0.11588004323314115,-0.08574608637245244,0.04552879813744694,-0.17145316428347077,-0.025444026770333156,-0.18614901926515037,-0.07355167903036541,0.037471767099330644,-0.00980955815152548,0.008683346668418537,-0.17277729478745898,0.02628453660211849,0.02830390105445653,-0.07811372942627393,-0.011760078986817319,-0.004388898523644997,-0.07288375035468836,-0.030156432454990546,0.041919909647354814,0.004279355898599765,0.09245689992721444,-0.09322960402952797,-0.12469938257203013,-0.07422806942721974,0.060635253441821475,-0.13947739683246266,0.13350596453441022,-0.1385332762506671,-0.09307251358658054,-0.07937490256532889,-0.048335771567517206,-0.12592691156569588,-0.16585069658619062,-0.14968868271393568,-0.05772525573341085,0.39335223986215057,-0.1372991089341481,-0.0021151110381332694,-0.1931155461518532,0.016435443505182643,-0.1443168750093658,0.08158500724485494,-0.07696301019169861,0.18741068375515338,-0.029355761841244863,-0.06680835602590131,0.02092954300277136,0.008115901105411875,0.24420330843793223,0.06440561597802706,0.08084846074896444,0.12873698426813407,-0.23322370518734853,0.0643220720768265,-0.14239794010641754,-0.2520988733653546,0.14811915236328238,-0.06278135495585005,0.025546131605414045,0.22104599594518407,0.06508076167990258,-0.004541870603000774,0.07628390181234772,0.06815850940986594,0.02254533946584079,0.02007884104370442,0.258585145349407,-0.05066286519967206,0.20373570820096362,-0.04777742036786638,-0.17614679511815837,-0.09874890988411443,-0.06953360938098008,-0.08368702189467318,0.04588730185178183,0.03419705313189555,0.15295195837459208,0.08140278616300378,0.08469390781335803,-0.03159841905497604,-0.14655117738207019,-0.12556425568680776,0.12685774478909595,0.38351768761662824,0.03803279121260238,0.5381054453043773,-0.14338211884023103,0.013332797950734672,-0.06912366619747269,0.5284983070419049,-0.13869534036077716,0.07217343296117282,0.0672393609077784,0.052094270628233734,0.012581454663279147,0.060824912806025325,-0.11489475780234694,-0.10147416314945455,-0.14349112946331433,0.28779384585439155,-0.12284472584114688,0.006053774380711749,-0.05544037142095209,-0.09106574120237503,-0.13978124220548607,-0.06777558285587003,-0.14164423803274853,-0.07888934975077888,-0.0647842671052114,0.23412804649044622,0.3085968903071819,-0.03595382621215796,0.14390158742797043,0.0628894357179859,-0.037098009285544735,-0.14462073938668324,0.037815013946973945,-0.12740604567982006,0.22021582056646102,-0.17393998943958636,-0.10630656947000364,0.13362826798410757,-0.06221602187025326,0.017570462012631637,0.01281050066169895,0.012823329563254536,-0.028900907934400123,0.1932949346045444,-0.008758245781124791,-0.019838882727906123,-0.013492451408040885,-0.03250299257143148,-0.04541323737542644,-0.01803389232272295,0.17146116710884976,-0.1823595358008429,0.01326865643864461,-0.02998468618944894,0.44917529094052305,0.22054002634749703,0.09689105479644083,0.14990361777686884,-0.11437347142225865,-0.05272163679936475,-0.08231181794786524,0.1412804321274763,0.17781474162493127,-0.002104895081960891,0.051527227379458015,-0.11275507514169227,0.13815476946603272,-0.0013690558055972294,0.07484674506798591,0.12469734478233276,-0.26505169366609316,-0.09821246359408191,-0.11386624503303973,0.08564097368688718,-0.13255006414150772,0.010363238374455554,-0.002573846501655205,-0.09915713290422931,0.06209150931202252,0.0044949077265062165,0.5943171625404476,-0.030732741185558955,-0.025829229228185733,-0.0462145343817875,-0.23935724156333488,0.10228469497224302,-0.08109354232028704,0.015105094134435535,-0.037992522129108336,-0.2871164471966835,-0.16968699078592545,-0.02882286705927812,-0.04224701356427545,-0.09640583480143276,-0.24213318932428446,0.16428278303239624,0.15489883795206333,0.04800876589796613,-0.039503695148325545,-0.05579359778154366,0.07556695553388926,-0.16551453641922415,-0.04138671792886298,0.1186184505608717,-0.07216599829375706,-0.1875205380510516,-0.14016938850448368,-0.18323688162103713,-0.15007199557747705,-0.12685225562255495,-0.009240211259966713,0.5106220839663227,-0.1196415541642663,0.208691080287509,0.03200823369621507,-0.15840004746187308,0.04414768962105854,-0.048146157458295924,-0.12892860738608297,0.23424771342433554,-0.06556929450958057,0.005831123543662036,0.11492118768757217,0.3872829491434955,-0.046802228288842215,0.08111985126211281,-0.09791446306511317,-0.08661104383953891,0.11269051374026508,0.17630095523377926,-0.017885957863806982,-0.04985910989474498,0.20048683243950444,0.08245258108377387,-0.07077478376009241,0.032540749506595926,0.5932080041514506,-0.004195474661388309,0.5891727970930691,0.0704144538272251,0.16305920891629208,-0.04852427133652217,0.032231662883756146,0.024148712495205397,-0.21197931813150014,0.003877652075408518,-0.1136744966343237,-0.11626980639613373,-0.01631839306599663,-0.02123078498105701,-0.1026753004034218,-0.27439349081971665,0.039149666680866134,0.062188474594255055,-0.27865143830855854,-0.24396886104188487,0.008717978195417746,0.16128575032119616,-0.14513610685109826,0.06401541715948018,-0.059300174537214485,-0.09703798739423211,-0.14192972613421873,-0.16653193467926336,0.01562501161687637,-0.1588743340924279,-0.06867474878627858,0.16033628248076573,0.07184144226991117,-0.15536816910470297,-0.06711058679128278,-0.14562591572750766,-0.08331138026053657,0.4234093586987046,-0.09103654709970287,0.056891529137007436,-0.09993298796666397,0.008582748551562505,0.13781610118016618,-0.2454022294820454,0.05687043735172765,0.024234547283569934,-0.15136621117072716,-0.007942849277002152,-0.1403996635637938,0.006657243690793146,0.013344632852962756,-0.10363741573130103,0.18031883922914468,0.08919896710448699,0.09039874796277736,0.15525527425370492,0.16683062380123223,-0.09093759098286386,-0.026506414018985697,0.06327540514712354,-0.03539723543785942,-0.09116455541081411,0.06927901967285821,0.08825526109583415,-0.05029935002360943,-0.16673801448899114,-0.0687014570638503,-0.07293964676178802,-0.2613010324094393,0.35659476112963134,-0.1386061875350468,-0.10318115193418491,-0.1606308792470693,-0.013000690561784364,0.15663273129338834,-0.20905307528256517,-0.07337180758677805,-0.1650551368394951,-0.005162489411852208,0.04555542736762386,-0.03258098632563599,0.06861153176525758,0.05398403263467056,-0.11308903609722375,0.4258788680590749,0.045233968420159525,0.04414632952582266,0.1869260340971608,0.03317461371547801,-0.16751466942391768,-0.2002300808946461,-0.002618669834010428,-0.1967962343759886,0.019033236862273085,-0.024246014668701892,0.02490132989536589,-0.04253781564200174,3.7540281555469385e-05,-0.09581502293882924,0.1285702324372462,-0.06074155703668159,-0.0028596956923886818,-0.2164159414223611,0.3219156675351879,-0.0628991476343924,-0.0688164910137608,0.32992871779728156,0.11756244338482773,-0.2883091823310177,0.028608587872577085,-0.02909603141281337,0.29618442364902,0.016316384218454938,0.43034740961113743,0.09565366105366205,0.5017023938949626,0.5934869800562108,-0.11255947148588565,-0.01567710246078764,0.0723154623010394,-0.07621388151510528,0.08898495950571125,-0.13962946359794812,0.519028450885035,-0.12802563441891876,0.02991673226268273,0.5911278781996685,0.020641554376448063,0.0868271018612594,-0.09275890683531403,-0.00284379788472729,-0.08640007376322457,0.06352540754354682,-0.2651256375717208,-0.1544444023941242,-0.02584233982410482,0.10217713224184939,0.1067500182305857,0.02910842066361181,0.05479092280746183,0.004593259268401002,0.09055514665336309,-0.20887079302758238,-0.0736892467138526,-0.015198329572704864,-0.11619310020854348,0.1127033313851967,-0.16252450291167975,0.06944332108521398,-0.06926336058101017,-0.07906187019792232,0.0083461860460599,-0.30576673045060887,-0.03165458831909609,0.14028559107674357,-0.074588641787328,0.02466569022243333,0.3065010435951218,-0.03618724597841923,-0.0017356284222399116,-0.19619574996288647,-0.2184421861847706,0.0691240710656413,0.2533327924567131,-0.15559453989430438,-0.13481333861612896,-0.16366916632440412,0.24532547056966847,-0.06470882013724778,-0.07323867830323072,-0.1672127242416836,0.08670304954868346,-0.03845690714715442,-0.12132834181666806,-0.08224935966507921,-0.11408365872465154,0.007602433912959128,-0.01378701194573683,0.3131023001319858,-0.029989067199803026,0.036731828552243176,0.04636693176001471,0.02048805952415245,0.003187348264354697,-0.00016552777636050903,-0.1614755972461308,-0.08354565428231547,-0.16203717672777834,-0.09180284464807113,0.1280071503522816,0.3321563704711095,0.2615839169836745,-0.18185437531977997,-0.01848033496745721,0.05449084449814486,0.17709790869066838,-0.005470305416828642,-0.02805367548992336,-0.09619791682647502,0.06124036733278031,-0.09238092539398138,-0.020287384990602873,-0.01355588200838659,0.11515531388296428,-0.30429986280058385,-0.13274032806848246,-0.1734624623981393,0.16176529929005318,0.07134165501934185,-0.23027606569158943,-0.12644143692064053,0.0630323730008537,-0.0818585837374335,0.014471438261686282,-0.03684837645079762,0.011085164807199899,-0.0011862073989442152,0.012071305507926816,-0.051940620627719404,-0.14771850997376676,-0.03302707348583725,0.028617419898121405,-0.03112240863502311,0.18564308633525103,0.01415072089770763,0.04573028258073322,-0.06561639500351338,-0.041360367673230064,-0.07308526998497404,-0.12725532315113125,-0.06614786590762954,-0.08057734738045001,0.09803668077898423,-0.08132301323605175,-0.06980796731209041,-0.16679979035539838,0.10907435318598163,-0.1870031863508173,0.18258109788893104,-0.06436841381150143,0.027299636271774192,0.14266433059970357,-0.2804312704573844,0.17289235685700624,0.08303023731364959,0.12822137854526097,-0.21199150255137847,0.06175756265169252,-0.10370403812757349,-0.11974137570944009,-0.07384176190521112,-0.0848518431300065,0.2697209590989981,-0.08196146538064687,0.1310816845938133,-0.024290024081668367,0.01562677614558691,0.05289527181583883,-0.07141625352687059,0.06075143798549194,0.13993357313783802,-0.2361538920368773,0.14993837577875585,0.5731466651179289,-0.07961082474151551,-0.02178344767990091,0.1121308704047945,-0.029676468101885727,0.2449812933539149,0.198558606522043,0.07341513030245803,0.02852122068826099,-0.2622431199721338,-0.031897206108154705,-0.16252450291167975,-0.015728252630331663,0.03571406842762893,-0.07743632233836782,0.10716908980997354,-0.14959330024720832,0.1959906637663373,0.31143477478204573,-0.07114258097724535,0.14638260737197856,-0.04651048301095544,0.09985472290696257,-0.30635191962277625,-0.0421427743189656,-0.03660828714377918,-0.0765672983277943,-0.09914059771269043,-0.0871547932434086,-0.12664525828178907,-0.20525859678756805,0.07768533740944068,-0.1864109344063248,-0.18629329865846864,0.060773841865302276,0.10157179265450607,-0.060094673402062225,0.11440208418958737,-0.06668375019495627,-0.20772983377646156,-0.10190785671619657,0.1506287775586022,-0.034611434699178226,-0.14683196352121944,0.05715519251192934,0.16949436063132448,-0.08353215133477261,-0.13546644022486026,-0.19196451322509642,-0.04615646551897767,-0.14016282986762413,-0.0401020783352562,0.04351115226112256,-0.23583145586540544,0.01999488426633252,-0.11491476571958546,-0.18132200054078698,0.28801097959483,-0.0009881594620694339,-0.1786489128899767,-0.03527749878437395,-0.00681129930825496,0.030601823665348263,-0.03284403924123835,-0.053591692034922145,-0.024092441849897492,0.23173501541559857,-0.30199421434722246,0.01209263564598513,-0.13714718875912063,0.3762522619423914,-0.007823675638031734,-0.17853391947106773,0.3413251382127243,0.024402846397653787,0.06412058229740664,-0.040317010593810626,0.0010265680661519351,0.09345983905351811,0.05702961467774954,0.09458489247951903,0.049051010086224936,0.09613359034984693,0.12504553528835913,-0.13132181282625346,-0.00843621857722707,0.3049329007441302,0.3852776166275171,0.27189176067268583,0.04040119072065185,0.08136262379981266,-0.19528294269322902,-0.011883781774976197,0.08564438472530309,0.0015530565777563208,-0.15437153652852156,0.17074846721757989,0.08760254023728038,-0.19188132008893158,-0.11233879350769219,-0.00017474392416710218,0.27668488380377565,-0.039037992848455265,0.21211810977164505,-0.09146158349473199,0.004604687246161757,0.30948775377466187,-0.11136351304442733,-0.014572043495808049,0.07109901669436557,0.009290158155370267,-0.08481900121947263,0.11131287624613034,-0.0841988338900019,0.04492481459005028,-0.011219899429627523,0.1693543939506283,-0.09262658260277701,0.048515418462409254,0.11187983481340544,-0.12372343852784227,-0.16389775969009296,0.07125171350585084,-0.047647224383706145,0.028059289382112367,-0.01675151046101645,0.024606730567281618,0.08134084704772462,-0.19548970815739744,-0.29665131464404715,-0.028859452123735314,-0.1831694409597453,-0.18809823843272055,0.5691111194360104,-0.15018096931069053,0.04987017967943587,0.03343174756369286,0.05049801284215417,-0.019840884131874466,-0.010174329845663218,0.10599525810054224,0.10808500869483871,-0.0010985492953142465,0.006150796778374106,0.3725743071582471,-0.014872977858591629,-0.1847787679805505,0.0585362778287748,0.18116416463809143,0.05067386929114773,-0.18186047736919092,0.04619201570838421,-0.026718098941909425,-0.10613058229295169,0.06449249810156236,0.15619477459910197,0.0973382668022752,0.07700743018257936,0.00922587409093315,-0.08552781166357645,0.17027144672365813,0.03229257925201991,0.02442822771079092,-0.1430169356732641,-0.10823145992295467,-0.13987836493818512,-0.0809399310954828,0.14124601500448378,-0.14409349461894697,0.17667103970349157,0.1720758368175782,0.14045667864742828,0.04282798549736877,-0.1964863020060987,0.05931798251668319,0.5869352321758377,-0.0512270942949455,0.1811377471380726,0.19279272180858625,-0.13798815708397028,-0.24626460663639946,0.16950642004652122,-0.01672204196251659,0.050375938820593616,-0.03980430931726503,-0.21648526459432138,0.0391906367440494,-0.08537814741549243,-0.09027848933996631,-0.011265663689005763,0.07744809348611224,0.11180049924462415,-0.08217199542718023,0.02902001005497577,-0.2316624533233435,-0.1541070276212816,-0.03769260976753441,-0.08887397131863284,-0.20293386212276732,-0.24450252728214766,0.013637089173261976,0.12187803894110995,-0.08783935092043794,0.27604184421884653,-0.12713583576076262,0.025844330589109524,-0.17145624085722025,-0.08189559347558645,-0.016079897611350633,0.07307091507732022,-0.027871775843306006,0.03150531739129308,0.3240366681267293,0.06581326469066907,0.06196484262119135,0.04143449835585542,-0.2703609696727152,-0.2565543648506648,-0.23180562428789767,0.046419233368613026,0.33895500994982,-0.045527230733784646,0.3819508437755592,-0.0535727210274778,-0.054141727469044616,-0.17351403063891424,0.013657547020217075,0.20067117615385255,0.14178737821730078,-0.1226750866211116,0.18592145328768508,-0.013098543659981934,0.022030447718613316,0.055805753142277814,0.20789007699838577,0.12585210670522115,0.060669367658858865,0.060767662257907765,-0.18923417286167996,0.1349070935756495,0.03934309963382148,-0.029767528364379143,0.13416803309242611,0.07730567219380735,0.08504728087856592,0.06007837250842968,0.12637160192320185,-0.0991913190468015,-0.08181860032535004,-0.028160053100211347,0.23963393256716659,-0.05246805620302405,-0.0031019301632057767,0.07069861302937792,0.13723172130135672,-0.05607644955498296,-0.008731077600592515,0.1240945053825165,0.141288614602465,-0.15662356149189918,0.10803376453848235,-0.07683322179775137,0.023773033506692458,0.1166932350055206,0.283084320117548,-0.1714937660353101,-0.08980153262157681,0.009840902796684625,-0.08746152091376952,0.42327030559996087,-0.2782975136910724,-0.018353320072292227,-0.02633332475537576,-0.014724002513857614,-0.04653031270603253,-0.14838430250223927,-0.05129176283904999,0.15880153463922128,0.23450160436567175,0.11230402051413159,-0.140996118949515,-0.11989352894311152,0.06259088848470766,0.19120014764230917,0.007110461181263809,-0.11933769654576605,-0.16137104541007133,0.2871782087178475,0.18585906754518,0.055903439722544995,-0.15779516073233238,-0.140082977082218,-0.019338386991896092,0.09669079082054388,0.02217445796068689,0.5284586188254445,0.03611158791229995,0.1136358384076261,0.101227025693592,0.1680421409384875,0.05011612750978369,-0.11748381051661753,0.08861330267977874,-0.2934929732176732,-0.034115631824164654,-0.10981752517866401,0.1307878059199949,-0.022922093777581067,-0.033437146693471836,0.2901353917666247,0.05114650950623225,-0.18337895777765054,0.11076523495047456,-0.08660091664465186,0.18385241607221509,-0.06428228384126417,-0.028974731516640508,0.09733460247993984,-0.05855672573162396,-0.006705854710259673,-0.07873440866619635,-0.11876489734215102,0.05853475296127781,0.06211621443691104,-0.03560040010302282,-0.035624742443085146,-0.04802558503154416,-0.11727679297782792,0.11567045076211471,0.24344289883780992,-0.14126872553663902,0.038593733229727784,0.18114298337762613,-8.951690709461253e-05,-0.14597364832355836,0.2484514322681799,0.047358160315116,-0.009468686997608888,-0.0333887579451718,-0.165928993181779,0.13694184123465655,0.12075862360324997,0.02421709882672975,-0.01726584005464045,-0.12878959589038483,-0.09634540246589006,-0.239553127764802,-0.20285254520944,0.021553514162372923,-0.012780199861046428,0.2242686492557957,-0.09489089421640486,-0.11199177779976974,-0.1377850945288678,0.10211968203167993,-7.545975434165594e-06,0.04137754561209775,-0.28160009945094755,-0.0073105976796984115,0.06090889888407714,-0.03376980363774019,0.012943391413136295,0.09703135106288761,-0.036841200107272515,-0.08461389532282804,-0.16871282370043747,0.10426467407531975,0.039928756268356964,0.19238019686653549,0.21761059506022615,-0.07127150759683773,-0.015256467768105705,0.07847775138919935,0.03142245136357106,0.20716528324375486,0.13982770436867042,0.09218743199322928,0.018407847849133383,0.35392845082738705,0.17427568454503517,-0.1343550782971672,-0.0031593729478267127,-0.18017952181167268,0.10128346362705978,-0.1925722411013686,-0.29191221331637407,0.0011264634173029737,0.05010129139806476,-0.17301364959315618,0.06591491606764109,-0.05222567146875658,0.3160515785084734,-0.06465720836797999,-0.0719040212365294,-0.21082693951344175,-0.2800020479641774,0.06697279110926951,-0.1593316908590307,-0.01678793231768317,-0.15892722338143792,-0.042875507576223146,-0.17647351112365656,-0.07738738699490423,0.17491367893755688,0.036532032411624296,-0.13313867308610364,0.10305927722037979,0.16611589963751622,-0.12149505798716402,-0.062157427063718226,-0.12296021998479872,0.0385512908443014,-0.047526131752740876,-0.0870766720689572,0.11601918667080754,0.0949690097828099,0.035640247559473,-0.30610707676902515,0.08575905130968618,0.017453471046807282,0.03042431028221397,0.2835088950244381,-0.15282331317586237,0.09869788089319466,-0.033777182600364136,-0.10912812428899968,-0.1472202908725522,0.0591479677423491,0.1635379241042766,-0.1713193284799755,-0.3014612962120003,-0.18378283971850987,-0.0010635534063960432,-0.044002945430827056,-0.2692648938091062,0.14353674491768806,0.26609424254051234,-0.11310727793203734,0.22504560758549233,-0.01690439057969054,0.10773065989199816,0.0030936361412501508,-0.16870825416534782,-0.08706182005365061,-0.033864708533934494,0.07294184509109881,0.10984870797638879,-0.12708743532423142,-0.21083419368165932,0.00810386754768687,-0.06730224913332167,-0.008519853357612532,-0.09363576912646278,-0.1667234371214496,0.02785091740497093,0.017421331244912047,-0.3020001658193644,0.03576276071540258,-0.1304756279705019,-0.060065381979357575,0.27865550566694197,0.17013538550062923,-0.08244134990530218,-0.2477220922765248,-0.2697463147748937,-0.11561230570815666,-0.02757824177966477,-0.11883207370698237,0.03078766950222143,-0.03569082484056717,0.12674495516689163,-0.07797316753747778,0.20401846884529998,-0.05100042898013204,-0.10543388185456147,-0.10422326615744618,-0.3041100696544891,0.010526918836304664,-0.19505109897903916,-0.07457351523833589,-0.15928160669433014,0.17850233760391454,0.09029623824270092,-0.044816016426657425,-0.0442528694779516,-0.06478682164928626,-0.11562875052861163,-0.07618805152395529,0.08869511265428157,-0.09305427383468708,-0.05522426526211606,0.08114936998732149,-0.0729868777407821,-0.11321719990839851,0.06476056132571033,-0.07275569057440558,0.13318827079384032,-0.15209396172960066,0.2672029244417498,-0.09493942550596238,-0.04090377413233567,-0.07274375788440567,0.20072766531387956,-0.07772597764076385,-0.15218202216427476,-0.030609708738684297,0.09521080483505802,-0.10461158199664168,-0.0810014531334965,0.11832061872855415,0.11503874939131462,-0.12669832701850295,0.007125483046894604,-0.007089846815929051,0.09261325073265524,0.05483045730357066,0.3894987687989042,-0.01594202080420524,-0.0014599793872390172,0.01745584797976143,-0.0511946509955773,-0.06107310673733719,-0.15013352879410388,0.33847557732273664,-0.047687359574725634,0.11671027660343491,0.11234134052345497,0.001460558316409804,-0.13771638995000432,0.42899074436167894,-0.008075287623348916,-0.16922626994820814,-0.1308335336935856,-0.008575836664823101,0.004533124169183781,-0.28826632142359293,-0.181211581755625,0.20785242511070354,0.10304507458303484,0.22589342052638728,-0.16888058370917244,0.07258645993793095,-0.0003213668908723657,0.10109334729988559,-0.04029603214370882,0.15546469301425556,0.08278361264832722,0.09071819841017137,-0.14947563721844678,-0.06621152037615557,-0.021777980405084864,-0.006848707785784739,-0.1437243915262362,0.22825177863938684,-0.07843816337677056,0.1908418395438127,-0.04253673093177096,-0.13136101286114954,-0.09348084431117129,0.019523316163803216,0.028712732987358693,0.1584919006598318,-0.20485514818931766,0.23269892286949007,-0.003805535259386018,-0.05069276036843749,0.08055016108543823,0.03734277211999076,-0.07970337471330759,-0.08282098820235874,0.10291534089026246,-0.19981539041534171,0.11670777099588862,0.15457008305316802,0.01799389334914611,-0.07110737504933029,0.20442070599293666,-0.27407432578886215,-0.1048130313773654,-0.12131179113155789,0.09746282960451051,-0.06424336720266055,0.0201647196925339,-0.04276239000607282,-0.05289444059191682,0.37049159702284845,0.02847815571055393,-0.08416052788812893,0.34227131834276403,-0.08350634403734523,0.18999320591006696,-0.059570747105702766,-0.08001544527087912,0.004356296649027982,0.09910539403090839,0.05624404087246134,-0.054970971136495746,0.06647752004755689,0.12112746858895881,0.5246444912050549,-0.1233421846360392,0.014671549957187821,-0.022704272901507658,-0.26992307768845825,-0.10053946865931196,0.007246379219386342,0.10092258987255401,0.14046333818280549,0.02132431523801947,0.08373642910799912,0.13667022596215359,-0.05553311932014457,-0.27454940269703315,0.23423667628767272,0.20864932909644282,0.07762664127323435,-0.13127182317993685,-0.01527916165862012,-0.03576518041272582,0.20462398853692748,0.013899853942614487,-0.1307943119511055,0.27523596754387697,0.006459944159116478,-0.09980874268520089,0.02190199537489766,-0.16241851523127854,0.18613065743365387,0.04159534332529644,0.18589955225030932,-0.13474233624846466,0.10189540667861162,-0.21858957294762485,-0.1513979055091335,0.009970076020905867,0.043852787202940946,-0.1158436290414214,-0.16524905309114182,0.1363890801119831,-0.30417579311236514,0.004124987907270964,-0.03534125635693939,-0.031745635367827284,0.13334622148013006,-0.020368193943572824,0.015161658023719325,-0.02414244169267798,0.2515155702852323,0.20438774478122854,-0.15990980258637244,-0.22510802016187287,-0.1380259754616012,0.00294017986372244,-0.16725484653435374,0.14943818779159,-0.16694681131026531,-0.0810408779879273,-0.07014945814641371,0.11590737878805213,0.2883954392793788,0.07021244531928565,0.24272644940520866,0.1422918324581385,0.013573947769624407,-0.09872989463788796,0.1065667698122327,0.22153810729414622,-0.25285051610637066,-0.17788843816694894,-0.07553584963241974,0.09244987738175225,-0.15937170367896902,0.2910532142455041,-0.03600914686182797,0.3485616259048024,0.08480981275786686,0.04547837749028344,0.0170025637434177,-0.13314937053190293,0.024647796486956346,-0.20302245357247445,0.14871109009104203,0.4560166070821655,-0.008984636345340625,-0.022666730082968944,-0.18763468574170541,-0.06522846844780915,-0.039285515201521536,0.5540649506535869,-0.16398785881319572,-0.017048234493080683,0.04062012079085405,-0.05724280946718843,0.32758253090823247,0.010589129514578346,-0.10891668910396363,-0.039537091318224644,-0.2545067555288376,0.27611704307412,0.11958849114108129,0.14562115338168985,-0.10930216477791568,-0.050150589252032696,-0.16604797556858902,0.2195552719280211,-0.08626867632693641,0.08192491923280651,-0.007343452102283766,0.09844393335654397,0.038398468318866945,-0.01860310771255888,-0.04499821972986834,-0.04117958357325246,-0.1469830678108318,-0.09840071953307104,0.0218901409506346,-0.15973061292850768,0.025463262080046404,-0.03195589989652729,0.004203216263883867,-0.00861825669577911,-0.04989069572102013,0.14701868841592483,-0.0048280280992086975,0.2864904784757542,-0.08114264169381376,0.5976074184354419,0.09714160144609194,0.08784402743544088,-0.06188176213937608,0.3262067725658066,0.2692669997145101,-0.11169570348474268,-0.012234504512457268,-0.11209218352438713,-0.027791435046338388,0.12535909465605055,0.222689643776511,0.005486942604659982,-0.29901407215485604,0.46517473981922974,-0.12576037287336103,-0.05732007834770398,-0.03372865990243989,0.13046767484926713,-0.13017955733281633,-0.028141709835356653,-0.03498576275782721,0.13171851962170003,0.0658186177355995,-0.0066067324922013,0.03273167715186603,0.06898626115030784,-0.038340300357216986,-0.12631494541391122,-0.10637347465902669,-0.034379376377093214,-0.08570138610838496,0.051787769456677145,0.010070057819054656,-0.21099454457110264,0.5366054866491411,-0.08716169932729219,0.10348705414980615,0.04182694996173822,0.10714089438489556,0.0011379016234214982,-0.1311970169400733,0.10965236438471701,0.02850161660876716,0.06848938850401004,-0.0813503457649997,0.10695758003790228,0.03061027513894971,-0.07062729445742764,-0.01599564985627197,0.16629839891157527,0.03987699937261825,0.07011376299382757,-0.026368726323191028,-0.018959444599082272,-0.058219760455329374,0.06948752209240831,0.027471343147938712,0.0651072229291504,0.5962415240707561,0.145658651906055,0.21527312715691918,-0.2576639295045352,-0.07118083734828003,-0.05219465951098624,-0.01937386206092318,-0.09997392432229772,-0.02940165295166567,-0.03274761389259082,0.06882532931935086,0.06913315176562221,-0.006062256048367615,0.33940569067386783,0.09070086368162261,0.07946425641289465,-0.013259997680735854,0.4524434066061329,0.0983722470541669,-0.11153292738526784,0.04064959987113214,0.19496162364283837,-0.040703746576165684,0.05964966133452271,0.3427969225919145,-0.01813579798035361,0.004475176619602948,0.027483139680295046,0.15799111907535607,-0.11128055950430457,-0.05263026544619742,0.028077356996363483,0.006188882668737108,0.003419673413620008,-0.022654295585051644,0.17361236572117678,-0.047421803102130296,-0.1740420253714366,-0.04456023939245123,0.25791141312591764,-0.16117891116135696,-0.14202722691483446,0.05736692424788072,-0.15767412339888184,0.22637603707754522,-0.04162782225966509,0.09100780851872294,0.26365321159500404,0.10362938834082365,-0.2121792807221,0.04979621263701082,0.12654169705512835,0.0974045720397882,-0.020238277679253392,0.14157131008361046,-0.04910785315591254,0.0879128415077588,0.17812692370153987,0.125651302692993,-0.06347025624832042,0.041032047548469563,-0.13188631562775013,-0.3006839740540282,0.06449412057048279,0.18873649138747148,-0.05613471567605374,-0.049676642364449915,0.05938606283651535,-0.1540832341616426,-0.032311690080690215,0.030590877217062468,0.29409472397862907,0.02366091048168518,-0.041146109933238716,0.12172953955606473,-0.030955305914226173,-0.2107555808621392,-0.15779778048882118,-0.029561661470113812,-0.015508585154724669,0.12815650347827778,-0.17258032272220586,0.07046372646236185,0.024553600758176364,-0.12252031747318795,-0.018249579669239468,0.06192465530884575,-0.07512770570045801,0.005688336744448376,0.19038810037527837,0.0024401517350882175,-0.14718740385247087,0.03675905778007755,-0.2262288847071301,0.14347908784640337,-0.07149589684063234,0.10856429415409051,0.06036725432330426,0.0013894066742492625,-0.10030699474104754,-0.018737429171923677,0.0201597367287245,-0.05166561697887272,-0.11011626593205616,-0.0439641574069562,0.04372844756187354,-0.055480586159878865,0.0035439925606460462,0.02701891180887291,0.5852127297757428,-0.10016700618534995,0.48553254520902206,0.026115979950606393,-0.08101447575306026,-0.17952216358261916,0.05666356434306747,-0.20274431542488341,-0.031042561894379175,-0.1027425417167546,0.06986413740785584,-0.03628068582706244,0.031361746964239615,0.07297325028839187,-0.1172727702850594,-0.16018985879180014,0.1491352025817487,-0.009675140685687228,0.17695730386188738,-0.034039230044961837,0.07902694817050065,0.05501234881094385,0.30281118624598624,0.35721112228059443,0.006053274618258667,-0.07824566596229557,-0.08161387741016374,-0.2293818677706167,-0.0508690280794708,0.14614672819451432,0.04615358543817117,-0.29246493616005054,0.10953263829982166,-0.1689362150266491,0.01017016027063263,0.15016061484141488,-0.05998166964701532,0.09049845919987173,0.019588404187384716,-0.13708263702555518,0.14947619602537365,-0.2775526602565488,0.3559654059907131,0.04066305383379468,-0.039158214213000316,0.024458964439832757,-0.06451587814113308,-0.02289086265749967,-0.09705996223617273,-0.17230329750162082,-0.13741564220121774,-0.003929680015435544,-0.13416856634144364,0.040577646366320545,0.018490828745082687,0.005328532776671232,-0.3022195088757019,-0.15961780527340125,-0.2968431360559411,-0.13450024797521062,-0.09356813330872125,-0.1634341851922709,0.08199178165641761,0.047683747958049034,0.002655226848078262,-0.1852535773603044,0.0008589255105524943,-0.10843259299561044,-0.04193956293130058,0.30839376269228297,0.06640096782880528,-0.005705699896658269,0.017494892398522244,-0.09924275522320188,0.1964749225044301,0.21988276678268942,-0.14479455798218682,-0.152302339617277,0.13992004327317473,-0.12348372402859274,-0.1203232764344285,0.04911740781479461,0.5044271918038175,0.032827600632034264,0.45305427426764755,0.2233362833408133,-0.06366605985995565,0.003193549917900766,0.03300629742720841,0.09580379581620464,-0.24472598821317884,-0.045605633524558244,-0.20026912715869324,0.1023933470564075,0.07648670490763611,0.3035650383988929,0.2092131928127707,-0.04131819872304978,-0.10549777825669693,0.1597464203001553,0.012420139879212994,-0.1471360350145287,0.09377654140320804,-0.058102782664863344,-0.0062643405068981235,-0.045923573375321054,0.3796304588246623,-0.2857993421080965,-0.22214334671541194,-0.084082950007673,0.08276217454121695,-0.16711745102672224,0.09542285915166106,0.0575194602644246,-0.04037708599418935,-0.08199982099728181,-0.04796412759021342,0.01800243999723082,-0.10930256868999975,-0.03789858276267926,-0.13632537057529023,-0.03830056242373315,-0.2729177397335059,0.006629939467074836,-0.13508440126856183,0.040906666687762556,-0.16325385980786472,0.5802110158428896,0.15975116915604406,0.08346579990536787,0.02562345507617099,-0.021745980053971332,0.23153392560823768,0.05665844176368728,0.015099907099073724,0.188870937917932,-0.026296473737275738,-0.17385413218412943,0.017128415328871566,0.046802011666446874,0.006210201100042837,-0.1061830256160268,0.34203594407599,-0.014915908762536456,-0.1375990234778309,0.12503553791665695,-0.0642765512754352,-0.23430816852563965,0.16063053069752542,0.16534147096013052,0.04389378862048927,-0.0536398053192807,0.020539303691265293,-0.021332610101603974,-0.17000531489161577,0.19840372807128004,-0.10534947379439562,0.05312861251533389,-0.030640663477997685,0.11165692404356434,-0.17606554485426923,-0.030645032443912642,-0.05983997192864932,0.015971967817735643,0.1681236916459042,0.033850480205480205,-0.15838488899779735,-0.012268586661675409,0.0006631171191656159,-0.2020170996993589,-0.050263878055850156,-0.26088199526386113,-0.018352547231046023,0.062481014863548595,-0.09776089746743978,0.06000653215568406,0.07250609524611618,-0.06420216123114757,-0.08480724409057303,-0.1831559991115098,0.07527291037340816,-0.21394214701784467,0.0805404984562343,0.1321821516017629,0.054808679364034205,-0.05108556535747879,-0.006763569206590815,0.5928151000679032,-0.01694788979735123,0.14148913675363456,-0.15255916378473994,-0.02116179458266112,0.037127654805306894,0.07877460943371395,-0.06726913780765689,0.0390291627632563,-0.07649480250647263,-0.16858558133012524,0.04628316835613859,-0.06004408736713022,-0.036285211325201065,-0.10231430025026336,0.06158788978633424,-0.04449416518106267,0.06301077679937021,0.10163260395528464,0.12971757329899142,-0.04102440190708769,0.05704544851562393,0.06081024351355076,-0.02734274385012879,-0.08866700538162324,-0.2070823083892161,-0.06110806396415326,-0.23606282890355887,0.03420992389539109,0.5105077379184005,-0.026303348160365055,0.004488786491236685,0.33139363960890117,0.07698120209539855,0.043460923360349485,0.14300907240228808,0.05651785902237791,0.19960262340262475,0.5955811002944431,0.13274561779847016,-0.012938069493217708,-0.18146230716030093,0.07418392032250914,0.01960383136793835,0.5683428080102776,-0.16096959390157695,-0.11629729176600888,-0.08825389469439296,-0.06496512353144054,0.16993318139100194,-0.0706650106513004,0.07183774367101663,0.14448497934010426,-0.1594796377155254,-0.05005182445732088,0.22075627498743897,-0.11704060739494633,0.002557450021257129,-0.0036464276276342114,0.02700110769091161,-0.04013199303841089,0.06930776784096965,0.5694503345514597,0.10522432636223707,-0.1641883680061746,0.06879734838274171,0.08618851272434629,-0.1140075322496148,0.4211406196146597,0.09307539326329055,0.21912538407229432,-0.27568459853315685,0.06113204049988193,-0.029876733494194622,-0.06003388544812614,0.08957486558276095,-0.06548612775750016,-0.03037479786070703,-0.10985495908764596,-0.08349098207696315,-0.011620342201058322,-0.0052597956146625295,-0.10304982080535356,0.25054734243412397,-0.16230008640477803,-0.10634515416077671,-0.04246837325997015,-0.08834708708366822,-0.09011372716175808,-0.0872870636940968,0.04903060856145594,-0.041146173159229214,-0.10232623054804353,0.11418850537072289,-0.012227432612693013,-0.14458399257341442,0.02279816472935318,-0.013297397738943782,0.12565882499501976,-0.1445035173125502,-0.18939138625405116,-0.20927313132050626,-0.09289043519207546,0.19454996717215317,-0.030547605666092215,-0.17431348896775578,0.07229641579097512,0.05815261926549043,-0.1655819364095942,-0.11799085770770121,0.06869373087795172,0.12389289155958082,-0.08676345371081373,0.020023617097738353,-0.21523402362532607,0.019000825524265155,0.09199356991369381,-0.08867832238947564,0.034403092736800554,-0.05303740274963476,0.14230392396608713,-0.06728353590404088,-0.07797541368329065,-0.13626718452541878,-0.10959171958320861,0.04968357478423276,-0.06628979344146387,0.014689911390095692,-0.07917641370344684,-0.05814736245053271,-0.04925887023522395,0.05686874009999619,-0.10340979915732777,0.020777716264332256,-0.07404234026453724,-0.11858283041546934,-0.07592469221039091,0.06703430210687497,0.035721776542580116,-0.20854445631840376,0.0848791977680226,-0.061872851054453705,-0.11617492138599152,-0.24701417547836338,0.12698545162901395,0.008526395896272985,-0.20770959166637476,-0.08380640958609226,0.34111980786942836,-0.01685434984345733,-0.18167895785029659,-0.13484207478231414,-0.004679392000172386,0.03798503449268136,-0.036817624737837895,-0.029674262546524367,0.009497320803834631,0.598376403511707,0.12168106932080633,0.018993877621535295,0.46659494952554,-0.12686452716018068,0.13667761012327542,-0.1368238000742063,-0.19655746953412503,0.5944603579768989,-0.01872177858397578,-0.1561535845505656,0.004961440472064196,0.1332637138137437,0.1034910413734426,0.055277574323667034,0.1100990750214533,-0.08065308668748407,0.15458079485864137,-0.11776044195457794,-0.06647478552400089,-0.08384270275260454,-0.036397845385262105,-0.05176921654232171,-0.07067855196724221,0.11074097541757197,0.3992748748330582,-0.08773883257751221,-0.04761250698432244,0.07466501041425164,-0.13137319547658988,0.08092837855709833,0.00139229944376431,-0.2246668363108112,-0.04056199261704046,-0.12934184822731998,-0.045000647054309076,-0.07017405802276232,-0.09828258830327538,-0.04134359638344218,0.05517268320622216,0.1665578001193713,0.21013931505161598,0.3936384347727866,-0.11077157747951456,-0.06495162701245345,-0.040267826101782325,-0.14133174376094473,0.15713432129586852,-0.03821235446950938,0.3202700885898707,-0.16439459100165293,-0.003358802252566082,0.10786478008676118,-0.2851955519188702,0.06393350696371258,-0.1297754357952282,-0.031208848813452515,-0.09412383708535711,-0.055834039380484375,0.06093241406743422,-0.11032253375022695,-0.01775194006031551,-0.007040360817812268,0.026838891602079695,0.03320610226363033,-0.018107438164992662,0.03931310210136692,-0.27717043400640184,-0.16916937182603853,-0.22069497939639196,-0.015976142781722984,-0.16492792457034014,-0.0648238587845961,-0.024136716813565176,-0.10851037852108132,0.00026156316304822946,-0.01684322240629321,-0.0488614348544923,-0.03666615058454407,0.12013067141368072,-0.000643174530790301,-0.15589350669889213,-0.16895456933358385,-0.06863017894728753,0.22220635495129587,-0.033470167083333675,-0.0400042367813202,0.036079530572888055,-0.04687487669497555,0.13890223412484798,0.05083678828144911,0.5127063472772244,-0.088653502799903,-0.1802188805983074,0.5835252667955875,0.060493571925383495,0.07784471550070249,-0.009814578526200858,0.14860358449287384,-0.05210285082646526,-0.1643542552810814,-0.08358190310265538,-0.25733338414249257,-0.0391244209226534,-0.2648456724725491,-0.0042898618444224875,-0.06581087658687661,-0.13735164453324178,0.016152880971680366,0.32247971875786446,0.10281617926092891,-0.06084567733553591,0.02686708068024057,0.07234989726866638,0.07097684735289708,-0.04272183276034138,-0.0865125900556136,-0.10171873341648566,0.3283636530495157,-0.0007063039979087705,0.013582151025175123,0.08100277107805928,0.13181119805545072,-0.046105602854526494,-0.2607617549125944,-0.15218905351110174,0.22276291934638925,0.1520116961189005,-0.020495686252461544,-0.1086792530799108,0.09481210554634788,0.5855069386504593,0.019771417996144007,-0.12791285648747774,-0.0070175505643398275,0.028867858993857392,-0.011670586617141616,0.11935521033075311,0.02656759865425082,-0.12382178015646651,0.06836950290981135,-0.1998684694081554,0.01318427025221713,-0.26215482531769163,-0.3044708454902573,0.0185383325487071,0.31785984646751875,0.317338601069254,-0.042186271842931826,-0.13632176826330922,-0.10760308086101307,-0.003121006249133739,-0.015462285797934693,0.24511444988419542,0.04957965921316529,0.03886221989616059,0.18506818046098292,-0.11450857337286073,0.0007642484026601426,0.08108803427389628,0.04241873587438257,0.09208747640176951,0.20898586899566132,-0.04155344484727573,0.03622043485950469,0.17114285136477153,-0.1035875890390986,0.3729560293849625,0.09474077258898396,-0.12172829009478742,-0.03381427272807601,0.00809012286241443,0.006661095963189284,0.13776166324649144,-0.02875028442157128,0.45989230973437584,-0.1804596810140742,-0.018678303065966812,0.023183510480386808,0.019270407405163042,0.08040555720736756,0.02730950331281517,-0.055272054881035315,0.0628894357179859,-0.028033589719017746,0.005351480186442116,-0.06004573296469223,-0.17169333996164188,-0.03512092062459097,-0.028045571139917735,-0.04661284636782352,0.11486419608744139,-0.017294854310776497,-0.11307008254159634,0.014490141674846179,0.0737448961472621,0.030298382878369987,0.03998787407819009,-0.3062474323781432,0.0854285093055744,-0.02721793703720254,-0.016728821103727528,-0.1366059670876955,-0.08325126661299756,0.08072320958077217,0.011675553051716966,0.13600361630008959,-0.1154926354218395,0.5381820928961947,-0.24433555453105243,0.14319511873489607,0.07142952755236247,-0.05483932057900105,0.03974035410647762,0.15090773017144068,-0.30193222599619257,-0.06365113428771048,0.12617941309450795,-0.20888385388466543,0.07335598691665828,-0.0037153242322041395,0.033736599800416205,0.01577080679582647,-0.014880717305826864,-0.11427175789982746,-0.030131565199358824,-0.19783970923468783,-0.02499619841656289,-0.008008098711872972,-0.038129095333030646,0.08975884099537224,-0.16479582808736012,-0.027151593584471424,-0.17837231229945094,0.02916714443432245,0.09510859224467165,0.03199559809829407,-0.07831390710964019,-0.07194652818322388,0.24998908381600218,-0.14344150064234923,-0.14158104071253672,-0.07069679953429737,-0.1631523502469782,-0.018417168564403032,0.04145611273123813,0.12475806212614127,-0.18325634230537277,-0.07734935545488818,0.03813251400779771,0.0631882765982086,0.012846174521408734,-0.03544572214383965,0.1825745358245641,-0.12421395040606421,0.07232112171450907,-0.22019959425528782,0.15529577895165703,0.12913385806309005,-0.24340174708791162,-0.07438305332770007,0.020573268030748982,0.053903748958732754,-0.09013431743349808,0.04755201482753346,0.059456759410361916,-0.09348511173673263,0.03753406010646476,-0.11213431037829028,-0.16446302947789848,-0.060210205427932005,-0.10562625169724009,-0.16702444989049825,-0.06983915434847524,0.48432142155485386,-0.09650485290856482,-0.05510747890748409,0.47776140931350075,0.020490627874640146,0.12557229399925265,-0.005660435334383827,-0.12275502584593755,-0.03905861158975618,0.21451666108435125,-0.07245560483014526,-0.20250289763144624,-0.07041622827175617,-0.12684184301674128,0.15507126529052498,0.1258732715622088,-0.09343949579108475,0.06279443226256642,-0.0227901887433127,-0.2118565491083597,-0.017791587313366535,-0.11843583943470028,-0.07202680573467662,-0.027213203151831337,0.26454011612474243,-0.18256267351652555,-0.021817980228843757,-0.17133661726129482,-0.14655400881723535,-0.10453179187564773,0.05422882926598368,0.025180551273351212,-0.11251972353919161,-0.11548292822955544,-0.22954773543760104,0.17594593751828072,0.20776647913131988,-0.08155168443304618,0.1142696707329882,-0.167541467477004,-0.19965723432795587,-0.08560736679707465,0.2994198025240143,0.11659468313794716,-0.06414208393198312,0.10771576280451049,-0.2329351013650933,-0.07789973723508029,0.037684143849874306,0.03612706008792641,-0.15900392260839502,-0.06487896703790083,0.29485659251155155,-0.08887607224735515,-0.05450287846307018,0.10990492529435222,0.015683534311376146,-0.15751893801600794,0.013447254228714348,0.035579400580144846,0.1634636585277892,0.03444240479725503,0.06351794415691983,-0.024845819432539232,-0.09132539915519702,-0.14609608627973014,0.23278786734069698,0.09135088549584246,-0.03976306815787968,-0.2598128612317059,0.06111694573730092,-0.03628281991941498,-0.0326799406743937,-0.3043731088139619,-0.14873916320682054,-0.05394318984459663,-0.07512439195476132,-0.18225865408164788,-0.3020001658193644,0.23559941418906943,0.013429463425747672,0.4146973116210188,0.08224935054075623,0.08639969285736802,-0.008054688014953794,-0.01546406668147562,-0.1061908427992786,0.43868357248298695,-0.09345838889032806,0.011031365733416443,0.06423679258484476,-0.08162708994630859,0.012517849318068127,-0.02677708011628865,-0.03878868785523364,0.3236625082404488,0.04467559024994063,-0.05671162156514299,0.08894244180490112,0.026167028870280457,-0.005788962931730827,-0.05560748505124923,-0.007828876785517294,0.18779021338036478,0.049686071481595526,-0.15154852242399044,-0.03846944409395996,-0.04782314271205359,0.14484491604804253,-0.23439218328254474,-0.06927807945507304,0.11821366655391072,0.25227999295286313,-0.045156112392442674,0.4131693429656021,0.23150966630357595,0.013653465496438437,0.2197144610368877,0.44697341357278597,0.005268439887508974,-0.09314661075437876,0.07191696157579978,-0.11857857496243905,-0.21625514523103606,-0.00033812959263905117,0.24832553142410646,0.13238539544480651,-0.1193838300845006,0.029691430755068454,0.06085340899428231,-0.1918012169911981,0.009521174764606552,-0.05197971905033379,-0.1412528893491762,-0.06571957238282469,0.13100122246645995,0.07272939497854644,-0.02281737365097673,0.3102946732275986,0.1096969772571126,0.12737363053302767,-0.14846754250860045,0.07074040300192924,-0.023889610247770584,0.07450848540357496,-0.0864780327044316,-0.09723191861186095,0.09829734661604966,0.005190650280652141,-0.06589311154054621,0.002386716386749876,0.37817001858115706,-0.1772809677597012,0.5299133527485193,-0.11742526682321042,-0.07060877417323408,-0.10105770181351152,-0.09066975585247575,0.418604050605726,0.19648579134315466,-0.02169274632313615,0.05667665678887511,-0.13879761752727596,-0.09887593332596357,-0.08126289679323,-0.011557625754613459,-0.1468354690558082,0.018630467561464085,0.15344786423099627,0.25370920896803095,0.06848965802133446,0.23232401973652875,0.2446489816827243,-0.017610655670219758,0.00782767741006608,-0.05575153164952106,-0.29148688366049164,-0.08371314061224906,0.08924187395040484,-0.019590177679220198,-0.07817277102839465,0.11890324209849797,0.13724979366502724,-0.01584983819775565,0.017478931795018145,0.08464911028973704,-0.004134474705239775,0.043960900969217405,-0.1947861267924814,0.1152329224627532,-0.16143164579546324,-0.08966471298595524,0.05097710865882617,-0.12132834181666806,-0.21110898810671777,-0.0688706793807335,0.05582939076218185,-0.12384348249717564,-0.05778049717760431,-0.05199248366269818,0.009458237999427483,-0.04077206959082519,0.17063389794385858,-0.19132447286252915,0.04895101527618083,0.3843626514112314,-0.06575786205574977,-0.08567228885909615,-0.07942833416983035,0.195354876340136,-0.20078311347688252,-0.05705633306625767,-0.046613636444193404,-0.007566758246426636,0.04149916755503683,0.1361460112607104,0.04087108006391425,-0.12380500022106214,0.1528531280844507,0.048180102895874134,-0.05046277740805553,-0.029627900504519442,-0.12024314582032052,0.019688924044023656,0.03120200109953207,-0.1344667649111635,-0.1146622659438935,-0.00568873106057023,-0.027509441555269784,-0.040027645060104616,0.10136274261811887,-0.04821567539301448,-0.09214498482512748,-0.1132668958312284,0.11383173188446465,-0.19313007347540018,0.16802383685368558,0.01536421868566184,-0.02252613414704963,-0.05517397415419655,-0.2332630340150058,0.009251331146635585,-0.11481599953243436,-0.12494414777681993,-0.11469052659854535,-0.18602681366121918,0.031755073252227255,-0.26635373070362733,0.21539413973631477,-0.056841323888762826,0.04073200858699477,-0.07587219748701816,0.028045883625510352,0.031157875751966876,0.33039954289213186,-0.04512728240765451,0.1608412762719492,0.00478819324195086,0.015316005336337648,0.25646474455487983,-0.09228878507059807,0.06208100166097794,-0.14727981058358827,0.23366029295199592,0.025238095721673407,0.18977243977533043,0.15834270151482455,-0.045330876708556456,-0.1006053489466311,-0.18662835560942845,0.023839341973197918,-0.16715039412784508,0.30627842605532885,0.023755989399990293,0.02498674453188475,0.017750009097154646,0.04345092067368453,0.3417089559184389,0.09536878318304098,-0.1691667718161037,0.1525375838897775,0.10446773833913606,0.178083227245278,0.202655619906228,0.11344356065999524,0.19947294234231602,-0.05580818952539313,-0.09454358793494015,0.15093085090462996,-0.07480122468718027,0.06913764304085442,-0.24119776609569882,0.28888580880112674,-0.2361258749340779,0.03854582110831256,0.0026117710247389055,-0.006800312822369083,-0.1806394985358475,0.07397781782461296,0.013027704311983343,0.12810887185031275,-0.10209321289850512,0.08320492557695298,-0.01021162930622112,-0.2507677546187504,-0.03940352721089842,-0.12019821317300015,0.03663975469030869,-0.2626610678166989,-0.059563600890760154,0.018833284151943668,0.04385914913004687,0.0024027880345587403,0.060125231390656585,0.003566898238721271,0.024868649009543338,-0.17497877235890116,-0.008484941511509103,0.1252642669525805,0.1815148769066014,0.1052121464130771,-0.05304499508743986,-0.0705844999583618,-0.017488971551368235,0.0819714930520763,-0.21803266072423877,0.04238658369614866,0.09169908339905254,0.11692843125355236,-0.014932821529012019,0.0011317545931876013,-0.039888843463800366,-7.539839915217354e-05,-0.19785822148712373,0.032958371189756495,-0.19171349768882895,-0.15008765425303322,0.16622336590945902,-0.041285896586362364,0.01738666794890803,-0.008666579878927449,-0.04905794846031475,0.10835949974233476,0.11804512868026126,-0.010779217808746683,-0.03510179564862383,-0.19667147309831834,0.0011807449412392583,-0.05370177473395394,-0.07060823606057134,0.4409482398906398,0.03993901073269915,-0.16423486513822788,-0.08504164283334004,-0.14192839719509973,-0.13395061944610603,0.0140521312208874,-0.012477816795276589,0.0854974727837952,0.035981490807281365,0.07086595629679879,-0.14348670202635438,0.09932987587354292,-0.01985175200501831,-0.05435545643159618,0.0025066476015947387,-0.10908918748882249,-0.16723517819278377,-0.07541847084059226,0.04631917202896032,0.3975170023638623,0.09468798114623359,0.07147333453324116,0.3264499863957056,-0.11283570229302627,0.03072686832909103,0.07717191265489469,0.14459790958913743,-0.1581778339744314,-0.11652551008584938,-0.14518373445929397,0.003073064915601463,-0.09650586068478374,0.10291100690884725,0.33271354199021685,0.18852894917160723,-0.11941086848656868,0.07764168965448844,0.008448812034522456,-0.19921826415542304,-0.3021434744238451,0.30592014841148524,0.19868155565946632,0.14199662034927032,0.014650050751132307,-0.18250902623015824,-0.03140463137856519,0.42911535939985346,-0.012082382145543342,0.056157708042305125,-0.06882998007411648,-0.023308169505957786,0.014185575176873568,0.1206630807854298,-0.15902321172746584,-0.11746174719564445,-0.10314855173734884,0.04833799027012859,-0.04928944814068231,-0.023807420271739637,0.21732082115905185,0.03914020554550929,-0.07921547154451912,0.11129453791655301,0.01525365431368481,0.0988533592288911,0.004365643004109961,-0.09915077741735645,0.16884749658655163,-0.030213274371299555,0.13146350616187805,-0.16607592769278925,0.007714737989199195,0.062481526427344294,0.06988598090250349,-0.3020001658193644,0.0758761020534042,-0.08281007429058941,0.15605957226074157,-0.14806080219889092,0.19406988495240785,-0.14818165910852735,-0.29442090632850976,-0.040916523631653275,0.10369347314490174,-0.1092999852383309,-0.1582138166394719,0.3603806185449012,0.1697301929835119,0.2723752252753766,-0.10419138697935942,-0.2347160599757879,-0.07165033694535619,-0.05039345201695528,-0.2672780836188553,-0.09180935089948271,-0.1424832959106497,-0.03139722396277122,0.0009207719437271931,0.42480647951255424,-0.14856410528076125,0.5866378786109712,0.15161581248231593,-0.09426026151469066,0.025721804299682866,-0.08980677109586285,-0.014422703282057128,-0.060814253211865864,-0.2570718146418903,0.006292345873447815,-0.07793212981734508,0.029497834103757878,0.14369772321485397,0.09046976831852858,0.11367114867335368,-0.1012252108228426,-0.13333228402430752,-0.01459992653419665,-0.09128807440744194,0.20246990501055157,-0.10909125918705434,0.06892897614133274,-0.09403306371429687,-0.06598367322755588,0.009839306832195208,0.07599472228320829,-0.16666507805610212,0.10749043667511199,0.18784466966606927,0.020398620768214304,0.15845779333388005,-0.22169603186823816,-0.07543664425244814,0.15215167244265168,-0.05394313560146367,0.1730365803119194,0.08833666432171344,0.0395069989311894,0.06420272775185146,-0.11870395886768609,-0.0634969092328344,0.18928974556928102,-0.14412884196968964,0.21040501965226735,-0.12728346913711905,-0.0224491449004844,0.10264399724863463,0.06648831078914112,-0.12328814141037903,-0.041631358504496986,-0.07175788456713733,-0.05130409615699517,0.48563028938870517,0.13245159811890517,-0.006205757224441681,-0.13647705091478193,0.012835516451754129,0.24427870188525347,-0.1641027186739329,0.032500825950703145,0.2538195938246331,0.10588870418045494,0.04838080328180614,-0.16039647675407392,-0.20650689299530514,-0.12296438540852932,0.176381170485822,0.4655439361913066,-0.04152727010536072,0.05786181583878454,0.46176772061702664,0.0035375160139429723,0.07279783847646627,-0.04270655779889664,0.10822388606366035,-0.009390108516249832,-0.13947796249468697,0.16585730964772888,-0.03192129325865724,0.10510621723804972,-0.026824426113152763,-0.14068031487110244,-0.04466707839335578,-0.19038002453187866,-0.009835834045616797,-0.032639435189953186,-0.05250245877293879,-0.14048616780291068,0.0745457243469272,-0.09773635680064365,0.08656409648044266,-0.17623303958890355,0.0050879371139842524,-0.07839215319752227,0.0028185289233496813,0.061788326853970336,-0.13427460181934553,-0.1654526316376318,0.09084146249434508,0.2137199552380998,0.03927409228385838,-0.007339435529754891,-0.14560551458832022,0.06637092428685874,-0.0765803564907136,-0.10324692194853734,-0.06731457906893643,0.02560102266659748,-0.1324795400786472,0.1945084741576113,0.23519147794439338,0.017509987693285222,0.050625387995502036,-0.04265661851916112,0.05997803541130796,-0.12508371954169423,-0.051262497866237586,0.09299120360048949,0.06437139360281872,-0.021099264505292818,0.018334487238180056,0.10806338805022676,0.37858831210999566,0.04341247280760859,0.1349240871380473,-0.24422220929116947,-0.0316044092393036,0.03126417208539067,0.03934603756367196,0.12637025269132207,-0.03400734968795511,0.0058893220457022395,0.01762155025503794,-0.005710421335498654,0.15463899693699587,-0.0881326554309248,0.08922800565039349,-0.016943260078892512,0.09088751543287167,0.0018035344660617802,-0.1418149923664738,-0.0038657009809756025,0.05999257809760441,-0.04583391368654945,-0.1467481360325389,0.06891592268199676,0.08729655336026258,0.06750360435503767,-0.22595509959000415,-0.049852162286566616,-0.11244058829508792,0.10813562973816959,-0.18694413390842543,-0.05874218138740281,0.04021014829398044,0.11078499066999274,0.1119081616209936,-0.1865907077346771,-0.005642671138583481,0.030826373133112383,-0.136987711619003,-0.180091727539493,0.033460446598506756,-0.11155167133442659,0.0032458109829907387,-0.017635922605400037,-0.02384529377481711,-0.053913091249707376,-0.020643746030553134,-0.047285034546904144,-0.1435197969041772,0.0692810124133303,-0.13684807382696088,0.1247157787580382,-0.0751409240402186,-0.21761004758480118,-0.05708348580737716,0.049547527371938335,-0.11915461775623636,-0.1643691723631232,0.12894227662617574,-0.08506914821123794,0.0016832023611032493,-0.11310546238168045,-0.14832767199943317,0.1625118375126742,0.11081019864753572,0.08088117816048915,-0.010882003359488426,0.12283152298239942,-0.02978436464277509,0.005666992706231712,-0.12329880195022679,0.15638513108201452,-0.09774912993264141,-0.09733484048590052,0.1108873156538263,0.07941398217228472,-0.10186200936024763,0.5076879256447947,-0.13924046168957707,0.40530391101367147,-0.12302341033993912,0.11153231904736885,0.05078805142609498,-0.1373382621485739,-0.05020434984701855,0.17940804912458003,0.45836281288657776,-0.04030301226869318,-0.12022283622055539,0.16172458624466468,-0.07690343372126293,0.16403686532676573,-0.04886245889957057,0.05481649744562691,-0.04729798249709501,0.07595288598832732,-0.05090755360188097,-0.015456549621461799,-0.18988460661940718,-0.05656019748059471,-0.019757849557233264,-0.16463517107147127,-0.04674992541655646,-0.04713221126976885,-0.20437704372027868,-0.04207817499339987,0.011114177271628011,0.04632636777529354,-0.16159973586488346,0.13093205168257943,0.10575826827629885,-0.019691612568704733,-0.13029939588300327,0.16302914789824005,0.03921611168039472,0.05435911286737184,0.019525756390568364,0.009626651512588387,-0.0803238217595076,-0.006310529217324276,-0.3017676565192944,0.017720454037233487,0.006886955845561671,-0.06242610631686752,0.08787189687922108,0.3094978424395355,0.15069083689685273,0.06664884607082669,0.38424359793740226,-0.028067060911825092,0.16853746385310236,-0.07321842808596465,0.054933067031555795,0.08147941734285519,-0.057141479711109805,0.12093902395466487,0.02743036127289705,0.422530453127532,0.05614322417288689,0.18228053852667006,0.0610810030616934,0.006422155839100576,-0.062382177556930145,0.07532601638932088,-0.03029054944910987,-0.0711052897247051,0.06283201511197643,-0.014148349178212062,-0.2513292766316424,0.10470772288904513,-0.13039836283756312,-0.012845682824434821,-0.17004686026696672,-0.13155571510565656,0.04037569324559431,-0.13007401804371854,-0.0943503143195773,-0.06371701566319345,-0.19042707547971266,-0.11742558939827798,0.02669009058318631,0.018214192623324115,-0.20943099869508963,0.07997384881286182,0.018662270024553857,0.18059495572099404,0.026367587238620135,-0.08089708033779414,-0.004336239941045951,-0.06273916160222279,-0.056391173845250844,-0.1678104008022991,0.07309848321394387,-0.009252239528281575,0.059220165871409394,0.1129193080712431,-0.292688969426909,-0.01491889129452062,-0.06901313807214933,-0.20762550609931266,-0.09337987023960305,-0.0435097260669347,0.11290045166059341,-0.04667396469480275,-0.15371016605984625,0.19547873899349788,0.14713417513924365,-0.009429633511283796,-0.14936890560305965,0.03126985538706216,-0.1090492589144793,-0.1802804342693451,0.3653354471135045,-0.09952876745595984,0.21016472163547323,0.11128411493797136,0.27901874078734046,-0.08907130437269592,-0.045702227845623236,0.0497967264188928,-0.1336329826272489,-0.03553308943966916,0.16223257023115883,-0.018145436481227555,-0.04285957526892171,0.10032223773562353,-0.07651714300880232,-0.008175312936024366,-0.04481121091760039,-0.1322175807214667,-0.16385260751423592,-0.23026430465089598,-0.17155052519534855,0.15742716894768907,-0.045096397646307865,0.11691787245494399,0.01835240885315977,0.17103835437829654,-0.1652531962036929,-0.1222183418161443,-0.1435927712004041,0.2151866133748412,-0.17088656160453225,-0.20876599783134225,-0.03012364774507912,-0.08286389677882064,-0.03540686849604808,0.019601874492631406,0.24769729914194089,0.059555919293548726,-0.12541164397343485,-0.021284819389931476,-0.17896960949742455,0.006962130032844722,-0.25920352771066113,0.09169109091248001,-0.11740591590808516,0.0750312943323305,-0.05576835673472154,-0.04812302849892793,-0.11530035928630943,0.05357658622684022,-0.28968361199021503,0.03948332491045433,-0.041673849857520665,0.007223841941147605,0.176481427355355,0.5087936579793885,0.07842243767829049,-0.060994840662120776,0.10260363534589868,0.04692469503199301,0.018272135508538657,0.03699512920988722,-0.017269043688463052,-0.002773345790026195,-0.03266914439013694,-0.0849942972642673,0.06538690255664431,-0.22407005693145265,0.06034258456515287,0.13113592928878756,-0.10005256924983333,0.03676674949339769,-0.004167131856946876,-0.050033186538371945,-0.2024845486455946,-0.10445414312733134,0.04427539846963929,0.010256493912913169,0.028784051654374972,0.0038053145193956564,-0.16274286981729308,-0.10442953726660358,-0.015750515498812904,-0.08558964496040353,-0.023703614561930812,-0.30480607294023804,-0.08908843646092761,-0.16864014957324203,-0.16302332115872162,0.019847882671368264,-0.10200986124627741,-0.05734674357991994,0.0978850256438138,-0.011427892927055682,0.36996718192152234,-0.0762203748220444,-0.0880403184446583,-0.055575691254622576,0.3061791002905893,-0.29066246317192374,0.08456357483205987,0.12516039408131707,0.2640802564866169,-0.1740711253278724,-0.08975163629317359,-0.28863940278428285,-0.10637148367115298,-0.12144159601053109,0.13960579758737668,0.021854074940228802,-0.13924661871213662,0.0906846927476159,-0.0162836169686694,0.15253227376913442,0.22546355029295995,0.011137884650942035,0.13328232405120963,-0.0030921827481751054,0.0013840255893187877,-0.046902343815524015,0.1589156325785232,0.12865740125911607,0.09625694100082752,-0.19571758031453376,0.07526735563257128,0.30293571578677647,-0.08129228723258865,0.10013837710412811,-0.07487251241173587,0.0044498467225958,0.018517760168656392,-0.01572114947821407,0.011574202391040666,-0.13670656822714358,0.03954477478320382,-0.032436937656758084,0.01283254698705526,-0.036066520861516724,-0.08624436349022913,0.03679038999327611,-0.06683152028277774,-0.1295687283868011,0.028567844093124493,0.29553295539915236,0.13726708901331253,0.020059730633664237,-0.11387560692211267,-0.12649561594678002,0.1256684744329928,-0.0469715018823122,0.02548758099674493,-0.03898247243959327,-0.009466669947468871,0.1296060354453204,0.2482084243307811,-0.1613542695596712,-0.04002635021291101,-0.12678180970244896,0.09114633938902075,0.18797683571847917,0.05631201242542304,-0.07554522103503554,0.15214869534751133,-0.238364238986274,0.34630805531473874,-0.12202669448294644,-0.027586818788414794,-0.14139593633793424,-0.07275141911152203,-0.0273167853617468,-0.12292178312150578,-0.07339969201741549,-0.04094398895307902,0.021115944144561066,0.08084450026507041,0.11304384851887465,0.11838867749088355,-0.15548263542911206,-0.0385947497181859,0.05119882789485366,0.4524434066061329,0.1259411745713252,-0.08997646205283769,0.3718753931297557,-0.06303228272572638,-0.06741319143404931,0.057136491443689465,0.07803354616719087,-0.3014155283522845,-0.008015976743396764,0.00428540205540189,-0.030436292323397854,0.012008905745405793,-0.1427539456583103,0.06782675172767624,-0.0754440367811995,-0.0381669704717021,0.029484748783499968,-0.013891151658281359,-0.09861340130817345,0.08746785764976331,-0.02110628853204671,0.07236130310022966,0.020470667644585886,-0.03668713698462347,-0.0980889442867811,-0.04710084627218468,-0.06678941458232121,-0.185855589296084,-0.0540343222625565,0.0388491237411788,0.09495984151618543,-0.05867722778676965,0.05143051667991857,-0.05433965416541573,-0.027473826456760216,-0.02141188806251866,-0.2937150757430819,-0.06670628434155511,0.0533174071304026,-0.01807645484525496,0.11997108754051299,0.16112466102410172,0.09810943527157592,0.005462591337745322,0.10524688494181772,-0.11956646259326108,0.028921028637115452,-0.1398274425429485,-0.2824588433087348,0.027942532913901936,0.161636408903278,0.01982713903191668,0.10163455178722221,0.12393261480979349,0.02523606072641845,0.08245698722176388,0.03437632250264497,-0.026791184342710293,0.15558439632607546,0.06842556509735204,0.3407868129740221,-0.09975194821638343,-0.1066586568182425,-0.15376724248729434,0.18757683968555278,-0.02238621513318508,0.11109128478285915,0.3348246253126057,-0.0441167117873553,-0.17154651654084527,-0.12047669208025218,-0.022743423835093677,-0.20126365298649926,0.14550249814096922,-0.0912056024960856,0.308442532645985,-0.05134477818203856,-0.2846285176123741,-0.25330994195401485,-0.10422185254514559,-0.08519720835624335,-0.09392605740418199,-0.2339165995388405,0.07827968088687298,0.039919632391564264,-0.059027976892664855,0.23584259564687726,-0.09837406491697506,-0.02169687636244005,0.09773418769797608,0.5979708447480298,0.006632795697307739,0.116784107788083,-0.18542338737641478,0.5852127297757428,0.015672923500650762,0.05461298471455635,0.4806661010260791,-0.16432657711213025,0.24581724918137163,0.15595173456591685,-0.045684609887026544,0.15709405042593516,-0.02729808828980386,-0.06600376248938049,0.028620516103040856,-0.11566952535156919,0.05953427958625031,-0.062907904098932,-0.12194296560301222,0.07077029625211965,0.04521270526146997,0.19913683138968494,-0.3046811388197762,-0.28366126795338326,0.4666297298273885,-0.13788406315382995,0.5964992780945969,-0.18365479212590885,0.10881659593258848,-0.030347239945954296,-0.09922081615298985,0.009898105184027837,0.02160044615838134,0.024588061074580066,-0.053721002688797564,-0.09360321674223077,-0.10641702420290196,-0.09479678715659981,-0.09001860422130004,-0.0869758800420781,-0.14964947391094388,0.1560828416510571,0.21034174405554174,0.10690429700540796,-0.08313814677827419,0.1684111906407334,-0.07320332159341104,0.04490448821982133,-0.07014593430142313,-0.11713406775489778,0.050203769899072374,-0.16951046632893815,-0.020399897918867226,-0.08347979718578652,0.11300187065487392,0.05129784460134231,0.19728006650094432,0.08539223190178889,0.01845088776983674,0.13601587364708814,-0.06485239122923805,0.09743024366467425,0.07159495187859964,0.23841605416904466,-0.1674146807635721,-0.18714990354867314,0.06841822426307735,0.12198644383476327,-0.2037676955620871,0.023376474572657933,0.12241430892442846,0.049147905085571236,0.06213358989610787,0.1419911782990294,0.18751447827638704,0.0016363977728917464,-0.06108479530587622,-0.002815210731306598,-0.01540774728834382,-0.18612286464344738,0.036222733740755124,-0.14673401974215763,-0.0812508108193306,0.09229070423260899,-0.015309499885568423,-0.20886695395753324,-0.03958645477974299,0.09304155470059619,0.09520995497135198,-0.15643169344416158,-0.0015496900635583572,0.08990185940719338,-0.18229642435971113,-0.08865172417032811,0.0494358430290856,0.1013318466085522,-0.04738666914728117,-0.07591886610412461,0.048113815500041145,0.03087529070431867,0.15172323633779566,0.13940072756269783,-0.21096332797853856,-0.007673502288512226,-0.018043608319955235,-0.02494470977111894,-0.03479972217690008,-0.05754056498536389,-0.004346774115252526,-0.012944736843710839,-0.08337630639304551,0.013600912520021987,0.2408654957520802,-0.09776597380473266,-0.20574678283527795,-0.20578655786509537,-0.17603947303728618,-0.1160926713101855,-0.08554172976691224,-0.014741086901118442,-0.1642755650283067,0.015582154309282127,-0.020360720715130932,-0.2194390234141743,0.013950156891844461,-0.1311945792146928,0.08247860143894252,0.0808453900190484,0.43837794786799195,-0.04281039629028499,0.14906673009507293,-0.008666253132790624,-0.1318326724709425,-0.13333551645658728,0.03130835860235933,-0.1466405910328834,0.11214390039382786,0.09832888822793491,-0.0641139557139919,0.24605171713754057,-0.052344232637723774,0.018913999998178038,-0.1821668368596073,-0.01982865916438703,0.01122321319874935,-0.15041319122863048,-0.1282199961575043,0.1080181228581669,-0.14922472244540733,0.41010359201503727,0.03422860911798256,0.2257347705122756,0.12924970823524515,0.048259252066941954,0.07611433623791454,-0.00566530010166442,0.08485341132841391,-0.021243690067966232,-0.024807452080894712,0.13678976916090024,-0.039805106325874265,-0.1509723271523156,0.2714209473799843,-0.006539130001590904,0.30778222952753853,0.025815374929480283,0.011409193499601516,0.218751108540971,-0.1126259386881734,-0.08063097882687326,-0.0026499414669208356,-0.06927199097018097,0.11777827322770115,0.03658693278313215,0.146982666893143,-0.18048829182103782,-0.01581308038959226,-0.12139219108165143,-0.008564088693860305,0.1228286403323622,-0.010192887162241818,-0.06970523412345257,-0.006241083559708979,0.017956331813035336,0.015210165904640562,-0.1379756078748668,-0.1548814262164895,-0.09801862653694737,0.04361546884046089,-0.1318900757601124,0.10609234156380665,-0.03025917657950651,0.016685435479924723,-0.1559131450043047,-0.2800023171555165,-0.13343637607463252,0.003960040061244462,0.07753469277198832,-0.13775482424881416,0.25976272095810315,-0.22086963035992882,-0.025417525379840548,-0.045350807598583634,-0.1289268625954827,-0.18428064751755818,0.03430407431527626,-0.020199775392730084,0.06270679373348487,-0.10916382135568825,0.11204015448108057,-0.07803749305829995,0.1736769523786419,-0.05905645290745327,-0.1177872256657664,0.024735471182101086,0.05781335921704652,-0.01577014944914067,-0.07806425853012945,-0.022667367331803565,0.5194805332828806,0.05418967251452936,-0.012981599930145707,0.02779475531234442,-0.06174829478549692,0.06756572511969158,-0.07864757674650022,-0.05965749515484426,-0.254937328699955,-0.2105364639533877,0.008433274964356745,-0.15543103784266307,-0.16607592769278925,-0.053139731934643435,0.09778107539043229,-0.03719147419130155,0.08096390831976141,-0.17485370173472117,0.02085324913192899,0.07962523033443092,-0.10106343146558325,0.047112490053565104,0.020857436384230564,0.09265540931764464,-0.30598717035945866,-0.06639808315571549,-0.19730088615272856,-0.08443671117460781,0.05010823815062581,-0.30298477490331555,-0.16747066147272227,-0.2889653304959452,0.1895382982199812,0.19592512562943062,0.09486494749841784,0.007960477495705484,-0.02191116227078724,0.08578302520198378,-0.02238301694347711,-0.10769702398828518,-0.07451773510274178,-0.026718888597394307,-0.13769266860997723,-0.05367604138250483,0.026041617225102114,-0.025486670600382972,0.02906201453742678,-0.058160044018155325,-0.22686127775429984,0.0029148506127477025,0.08554978719172128,-0.21756783109259242,0.40789411007718035,0.08005216916443919,0.02859136918031526,-0.023093249954288114,0.23614643236460162,-0.09584093021660436,-0.264092950561556,0.07627203323988505,-0.30443988407993455,0.04057969657282315,-0.18410258430197918,0.024338866878139687,0.11498281964003437,-0.13143103311073437,0.009021266580944041,-0.0707093478611717,0.1005795566337825,0.10685667335027034,0.10191722667097451,-0.052698759748429004,0.035528390699073785,-0.043188216745905325,0.009441222240872047,-0.05880522710903138,-0.14158745780727283,-0.2140915929302938,0.02882132295021044,0.12629443929215767,-0.07655007456803671,0.18031771403422736,-0.23195825123781003,0.18565837936332427,0.03498920084794227,0.012719820843950276,-0.06210319839239974,0.12205916951442072,0.061610968754054,0.14780912986894695,-0.1278601463903968,-0.09294083615352264,-0.1003538573850119,-0.020280387318232212,0.5362029204804878,0.17238134045270298,-0.15322531167735404,-0.04379522561429716,-0.059425563122318024,-0.02861339410398713,-0.030535855274499975,0.1563168865404504,0.08240139760571748,0.09152960926042261,0.24113032328180334,-0.15067153693019888,-0.003690920719010058,0.42459601889452625,-0.07878511014275297,-0.13674534873080763,0.2699744486885996,-0.059689799223498645,-0.11841664850318047,0.05475765078967518,0.132484003865068,-0.03984757739134705,-0.3030199337515157,0.22283883842552685,-0.13773978393192282,-0.03670869281418723,-0.18517213614207437,-0.03708098620975599,0.08994500864173072,0.09746821278744038,0.12755994050034905,-0.12197298088328155,-0.16509544399975792,0.09752124197479807,-0.01430534590446103,0.20945129323799824,0.061320815264144624,0.025509720505011754,-0.13099068693502597,-0.15731856272408373,-0.012726836923634329,0.16537809816556093,0.09199253102760654,-0.0325906163168034,0.08350832829866858,-0.03956669955513235,0.11043288453726711,-0.17598343266651334,-0.04900727640156041,0.12073098100889866,0.05139056698878914,-0.23881391963979068,0.1695683697189125,-0.04196358110236034,0.08837953539338483,-0.05183722837026331,-0.012255563186169357,0.004495551303545124,-0.19392367629205434,-0.10613567066581926,-0.1471469442863026,-0.08379724692840718,-0.10791344004740608,0.14860797844255705,-0.03870036175348177,-0.07465151326535956,-0.024356108945942735,-0.08474747893711095,-0.024589927304646788,-0.050365292713317106,-0.16786357879754113,-0.0885729718673965,-0.235633075017747,-0.10387190945723913,-0.17854057032285722,0.004636900930519301,-0.24860901081351522,-0.1085237190559106,0.006133292734421826,-0.1416635088702618,-0.26298769320024573,-0.08017444386473203,-0.016809827068511793,-0.1785605972877889,-0.05687798254092663,0.11206805594195002,-0.11456970510838901,0.06799037023968363,0.10848341356826849,0.04082660974328672,-0.08700193886360023,0.020068638344811332,-0.02264354661871387,0.3298006221462182,-0.15194600133441197,0.1988616479488296,0.002742305903576918,0.027041725968232462,0.04146551156767084,0.02669740752299569,0.2925860588232477,-0.18846761613919255,-0.009563121937780215,-0.11703205339161574,-0.01406797655521261,-0.11647377317644536,0.08245877092526786,0.31602092877768884,-0.019333254324417418,-0.04006994176057453,0.05926127730717714,-0.10827226392646186,-0.006854321646198956,-0.00028366640067301535,-0.1846814008239136,0.03007506564789041,-0.10316345677152441,0.16281375067491477,-0.06715910874870579,-0.007788152832939774,-0.009371825936294214,0.08967674282181637,0.13116461589243142,0.08781071179175157,0.14808201679310487,0.595808876116826,-0.11140231554726257,0.09414942747081105,-0.07632018590369209,0.29624216542225845,0.002502831156789244,0.030627265604122877,7.053221545500642e-05,-0.1317384437643565,-0.048000103773010894,-0.14860552456753284,-0.012179819266309986,0.1415516115506631,0.04414425785606694,-0.06463650948492602,0.10003669481746576,-0.14159842263721578,0.18210533566736295,0.08035980027528647,0.0001978443198120784,0.07864036584373993,-0.0005246847981384338,0.033805838785936654,0.026015506074280367,0.013808426983239117,-0.2605508222892006,0.0652780759744475,0.15281385402903602,-0.13147135302119997,0.04150624093575937,0.11032924531129983,0.15345974481792454,-0.1325010100832816,0.027218611021173532,-0.16657390356359136,0.09191704648489432,0.042424911452564454,-0.1965806689452807,0.026505923196545163,-0.17743375622480848,0.455504033994182,0.0918827259121322,-0.04971419744154707,0.1422067553346002,-0.07284616397543188,-0.050954559851557266,0.1349518337151427,-0.008694996077189324,0.28849268184725596,-0.09370865262282121,0.011962085492206695,-0.08323288021197221,-0.15925976818189863,-0.14524683240287206,-0.03588417999209101,0.14186777539071044,-0.04971554830348605,0.13621695641009424,-0.1286064855717071,-0.20742320096867536,-0.06972326553616573,-0.09929670369965535,0.0832485801359335,0.11278844045784578,-0.14698215801696127,0.0403828760651015,-0.02570468043474678,0.37266787261090456,-0.01847023007360819,0.02593526699996701,-0.07259843721569634,0.033670683369763234,0.007384100033410292,0.01726810010625552,0.005299052651636311,-0.07119735583068504,-0.06055996961882731,0.12069521106456961,-0.3052988634073302,0.15125858604730183,-0.24651845279283485,-0.08685705775554754,-0.16727993523671963,-0.09979078373307908,-0.1472954402959458,-0.013998912643773736,0.013209813616789634,-0.03109804747155276,0.0905783615353692,-0.005963978790419472,-0.04958973363227497,0.22277220180515875,0.035443526181614615,-0.07735310366352963,0.09944523403378094,-0.16560969208028245,0.0444745347179009,-0.03300928626463936,-0.0253600867673464,0.04408212492459962,-0.035471353496262556,-0.016579894863032876,0.042163369366720646,-0.033535620994061724,-0.14644626419040624,-0.30281475311890643,0.0831560564137392,0.034185183985014064,0.04668816854983383,-0.12538027065486132,-0.0002567897901716094,-0.06080559916341965,-0.19185893382086788,0.046989595281617745,0.048140488984388906,0.12875839559040012,0.0640756822519154,0.013017830853784329,-0.04875481781878227,0.08008034734414461,-0.023397772690668017,0.10202266753305717,-0.1477432608137574,-0.060683535642335903,0.014212062188047159,-0.08794770152850061,-0.06103736647109647,0.2376110758675371,0.08232717999842104,0.1424121557978529,0.0071254585731043195,0.22532288622274768,0.15332739815463287,-0.12071007208887806,-0.024646235713211055,-0.07062053493273737,-0.005859932949134483,0.061632688032165,0.04301585917507404,0.035786054126706074,0.07054625724715066,-0.04097116271878173,-0.2384566376517171,-0.0042902756273374915,0.050150756073663434,-0.062000906273502236,-0.021251136975120634,0.05796576555611686,-0.10797714784382387,0.07226188710602421,-0.2507170660453011,-0.04218514247293731,0.08623818725240641,0.4524434066061329,-0.1101380739289968,0.0794379876157343,-0.1909242113869402,0.17557249489491814,0.027840773733818262,-0.04165263444554065,-0.011671368217004093,-0.046873311102480664,-0.16109318158370134,-0.0725740137736219,-0.14380084650596536,0.02554048926410888,-0.07634547862154704,-0.05620452059176105,-0.11627135483610028,-0.2658028665745285,0.05663262632400546,-0.12134865664889392,-0.11823157739756489,0.026197393232903116,-0.1060770198007852,-0.13011118795048734,-0.1256194767281083,0.033443033256509526,0.06709353946413894,0.005608455857735334,0.20670933730492408,0.09104845229292519,0.12459456712537104,-0.2498054283842536,0.399533937787781,0.00903215348291409,0.11598731288739911,0.04020690920077627,0.022017422329142,0.0980625031444152,-0.11115727204561493,-0.11596826502959293,-0.04816231017285798,-0.025338819193885028,0.08879999382846693,0.04317313229712389,-0.08962154506961424,-0.09819847293981371,-0.11995758152855424,-0.12151095528910787,0.265242449608313,-0.011300573254185747,-0.04072823938713752,-0.20899004066859275,-0.2761573770185905,-0.055292910147291874,0.0559180004383453,-0.14913805988773174,-0.14374869095352086,-0.04172804336694181,-0.016938750433779546,0.08604402674058619,0.03092661727733031,-0.06250665125784619,-0.0478737626180841,0.0007022591511666498,-0.02034499023225489,-0.07759106974970242,0.011514483327468242,-0.02674294002207442,-0.016902578993058682,0.07771399692601234,-0.1654526316376318,0.1957867571331275,-0.1648777270979138,0.11788612798485605,-0.05049341524625354,-0.0015905638238544553,-0.02799362219515692,-0.029626946470405185,-0.04106634918573156,-0.17088233953842838,-0.06500886350919294,-0.14792977089624038,-0.12246540019902938,-0.16623264230890658,-0.20921451427385002,-0.3054399733540715,0.5400488320835027,0.02487451069426975,-0.0554131202523473,0.5934975856542098,-0.09138177888841603,-0.018627156332065704,0.4717692690713288,0.01640876897824378,-0.05150673122392241,-0.11314638211786338,-0.03362505133693943,-0.0005067557398579724,0.23012346315111487,-0.03875366568780837,0.0009737420284136951,0.017202027104491844,0.05628159206408237,-0.14079629797202578,-0.0016082766412530186,-0.0522453261624733,-0.19565904262777745,-0.057847101682197824,0.003399297733781679,-0.09811171172379743,-3.2009372072392995e-05,0.07709163564214427,0.0629393561008308,-0.11741679563431273,0.040879657755907,0.0129471684051166,-0.20798723991453952,-0.17741386714061447,-0.041833623119910145,0.3348246253126057,-0.19945691603410215,0.006540545160086231,-0.01615515599993267,0.4018928191418865,-0.18931905537233526,0.07532936301513125,-0.12424248634702238,-0.04912123134692239,0.15684184340741358,0.0056902102045597565,-0.12066811100316524,-0.018005684515906128,-0.2805450968967267,-0.025227856154846354,-0.11767735405095923,0.0340901744510505,-0.2126298856153535,0.12404458659872196,-0.09775792634012812,-0.033718218450795735,0.5954776656550709,-0.020171677298135206,-0.12039110605178237,-0.052046584512132094,-0.1687032279495122,-0.015961183372834697,0.18148011311861867,0.1643986063236503,-0.24295562887703487,-0.06061734125661875,-0.17964993432550122,0.22893553209163076,0.03007009355665529,0.3430292702820821,-0.09837952946656434,-0.1652309611713316,-0.0803878206077337,0.032258689901635225,-0.075234435567708,-0.29479448849882184,-0.21040960824152846,0.06559200082134656,0.07564338272589526,-0.060906965963385504,0.5651806539400798,0.07578621945585831,-0.06465007196323992,0.020356992543584213,0.5622226344486944,0.08048491060604719,0.01005780396313914,-0.08057558808028688,-0.16778823355483574,0.059720015978284455,-0.07684781523124869,-0.019863418393495428,-0.21249757579368597,0.15033250420624722,0.12598444912513576,0.10557930502848276,0.32385023953381714,-0.14248459666623395,0.35219595474887233,-0.08399512922667343,-0.021662950735313057,-0.14826561438141364,-0.09918003155591007,-0.12402390971569509,-0.2990799575434307,0.07010354071715627,-0.05525492065609352,0.2349170664321835,0.15884992113612273,0.03460999618377437,-0.1362118743551182,0.02892963215526903,0.06767553255580834,0.08115918092304686,0.0198857619000142,-0.05244505182219621,0.35931795505144093,0.17912947660557577,0.05630554447741782,-0.05156088490846879,0.11813327785005058,-0.16298643981233185,-0.3020001658193644,-0.061220188832215576,0.08527737327706382,-0.12448998542819474,0.116733902903575,0.23338824403122943,0.10485359175646757,-0.1487440962204837,0.006640150031228936,-0.13456179079521557,-0.020500014318987302,0.02877596424207227,-0.1182475131799758,-0.08447034639837622,0.015095153144261412,-0.14133403968024458,-0.021425006111210242,-0.10155296638535213,-0.09555539122021693,-0.14975575052309853,-0.14416404204553063,0.17831248417693701,-0.18876351218408116,-0.08129168029227693,0.11481305464598399,-0.27386030936616096,0.07733900793742893,-0.07850453444412163,-0.05284153058892681,0.0014971397347734834,-0.2307399743679269,0.07240013728990333,0.1155091816089795,-0.06271653449727467,0.04758662900005261,-0.07350572338799995,-0.012701343028666559,-0.2360336851241312,0.07389120267458435,-0.05051123239831162,0.09859028913297965,0.19992754962676718,-0.025522036861451165,0.2653086798918719,0.08409766448452514,0.5685061686343728,-0.05954796363076032,0.333733773995877,0.033726670480707314,0.1361456999894189,-0.12308606232907182,0.030310448413012257,0.17326612561842544,-0.14681455213780223,-0.10012921165031184,0.20490496473573486,-0.020336803780048186,0.218088251780846,-0.05606967105257257,0.1331045414743471,-0.007787146768440627,0.304092959211398,0.022645967838280143,-0.1239125366886401,-0.250851737902638,-0.044240253239463116,0.0017760105506987168,-0.011270243302451343,-0.11474617128931436,0.08344624684008276,0.07974131949164082,0.013619529132729418,0.19691595939754392,-0.09614131515769213,0.35038054794473444,-0.045018235534896645,-0.11957746710217936,-0.24474472532524294,0.0731108579856068,-0.3054244972275962,-0.1726412325040076,-0.018987501877626226,0.2151562448914278,0.04148186258293753,0.2752971278334625,-0.045397360780859694,-0.20311142599164256,-0.011828761387648375,-0.0988602276043196,0.2444508428877473,-0.05222571327233943,-0.03199719682854656,-0.1631523502469782,-0.09223765462936347,-0.050873974399970705,-0.21074865679272842,-0.034362916010687576,0.04793271156734898,0.04866425506877286,-0.07366433733392602,0.0014925470776130105,0.15356963116458688,-0.08232764117392666,-0.13506756243612456,-0.08700731818590655,-0.2078222393876968,-0.044948214451789086,-0.012706625877576776,0.21417757655162492,0.01892458022809286,-0.013153007756123691,-0.009149760742638591,-0.20602122417374363,-0.18876412547545995,-0.09447265062222554,-0.14673401974215763,-0.2401360320157295,-0.11204463857505584,-0.02991188855566553,0.5875086741073259,-0.10635723361458639,-0.03085353632701307,-0.13642719897553132,0.26483556072013115,-0.06779639949673963,-0.03816837023409593,0.14023942255774913,-0.046744053797116,-0.03800806291152976,-0.1882097985557059,0.5851080340967426,-0.1451363354799333,-0.2866437991570441,-0.06516772718009306,0.51331937625668,-0.06970531741696166,-0.11929507880852687,-0.07624097981769558,-0.24101491638335143,0.08808228819692042,-0.16786433020737304,0.23222892457621097,0.5484124841848869,-0.06196072032226128,0.007676666568501218,-0.029084215896526787,0.06449517350166517,0.04750347859318247,-0.05637128379127283,0.03162038205746092,-0.0063066630144942865,0.02477760411364933,0.5980933707148732,0.006034553376586139,-0.09858038157750722,-0.02174924799305745,0.05533915412709194,0.035679207409895,0.05777432171706332,0.0961509803490056,-0.028211427963914303,-0.03753953133799041,-0.27983812333965097,-0.3031422830149386,-0.12100438462634071,-0.2578702054834743,-0.06743967010059311,-0.09819519501874882],"z":[-0.116712082157897,-0.02464459971565084,-0.12171611712110447,-0.0032173640307780805,-0.060586832154951795,-0.16585565232360455,-0.1660196023076421,-0.0912751545107816,-0.10322328619773394,-0.041739941701116924,-0.2820457723716708,-0.07686553410502754,-0.09162122306862451,-0.03272094445808627,0.15863351275628768,-0.05126994612916069,-0.018685452699604185,-0.14058288621047496,0.05721144479594384,-0.18300784436521614,-0.0834376513576414,-0.09321994006215063,-0.11285050086465373,0.04276221717983637,0.10470694840961557,-0.16485447818884383,0.1415239896845851,-0.10272315219784041,-0.14361961711700302,-0.09215817726591956,-0.1940646199841989,0.1056269713702201,-0.05071836660350245,-0.03292323313095505,0.03672960434257947,-0.0481864903471353,-0.034179088666377935,-0.054916914893824165,-0.19902617440598216,-0.0021314557347058005,0.052098030578195294,-0.06330111792184173,-0.014969063047519243,-0.19426537622552728,-0.14778352744148032,-0.061071352770713516,-0.08170713303028528,-0.06065841453940457,-0.07690425451435205,-0.040464810516601606,-0.11308244641653915,-0.019129656846437602,-0.0396667830466852,-0.1159915764112164,-0.13584867940160264,-0.05207843746599722,-0.13363445863042586,-0.0529412001954077,-0.16847549656916486,-0.06840331687243302,-0.11494257498615766,-0.011325959443848933,-0.09354986838658413,0.016960426610992573,-0.2699725343631263,-0.016701162886303373,-0.08839173755486046,-0.0544250977544333,-0.04881270313431902,-0.1036166746691743,-0.17407251078327543,-0.05179340612447245,-0.07065768137417042,-0.07162217550021027,-0.02167501441691587,-0.09770313739079976,0.013256622181646324,-0.11218683441107133,-0.18031266969649404,-0.16536534229510808,-0.10592893085759628,-0.003350722378329079,-0.20844118283393118,0.14532366601454544,-0.07598133001378664,-0.09666621935594705,-0.11752097462629463,0.0066496881225220185,-0.1966933483781872,-0.11831625314556678,-0.023558423330807425,-0.015581052543558476,-0.0198711792568301,-0.07481522340420985,0.08258580376568635,0.26603241855055987,-0.19047278529985703,-0.03227287645662977,-0.021523662598903703,-0.07393190617306263,-0.18729317474367185,-0.11053671957737411,0.04664517841744746,0.3028072447002344,-0.10601685241973992,0.03402243208236069,-0.1124199495342873,-0.17711017540221585,0.3902328094046069,-0.17017405929796087,-0.047056411013515716,-0.12214266937582573,-0.12538704425824843,-0.11402438951420361,0.04759302350262675,-0.05117522661783257,-0.23164888763901734,-0.19245286512737647,-0.11149491861106461,-0.049501547266739124,-0.25584744458254544,-0.0791541182213962,-0.12096536261166595,-0.032853775511355515,0.06393381889114001,0.6133160113399955,0.027747511137640016,-0.09815977097633483,-0.13037971692676975,0.0524351814007059,-0.07383737099136783,-0.14795658191608235,-0.03265516420936516,-0.050613936339291635,-0.10418084141401807,-0.09268870162492043,-0.11360384556360797,0.026045869749093113,-0.06751940048919436,-0.1458411028567966,0.00869178305322355,-0.16110757360540018,-0.06934483660421806,-0.07254895560693246,-0.12575286081781065,-0.0880165705668258,-0.015784969451799215,-0.104378319360351,-0.1959255633746115,-0.058239395118797575,-0.041014719556595886,-0.04533295363749274,-0.004804297659088941,0.04885432972771401,0.007569468991403104,-0.036982861245678525,-0.05900107529560914,-0.07535028859789146,-0.13584236670624275,-0.2789538423530234,-0.14905869505091582,-0.140978954294585,-0.17360368218814126,0.2322866226167951,-0.0049141776275344745,0.19771780452765975,-0.07226805982083727,-0.17178906422107293,-0.07195255121790028,-0.13520856395610834,-0.20465379615541526,0.06685611864127082,0.22814467982468772,-0.060687555842243744,0.08866854238770228,-0.0765945994105714,-0.1262401763602031,-0.10945129501817866,-0.12539130094526546,-0.12775996726499853,0.008747369878567012,-0.11705427656830326,-0.11615511969610876,0.18845320710533547,-0.12146543207579626,-0.03995883998064084,-0.07214716953209578,-0.06800194690481319,-0.1578067314484653,0.009325056143854196,-0.32290074222847587,-0.14036227874080473,-0.12973139556168042,-0.03779917649761639,-0.05025865581709028,0.012153894390641823,-0.02264101105035957,-0.13730718812976744,-0.013372860669258682,-0.051739274329492255,-0.1311110622026469,-0.1066794238687731,-0.031775372058137075,0.007633553968768636,-0.0019798445803023007,-0.007233636467722689,-0.053513003169397895,-0.05367007759277892,0.1516012698887809,-0.06138665714927889,-0.04592852645129554,0.02953260601508661,-0.052288589166196525,-0.07799375108234172,-0.027322169756281645,0.13733846321724458,0.13281410176816696,-0.09105133549737546,-0.10605749497989535,-0.058423603793866526,-0.07816148847761152,0.02020215128360845,-0.06973067660164704,-0.061186760236179576,0.1732289451290735,-0.14538049940525977,-0.15796881143151717,-0.08707900083935774,-0.16078589849665367,-0.06425819856306898,-0.040033657424188844,-0.08800587027803927,-0.02859803325950757,-0.15691046642453044,0.04515416630121391,-0.06251079315557959,0.00665560581916853,-0.1258047969703141,-0.046897649493599286,0.33239012081045605,-0.10999115989269169,0.02779611002736035,-0.0211737167026437,-0.00624587487394884,0.05480106114932547,-0.06852231020382543,-0.0627762057322077,-0.0724133877957179,-0.14640113994931192,-0.17963441865234042,0.038626846724452386,0.09486048300079217,-0.1132709736813068,-0.05012538330225172,-0.020179401245848896,-0.1901160610405119,-0.21180964015394793,0.1922531155360308,-0.1848796631787968,0.08569251933878236,-0.07538973462296455,-0.06276183027283191,0.03531564903186795,-0.1163920199001998,-0.1835700617702708,-0.10616553538446764,-0.09075119742349717,-0.10384289519461473,0.014308940201997246,-0.084317694213783,0.028567073238924268,0.009025400335360104,-0.13490359670976998,-0.09682624129423881,-0.12184106129724667,-0.020469335664427848,-0.11122539499760108,-0.14401082218071934,0.004686140927548655,-0.07440798313914372,-0.05707370903974892,-0.07640402090273024,-0.09476756673439682,-0.03428689720632023,-0.0924578310347767,-0.12002949288826308,-0.1946129213962813,-0.0923585946042401,-0.13905405132834137,0.01053039092433438,-0.02145379626754826,-0.12776828388639722,-0.19424468699698966,-0.023417165576406544,-0.17262518935416366,-0.14427659212172436,0.06271690693312569,0.028208613942485472,-0.07584109873365075,-0.0685204016408246,-0.05050312391738916,0.02923161349041445,0.0025595741693345752,0.06565133827349218,-0.03937152250682173,0.3829627197384849,-0.1473274852114324,-0.2784449336673489,-0.04574492366853086,0.09420830847399544,-0.16627517793775967,0.1603172648006451,0.03902797121455238,-0.10107186784125632,-0.18238692642324517,-0.09699939063409581,-0.10547205193404889,-0.11660670560845893,-0.11598379871059834,-0.06440580999765137,0.0001274431681839162,-0.0794478676292623,-0.09704030610370058,-0.060175155491742745,-0.11582648307192223,-0.1641121703623308,0.09723768752557398,-0.08122948425381081,-0.12756346336714464,0.06304440710654756,-0.09786100131256553,-0.07628682998188022,-0.10729129177349604,-0.02552129482939729,0.11247177713495242,-0.03947415575232283,-0.1952276888630434,-0.10963484160538786,-0.04141483277518457,-0.1399816954168591,-0.0038161118509544988,0.018153512467533498,0.26651570868808105,-0.1836271780732827,-0.026615340789274716,0.1976145935399271,-0.13800429811317466,0.5242353727326171,-0.02636788646992974,-0.006620433868150645,-0.17270436671566045,-0.13132559541562716,-0.13878970775655897,-0.09906939135226,-0.07947556117442274,-0.11975724149460759,-0.09686663983151102,-0.10766732352083291,0.27962994067020785,-0.06837725789532037,-0.1831934311196676,-0.0989463547878497,-0.15991121264430375,-0.10750307380510775,-0.006491461199362114,-0.18123493415223427,-0.13455770680246024,-0.0019148583171129793,0.001181849915749664,-0.09166083296078746,-0.04971009609950186,-0.06630092539918901,-0.09304629537932442,0.002161114416766022,0.03091905730845001,-0.17896461278159587,-0.09351886128162885,0.002419750639016704,-0.09762258854819693,-0.04257920465661582,-0.2069951606307474,-0.05579782713774948,-0.14081484185327287,0.29356992574165064,-0.005381148949750017,-0.008681830381562377,-0.10777074426247124,0.2175231200349662,-0.05534670985303303,0.6070521210954051,0.10251527123363564,-0.08927722156593676,-0.18260596818147717,-0.11202341218932665,0.018575242425353414,-0.0916545392726563,-0.08976252958764629,-0.00496924600572077,0.17215509710685598,0.2411798730147239,0.05436693917412372,0.07526351702256318,-0.05746716076651415,-0.05485899691405024,-0.0977077938874647,0.052934010762868536,-0.14584647809925858,-0.02332185735733916,0.05252100533202,-0.024029881370801337,0.012222033973837927,0.05942314709345121,0.009855635077005606,-0.11669065916777797,0.1495702964844193,0.023179163089619065,0.1714563107963929,-0.14299721151203595,0.11798319274644334,0.4384560463052195,0.25092283495940176,-0.001981444449864854,-0.10063877175642194,-0.062280575085805964,-0.03387066686186104,0.02431676338127612,0.23693576475663192,0.21212871442797335,-0.08392821719181276,-0.06786643935621058,-0.14894332831370574,0.5261439383782638,0.07653049712325788,-0.04211698288771568,-0.06738200118660237,0.08187337541603605,-0.19812055889859145,-0.10582150735755048,0.23734277032896975,-0.12089135314932864,-0.1426203150396282,-0.13088719226364054,0.031081568005540457,-0.07513449304572054,-0.013621339160819326,-0.12422290462385335,0.2308501780466828,-0.13650830775457642,-0.20481134657725591,-0.2315355012090397,-0.004598884822124069,-0.10407200945518263,0.019361835566083955,0.6111082057876137,0.01908062558575705,-0.056342657595801905,-0.08599255829081422,0.07990078142744354,-0.05467779100507134,-0.15216122515923716,-0.07284513423926617,-0.11586628177117297,0.3382684688729483,-0.08568043710094576,-0.08967178966305506,-0.04305933983852904,0.024280373225811523,-0.10602437658679434,-0.03546759794449203,0.05054057310635749,-0.2630082849151006,0.08624280105218887,0.03927064715351336,-0.1583582673712315,0.013565752682728726,-0.07244588596855406,-0.07278680001905859,0.03744914738919931,-0.04402897532829561,0.034295183796198135,-0.003005473018847558,0.06505662473863291,-0.07026437545911371,0.004227904367962666,-0.07520748037033138,-0.08760936107866416,-0.1112240877907516,0.14598365849373285,-0.17151032496619634,-0.1814764373083887,-0.027403791237955683,0.00697309958155846,0.10155979078133157,0.4249856492973427,-0.06600877561393192,0.20702933207224677,-0.25133249992735185,-0.06165828868291388,-0.004690010186263741,-0.04941406837609451,0.014239387085599558,-0.035774063825770884,0.2239766863172551,0.19377864568458056,-0.09780806375049952,0.10010478672474428,-0.13774293846908806,-0.03387302634609725,-0.2308096562067788,-0.0007177155140473683,-0.07254156069901659,-0.03862259610558089,-0.12509764676877025,0.020187307369477117,0.0429174394687839,0.01423632539622304,0.014391088375053259,0.19007466766099507,-0.07438865135082667,0.1722079622201492,-0.08397294185073091,-0.09639838276076827,-0.07589200050572342,-0.04488744268473985,0.0007918389111077025,-0.26079105975553357,-0.031852780457848937,-0.015298808898499428,0.07553515533881007,0.07520474309560454,0.226146675355843,0.021868886385330882,-0.2772393786182099,-0.04550746438362025,0.010138694681169406,-0.04885818665618913,-0.15153070999280047,-0.07135372689303794,0.030284094804896307,-0.15169076660192554,-0.08201083060241218,-0.02071379885825257,0.14573326210929904,-0.06925906952276231,-0.004254476666951511,0.026397489110904743,-0.07177087756436544,-0.01568961598371981,-0.05652677526831012,0.2014358944815194,0.027004923324897737,0.07842010521964593,-0.20248527996127996,0.08928656038251816,0.24201058813866913,0.3124152179215476,0.04740752536798004,-0.054484983010571905,0.027437824701198214,-0.04328644070986875,-0.09575645872848448,0.08471863988715682,-0.08911205044860701,-0.029224827014162326,-0.13654427425481228,-0.059470856570149035,0.06494835280330225,0.503374858380774,0.20731038677699948,-0.0384968569807769,0.1276182048947887,-0.016674491612000863,0.26581853556994994,-0.04262856712413819,-0.029039044602545017,0.03435508102034101,0.23436791533625173,0.0749983768153034,-0.0673198209424852,0.15365083534921434,0.05428778522872639,0.009220042343609902,0.24399671399768194,0.11966198347502922,0.003935617110373228,0.39389952992303073,-0.23981406673756295,-0.18617418166929944,0.062161027319695875,0.29300896865960435,-0.06470920742751288,0.07182748308728618,0.16993708855513967,-0.0668667804685644,0.21116492611866064,-0.005557606958634519,0.026699101459202873,0.048785717734110834,0.04352784137267935,0.07739205609499183,0.004161369578353379,-0.0012712727548628672,-0.01892419063239694,-0.14623805739390194,-0.09302662325913166,-0.014774807647223396,0.07399252113611557,0.018147835021431448,0.22685857410687893,-0.07071501130755585,-0.08890549878850541,-0.07806125336471617,0.01669327350284541,-0.023450114638332374,-0.044594868857324076,0.15673532347048913,-0.20631537487483442,0.08509990478442434,-0.18218776128986588,0.6101266437115348,-0.007877224676861899,0.04829424291827077,-0.04265896690946536,-0.008972229044789247,-0.06449965486420768,-0.0425798134273524,-0.1096595013498232,0.056436967572603185,0.09214785214142424,0.29587499449700344,-0.018774834510414142,-0.03807258512891596,0.019855858603787795,0.1861470019506648,-0.015167609452623122,-0.22836313587590013,0.159485130817085,-0.05285704980314232,0.07086488271297632,0.035718334376038934,0.04887972676245764,-0.11944145261692314,-0.08329249439478853,0.14687511482017507,-0.07839824886934074,0.03478600761510545,0.6285872431404598,-0.047996790750333765,0.23489950366257129,-0.23915513366887775,0.1778088253860715,0.09042672869438255,0.015168211443619828,-0.14916885076988823,-0.12249245279895668,-0.05740666324527906,-0.03368530845185911,-0.045011381271229195,-0.06478229886236304,0.15878092099887464,0.03029924785749527,-0.05696304124993947,-0.018756500541294283,-0.01619513182346949,0.09353322419737999,-0.0054407853731874905,0.029340769053025798,0.13341213421220222,0.057699905557006693,0.332263262072678,-0.014822189759604773,-0.1192431284950231,0.1792069972452818,-0.018425865695885436,0.048425547819772506,0.09033300824184176,-0.09626080200213491,-0.021125954273451555,-0.005846593341199854,0.058588291277306935,0.12036722323036357,-0.16072253679558954,-0.07391809466244478,-0.10070595217442378,-0.07259505331145068,-0.049324737691691045,-0.09219839452649482,-0.12482507292180071,0.16809867476992751,-0.058240337837455525,0.009490034732777401,-0.056890244272127716,-0.13005552311074733,-0.03517325403336044,-0.01147826131452544,0.051515262840710965,0.041092974026887975,-0.05594798962135818,-0.1291566470021306,0.3262397799261499,0.3006953827940888,-0.0652113785383038,-0.003563249852231363,0.005848305075138309,0.003803130254194543,0.01178767904796951,-0.034195199582462255,-0.2764505308385675,0.029489143343012326,-0.15398763760171189,0.0022464281494622763,0.05562884421663593,-0.05014806337947073,0.031709284362032186,-0.12580071427301656,-0.09697166781534675,0.006843592173720432,0.06657934007690063,0.031227453433010335,-0.027452896550900854,-0.043822135981670236,-0.02024945752068937,0.07644153763742884,0.025498307381604447,0.33672921658494154,0.018773164341110784,-0.2714502220759822,0.1916133360638733,-0.061984354219928546,-0.017600949645731088,-0.1944560244530604,-0.013774818379624831,0.19755860500940223,0.05753187143300983,-0.04137610019806098,0.06968466534183597,-0.04087201214302316,-0.11607985137244728,-0.13886113187453586,0.09279138417298698,-0.18157732122054238,-0.014908578147650004,0.043813584296278345,-0.12365935631866315,-0.07808685959231508,0.055205770891748905,-0.06463246157674264,-0.27833099672611167,-0.0440136325497382,-0.05821033506304858,-0.047909348728290516,-0.13254869236907832,-0.09687146212598989,-0.020600549021898215,-0.05725791868321659,-0.06856617607074286,-0.05303501335578141,0.14876522452241348,0.010994532028275343,-0.038019970372474406,-0.11638664737512644,-0.10643479129513629,-0.11445735169504793,-0.12296883494324548,-0.10087212980121413,0.018278379688992798,-0.032492757863875336,-0.09104115327788047,-0.08406977039292977,0.003992708557789614,-0.03750659483701111,0.03621957311113801,-0.06925689994593388,-0.04470354246170998,0.08081855029766082,0.06522085709232082,-0.06646902192384822,-0.08171087740821904,-0.014438535379797078,0.28138542436597896,0.07456286789505813,-0.06828943310133023,0.00023984781643227518,0.0009115885074521587,-0.04070303412302179,-0.07011345964701234,-0.09538425407406621,0.11811483838583367,-0.03239109506223226,0.07625709961443085,-0.12337845077802025,0.08586458648212796,0.06406409307735583,-0.003410249176524425,-0.007032203381539671,-0.18749934748116434,-0.012709439342227935,0.07403686174277144,0.08571990365507892,-0.23282419952765154,0.23330834501357192,0.041381882575474066,-0.11496516591704006,0.05141070128182476,0.053203871376147356,-0.1114483181872412,-0.023631482461895992,-0.06971360011611173,0.12199847723104096,-0.08442519129893424,0.14001007429552004,-0.06855684468770255,0.02535447223189081,0.09943019107356137,-0.07624708715121781,-0.12604951199275338,-0.04087609209162711,-0.03356808004669134,0.15893901590444418,0.12461835242981546,0.0026978806347907206,0.006511901975780535,0.01190770475635013,-0.09601010705839876,-0.07328120695661805,-0.13015280069723809,-0.03710885043967304,0.05045404715587411,-0.08692268597848335,0.0018606340897667705,-0.07641306224149007,0.0643430982005319,-0.043391017428419165,-0.028380601807530777,-0.2764505308385675,0.12763952476069615,-0.04981948469371774,-0.1748381978784388,-0.011938939883393757,0.08076079448436611,0.014813230027773808,0.19621353905110445,-0.0431284008335494,-0.02488153667105975,0.050984633145804086,0.08052518804523538,-0.04015841592732948,0.0159905100373945,0.020325990834443405,-0.029727582079396943,0.05840895244983148,0.02379211765847132,-0.08699902771850118,-0.010646365146265916,-0.041108521432182056,-0.031895451966194616,0.07767991041196382,0.6159150399146459,-0.07345365049084154,-0.02095702293496916,0.2471221333947983,-0.016043933196207723,0.13166346996376294,0.07849190687027743,0.23612361116404815,-0.1719862368509937,-0.07858573051967194,-0.010647986547491762,0.07767895584164654,-0.04009131711675167,0.015133929951690836,0.06101782243505413,-0.15817154466940664,-0.021300992070672223,0.0194683436357118,0.33546630437287955,0.004106425443141874,-0.04340318961474063,0.20286479773074562,0.04200683622285117,0.0435134081453891,-0.028810505982909965,-0.12382095810905965,-0.0321687489420877,-0.04621413955979254,0.231152329562093,-0.008480490273779707,0.08279561729990632,0.011482567814218643,-0.17833630068004167,-0.118524491334795,0.03525214010795636,-0.14754280989930313,0.002358762163206781,-0.05897785324264003,-0.009371894696676899,-0.022634669967483482,0.000617076025815675,0.6078564273932665,-0.008976725757982779,0.12652129585007219,-0.14470382587226427,0.017205557814120823,-0.04511685908186898,0.6286571577459589,-0.04102797647881083,-0.05716455480090057,-0.0406553542724882,-0.042106707545815365,-0.08083631791189268,0.024017454909170014,0.11740795747414143,-0.01785854995324826,0.014777645202375456,-0.059099154469316566,-0.05658804903785414,-0.053593346203469366,-0.013045211116125069,-0.05902668133786456,0.0010708151269765039,-0.1299433761210109,-0.05013123882817134,-0.09284440512952774,-0.21550443479903425,-0.015927061385831005,0.07518957137590204,0.1321455496971662,0.001084327197513819,0.34958960346286366,-0.18780403474339302,-0.01736012174310426,-0.13409186643598336,-0.06203643783918767,-0.04687003979441692,-0.0797187690794207,0.16007554133271976,-0.02805880374141067,-0.042575789193944794,-0.13196780610554165,-0.021783506951683324,0.20548257535433914,-0.09860837888979125,-0.13002564496296165,0.2877612058377083,0.04598600136681525,-0.04486535089557952,-0.00564633395149288,0.025491440660522526,-0.012220147616069535,0.029167073568869818,-0.013066445083028016,0.030697726394043653,-0.018196121215570262,0.13545021369295163,0.015666563392329247,-0.05164982214201707,-0.1003520635993879,-0.0665895698149533,0.5560744361198213,-0.0560314926189991,0.07911983249379746,-0.04294937919336535,0.10084762538544832,-0.044714789763195865,-0.08496813585627419,-0.06370439463652458,0.05571109762147916,-0.036004948788535315,-0.03740284631855614,-0.014793359030810239,-0.0646357157271547,0.11944447588580613,-0.0336874309384325,0.11409960320189089,-0.09525625402130371,-0.05107498745527457,0.01567063012064886,0.06619686730498377,0.032615717496713804,-0.025278108185565203,-0.012367224129416009,0.05937580814482254,0.3824735462575764,0.2653076556459231,0.043241724702387185,-0.009925134180374585,-0.031886950097682186,-0.052200620547334334,0.21950493760466852,-0.11617447067585263,-0.11028094234502846,0.16410204674510023,-0.10541706781404087,-0.10701235969218362,0.2208658231682888,-0.007310153897403527,-0.021996798649423048,0.18602072614388884,-0.1047920742858292,0.003068089217903731,-0.0021473615231196556,-0.0532909327221574,-0.10839992113087671,0.02470020202170289,0.49000927846648096,-0.023657480390435113,0.628195889908619,0.048008015668870474,-0.13984222514361483,-0.05134539251421256,-0.05084782425173815,0.028222625383597405,0.03590951436177229,-0.013137885034686062,0.4466508886780507,0.13851368076238726,-0.04954714424542971,0.05514196874219227,-0.024201686267664026,0.3752475843010055,-0.03922868471923043,-0.05707838757222589,-0.058963662579482226,0.06887725882636922,0.1314220608852863,-0.022068920243125238,-0.05685864480657154,-0.10220689967113973,0.39237965819882864,0.07205920763391654,-0.1446085492477033,-0.062067823077550016,0.05875274445431885,-0.05657369878768774,-0.06930995960154193,0.34816133635566443,0.16668836337823484,0.10501853174727124,0.20377544781991758,0.04941532387874905,0.015676205731465696,0.1329907739275789,0.6289540719517336,-0.12272878898589222,-0.1816695260199478,-0.00023317869917484447,-0.019679077463775757,0.09997888696202259,-0.058445739481773194,0.06193724905342143,-0.13264079465714607,-0.11780705519178973,-0.18158415274517758,-0.11943076082148889,0.026021386948391543,-0.19469140333317608,-0.00875374483112647,-0.10606367201396595,-0.15127690620896378,-0.08803538333160559,-0.09124940465871775,-0.12062342922409576,-0.09839437594904574,0.101887134701048,0.04400974667454198,0.06365214438404143,0.019630570045335492,0.08697518195358775,-0.12127306088616358,-0.024430904557207124,-0.03062590827156984,-0.05578513915411344,-0.054220331542920046,-0.07346143662022232,-0.08878299164231526,0.006186298280231039,0.10516873061523863,-0.03237398153433045,-0.13871503607732957,-0.07043037400867698,0.05068613134920709,0.12567932730675874,0.18786268038925072,-0.08482453317844196,-0.023722883525358705,-0.09482372056303276,-0.0520323894962072,0.14584619246822308,0.1059700137764948,-0.03277591447814236,0.07450111014124641,-0.042206741643503846,0.03194196037573376,0.02765048124445496,0.12804393625822993,0.04493215274665294,-0.11394373342802074,0.0951525829130679,0.26643465083013623,0.00479624543561477,-0.08264956342125329,-0.08549584654696823,0.012127637153865292,0.22873959380688585,-0.006715368873922594,-0.07254414952104905,-0.055514537910286994,-0.11487027211664719,-0.12228416098452685,0.21343476505229397,0.24186422987158285,0.20484058379629033,-0.12822870789212368,-0.053286535171619506,0.018436166129990685,-0.11448303352678058,-0.10101513715283834,0.1630047508820502,-0.11284425744115029,0.05519525369691286,-0.08391052857311442,-0.00541875303248676,0.06800070243912247,-0.1441431331942382,-0.05450905068795432,0.2558434930265724,0.0359896849906814,-0.016875130551654318,-0.10229457591614464,-0.07182091247328617,0.03696811525259569,-0.04933364819673848,-0.011809749214585592,-0.026209963892050762,-0.11001429990317764,0.21318656599548289,-0.06061596182214999,-0.14494567559498037,-0.12833333799023391,0.04367468784191027,0.3268423765824639,-0.10475474839077153,-0.08992649232776435,-0.11203087707174336,-0.054980609385567356,-0.20848435557348946,0.005664453433355299,-0.01221876900242349,0.02367587187129794,-0.07612866401148244,0.026066658978652443,-0.10406807381035248,0.0030299473301215714,-0.11799643701182776,-0.14860079609325522,0.08081666829725025,-0.07451963121655412,-0.1158751704220352,0.013194502703070224,-0.07735474862216311,0.16199302078185945,-0.15296992181990074,0.015397943990146209,0.09660573251556318,0.03459528974803968,-0.10184051717476186,-0.03264722473734441,-0.08224870901095643,-0.014136817748470091,0.16174968381392196,0.007898196221055474,0.050581541533787355,0.11188697014986707,-0.14312132262055136,0.06355969768453473,0.024411408587488238,-0.012749779163833134,-0.06212211568345868,-0.057633135930716624,0.003600361390641289,-0.062016480190255276,0.034347657534795975,-0.0920656336013811,0.16146826519341687,0.032799288421653604,0.18687710992635115,0.03451394914401959,-0.052191284257946675,0.29194068956670355,-0.10448267544200686,-0.08927212845939678,0.03173410696047198,0.08432119973175359,-0.05380159744922928,-0.05677532678553482,0.1023562520095968,0.064828945324803,-0.07974618563603586,0.1012573104300695,-0.10240339435310815,-0.08900181977906442,-0.058352723560980056,-0.1596439206004318,0.013528007535776489,0.0002649258276656234,0.005498722202528482,0.1323011055240743,-0.04379467934523653,0.047902439457563985,0.10108279309773326,-0.017265425097862722,-0.026771874637235234,-0.04320682754021064,0.045897269879122134,-0.09370086223889397,0.06755430227027416,-0.051778044106299465,-0.06057299844072064,-0.05671327352835086,-0.048848093087018644,0.016042732378787276,-0.01897960543104689,0.15622497014724415,-0.008161808914734723,-0.11460650874746446,-0.05178212605288726,-0.034755338556766005,-0.01084118072367119,-0.14953050696194334,-0.14481738168413694,-0.0679462564618917,-0.0677654225541442,-0.07851035019671178,0.06875787948869566,-0.0646363159670514,-0.02863198907224847,-0.042789631565160385,0.0864529142421274,-0.08383225458891566,-0.12669969805300194,-0.07086164249699112,0.07654009203495996,-0.09286441869558504,0.03107723278905522,0.22243655138906884,0.21722319551992247,0.2832950533207724,-0.05526441374595003,-0.06467486070090668,0.03310234280021054,0.30942273945574095,0.2056386290788719,0.2118518128562908,0.007237917012529625,-0.17094911014802905,0.1130663647453328,-0.04257636173595059,-0.018494948025301383,-0.019396011137831775,0.1885089224627992,0.007132735319203462,0.026873437475600115,0.07955219059209033,-0.04502005351546106,0.06957141231340248,-0.041756150834460506,0.05211559812047549,-0.00983558626854412,0.059809730404942925,0.3666572906891206,-0.039004062978329576,0.2130073046018781,0.03303174071398043,0.08830892155260943,0.12022434120216266,-0.04858296793787748,-0.04108152552173492,0.06283386958881787,0.13600276934190222,-0.006244899228297794,-0.10262858469151116,-0.0122359073460556,-0.09478867404198567,-0.05861192121943565,-0.02706868888722805,0.009468558815967922,-0.15211878790975367,-0.03294957151243917,-0.21029274254682281,0.0464003320911308,0.07494000124164346,-0.0409675616985623,-0.029048777700587093,-0.2011192309350263,-0.0962797054005959,-0.09691263922718109,-0.25071762115403357,-0.19816717112623553,0.08169784551981214,-0.0159787081985919,-0.01801754616504828,0.17637568183228072,-0.09837686918646484,-0.11498606857614481,0.1204391246454295,-0.0796624916021913,-0.0988094398288852,0.019212307300339224,-0.03844087305450115,-0.03538465618202955,-0.02709551675253315,-0.17658208139101314,0.07714236832201371,-0.05892382847557608,-0.18817173595029646,-0.011136090959835447,0.18734345884403142,0.6061945195372785,-0.06821255684704733,0.4593772581130951,-0.1834420392766967,0.0031875550451721107,0.23287932369250483,0.0011951743122378751,-0.12709383199431942,0.48700424045296115,-0.1560244815770284,0.18116189726161183,9.084056414555695e-05,0.10805815891151915,-0.003282848854351645,0.28097555533897767,-0.10535637640726858,0.22900848412775293,-0.23525989860911742,-0.07900696127773893,-0.215895756457982,0.033877290376693424,0.01735696060310294,-0.05318553677361025,0.05912713955108281,0.060581185708383986,0.07557985343331367,0.06099143775133147,0.14163277257896803,-0.18568614186240104,-0.09919807926261542,0.028606820296449564,-0.00895490780938326,-0.12333978829827257,-0.07206580378647424,0.06868186308974931,-0.09324525853819958,0.13849284469185072,-0.24000709457499408,-0.05135336109816522,-0.08570006442136126,0.18605593480817958,-0.007882812641750532,0.27195322699959734,0.14615185831767138,-0.147217295903501,-0.03246211266053503,0.25541481927589504,-0.11051987411020038,-0.04718915487628908,0.22066170123693382,0.14989553151291946,-0.02957382395824377,0.1309236618911598,-0.037841142463237325,-0.04527918612678615,-0.005132641794686516,-0.0110553994499354,-0.0890025754983207,-0.05671473977969128,-0.008224626733052525,-0.08672305401248641,-0.11310661779078429,-0.02442064177870286,0.04947269953093893,-0.008256890758974116,-0.11092799843207067,0.4913563803417052,0.013819861861337289,0.22848627268967814,-0.005697441882697334,0.04290462556902577,-0.14131292645974403,-0.04987312015286034,-0.030591380120558634,-0.0923878976267109,-0.09468419521647643,0.0063301942850698505,-0.19818262816010881,-0.11426476096630472,-0.08826987590460349,0.08739776214509765,-0.14083288475206115,0.19344939919148002,0.05311355900212289,-0.08404596639294891,-0.12448088114877473,0.3092604060843811,-0.029525124913357677,0.1766429356715417,0.22883521645101115,0.005979127250182499,-0.05766177974499014,-0.06726822346713424,0.0603242130610009,-0.08388465476216876,-0.12357016104442468,-0.10544847700962875,0.02115080046317311,-0.07077570008239341,-0.18437861930989993,0.3493538138960994,-0.08748592110169898,-0.028511400567451165,0.2603600364268855,-0.006574830739569648,-0.08652390604883618,-0.024635876596350165,-0.02458920811481557,-0.08822712946757032,0.09808248827758234,-0.05540229705517242,0.19826815382054674,-0.10194314313168931,-0.05636687942993527,-0.03033411760655054,-0.08230860791770678,0.12002694698655969,0.03256014347180535,0.08315763084103714,-0.0007205352874388118,0.05948267376492891,0.013255308574966587,0.11556805004991323,-0.08236271896066147,-0.027788717727438766,-0.15238634244178634,0.010918239837520436,0.004679896127807871,-0.15006407106929212,0.27690015421403363,0.1927569834456424,-0.023867355476805,-0.022824080388899798,-0.05482229172999527,0.058895821653248366,-0.03564825633300962,-0.09183228094520424,0.021201687244639007,-0.022949443543203364,0.09939560908699978,0.16529532769798239,-0.08882444365335505,-0.08183739897894937,-0.1680388090677704,-0.050644026023781284,-0.016337557467864326,0.2094902790934375,-0.06723333809562647,-0.14105088848004121,-0.08119350701264982,-0.10315126972639446,-0.0485419154308163,0.0108175447228033,-0.0892788361535003,0.007080418383137452,-0.0713704308496293,-0.20926256224158282,0.012816194563839629,0.020695600948155073,-0.14468652323034775,-0.09835940223973874,-0.03596869260551418,0.03577552200363489,0.07035603798621119,-0.024180161184591823,0.20504825659225234,-0.026909625237042427,-0.11335787440461106,-0.05821153796171213,-0.1266855106937832,0.11183524849517153,-0.028651514155316832,-0.10890898247493455,0.05396503344603539,-0.08632461749981905,0.1776435171299941,0.017593435838965815,-0.02008954558825871,0.09988368419745539,-0.1916580906451495,-0.0779035420174526,0.08706383258617109,0.09308399118347815,0.15240587361793762,0.09780140548384526,-0.045029027939328954,0.07224508706909825,-0.09053026544495402,-0.08143074481421095,-0.0681826998909689,-0.09106733317026558,0.03873555353617818,0.00024959219772160865,0.0004387046133864725,-0.07364663997270776,-0.2698806850295333,-0.06343603128419058,-0.10009744946904013,-0.09608705947647937,0.015466552417666795,0.00254749510591059,0.11645517637711236,0.026841193687083674,0.13926774834075933,-0.04550945512205072,0.1583389156886279,0.013787624235683889,-0.10053757538257906,0.026161519586113395,-0.18666515796282881,0.41330267240265045,-0.057064047189604956,-0.016568637850909167,-0.11526325550714477,0.026257643491035355,0.058537917834727504,-0.01139375528005588,-0.04795084572843505,-0.12323612811941552,0.14662401283931825,-0.021492162221006444,0.03518200332972488,0.08012881771213895,-0.023666227063902995,-0.0241645987014906,-0.12679472177875925,-0.004661563731059477,0.07251453858903165,0.24345019859219752,0.22910152468611392,0.09919348783255778,0.06750220472482168,-0.10494346767981605,0.06659078730661984,-0.05846782564805007,0.02382235074464607,-0.03572760239044542,-0.0634130182048034,-0.1515653108625323,-0.04641436084877314,-0.04586147262587107,-0.07456846917576442,-0.015810261614976964,0.04670447795677077,-0.04433899337124242,-0.08455581853251494,-0.0786680081077714,-0.20152093188652742,0.08566317868299778,0.027535334509840807,0.00672256885308452,-0.09026346570791564,0.00799127810485531,0.001981231104287874,-0.025546041180340012,0.15968251891945573,-0.014334517247837917,0.1849592225704121,0.03771357530116565,0.044114368028459444,-0.030539320095862646,0.0044271342140614,-0.029700766561622214,0.06537519529124142,0.05019809158824136,0.09367688454287304,0.1662187508394457,-0.002669678702483586,0.051109132818867,0.06138701442400261,-0.12406446877120712,-0.04970990766014264,0.04913752611689623,-0.031055828795914165,-0.12414673060912554,0.03201241976753246,0.4948718013368494,0.05252078899957756,0.08333073917024594,0.0006897698961798895,0.04810093136931143,-0.14230481355789384,0.06200168773840006,-0.04294837402931636,0.00379937652000206,-0.19742315908880315,0.38965164264518254,0.0641091414318349,0.0522111922515216,-0.048394641754457124,-0.0899357517321751,-0.10173494513497877,-0.05369268653097849,-0.1154694643806227,-0.08517388252237994,-0.02840199793919837,-0.08148807551432188,-0.02920695311831458,-0.13021162924447066,0.31271341002939795,0.0023798510336291282,-0.06804618418549649,-0.10214257021263277,-0.20466928178104232,-0.15716301154942025,0.3007863565369931,-0.0897424151561474,0.08478152199742792,0.029743817746876035,-0.21994786633605284,-0.06726321121558959,0.12782787036854731,0.35517279383420775,0.03908897789928739,0.10527973037301813,-0.024278572001160417,0.15960222090246717,-0.016680182015017864,0.610966790257185,0.010197445995286401,-0.07354244390285411,-0.04407818513041567,0.11196689912198864,-0.0865350482945498,-0.09961223763660244,-0.038175349191944846,0.1535973790927354,-0.11089849902720307,-0.1911979809707842,0.028929114801785488,-0.054039183227372764,-0.2813540559935014,0.23213378358149006,0.6101130701606433,-0.08336818521796163,0.07192718606199053,0.05718516972821269,-0.04196387646653886,-0.034755678348150854,0.012709443293341793,-0.08341491581144932,0.0839829842348192,-0.0902819960035144,0.1964820811200221,0.14447420099713765,-0.06920243450872017,0.09216538996358305,0.34050991578667306,-0.05420988881422018,0.14510133823799495,0.030745644025428363,-0.05816098579835276,0.053688659861582,-0.07921481657673064,0.0061350850774636026,-0.1819945604722672,-0.10692500266595409,-0.23206079513901437,0.012543573616799373,0.058402400156124926,0.02330764194436604,0.010487373784909574,-0.1289880087509596,-0.07872165951870695,-0.05826114181740767,0.2458690767200317,-0.06801085334832109,-0.16039736895994022,-0.20184983289827932,0.06573486082735416,0.001331404879298162,-0.07911439982259377,0.05319845186587062,-0.0036118942122189053,0.013391484194590587,0.00947689370664809,0.051337791011866175,-0.16605935455537948,0.04410697176015157,-0.012637939434556848,-0.11750939921225279,0.4046569835851396,0.28429893212430196,-0.057815438639362,-0.1262666603872122,-0.06374599941024085,0.038131405152964676,0.007828603195036159,0.15042706330209865,-0.0811158220892706,-0.2403614861755188,0.07021692207809212,0.17491982402999892,-0.001310622648011116,0.09190952551480377,-0.05434093118537787,-0.052363604422551724,0.04768237377147761,-0.1172104000623138,0.04315649128015895,-0.0833962791554566,-0.15008045413838586,-0.037327594431798554,0.0012777561914742567,-0.013892949128909595,-0.2561334106754451,-0.02360476386741989,0.1829690537217987,0.11301067608924699,0.03774367532423619,-0.029765361990443145,0.04547284209470115,0.019039434609540492,-0.056024690345143104,-0.029417587989692455,-0.254570567163777,-0.05515610844596238,0.04354011157334478,-0.09511879434243742,0.04462304507370502,-0.0007840625985466654,-0.0015617664122817585,0.09931526299573958,-0.08947752042044811,-0.048068227839436484,-0.135956643482956,-0.11723585529291784,0.04550813643369416,0.015338846967988538,0.17065828368151822,-0.042230153743961676,-0.09895111954138694,-0.06446658407334013,0.01885847472494966,-0.09298148243723037,0.00715207731796053,0.0981009299520764,0.07059628897724081,0.08407909916783445,0.36861947793048205,-0.18367160767582869,-0.021451020140321377,-0.08305059549560243,0.26468760047394274,-0.11864796956081071,-0.23730038644686033,0.030379514091893225,0.03739454316613274,-0.0404585852433929,0.044549565526181216,0.08251541794160991,-0.012459738901615221,-0.14663864011142017,-0.020953073790802423,-0.11004435245067914,-0.08889685032987082,-0.007139092668139828,-0.19638627903261555,-0.13891330655747744,0.12794070135405153,0.2003973327093883,-0.22358447342838908,-0.035470442933737294,0.6046201349847387,-0.03304341819076221,-0.04041950973024599,0.5615908638096604,-0.08384352597988001,0.1580145683547077,-0.1845467113383908,0.09890591807489016,0.3854200510492665,0.15784875509073182,-0.09162648052187779,-0.14784968507563562,0.14889682491621278,-0.032531732949111956,-0.20864533563539073,-0.09486493485544464,-0.06763723075792318,-0.022902286474536827,0.08054763504554997,-0.007341641416053967,0.16096216085448656,-0.03957343811222847,-0.08106226669978348,0.30639520170449425,-0.09703407110691487,0.06887584608970597,0.05994986558745651,0.0713912386716786,-0.029716422504554137,0.13649353757674335,0.012526157888757055,-0.061118423007257214,-0.026576785456711213,0.03678427170758318,-0.031740232108202204,0.02401521933927158,-0.015562628677025068,0.010304526824857902,-0.20741817419212638,-0.03639746177010568,0.42816860757674446,-0.10119203080247255,0.01909458021131431,-0.11595538313671698,0.04704628752348948,-0.13191191629564644,-0.04401314414144304,0.09121471498912154,0.15545051933251658,-0.0103963782534979,0.09652075877200383,-0.1759720883942704,0.06808371286935887,0.007401201271040979,-0.026883784140870146,-0.11953063459322068,-0.05303571498091112,0.009331021688957759,0.009351919783687518,0.002238355209611151,-0.005646508896760666,0.0029499395938253144,-0.028581239115548593,-0.06285246510348395,-0.0972629815721868,-0.04380724197539527,0.05767900863750149,-0.17403851481960317,-0.04351579447964943,-0.03765721999196412,-0.11626449294851397,-0.05123858866400337,0.002697856216206309,0.26084590622554643,-0.02273161728483818,-0.07346759562436975,-0.0035941136167107687,-0.056433661908749586,-0.01604435076898687,-0.14575055998934422,-0.14693668075381974,-0.10754626984332118,-0.027967470808558933,0.06637293582556471,-0.04617023268416841,-0.03147936570840061,0.08794357670622654,-0.04521760042215911,-0.1020757846045427,-0.005830549940931144,0.015873066429923564,0.09770004097550908,-0.05457988426900851,0.053507894516666044,0.01391265965212777,0.11750420822043935,0.08307748766376793,-0.07444717223221468,-0.04471099872934542,0.12445482988761995,-0.005310432107587698,-0.06516441343235771,0.15210136360484855,-0.03166541877057892,0.2780233669439893,-0.07948940249688684,0.16473340264765482,0.05319807108638244,-0.07229573758940491,-0.03365514825042084,-0.03240485854335568,-0.2727506097890875,-0.031454775966582235,0.597691585684734,0.027191783492653093,0.21655017476949914,-0.1822521019255813,0.06993275026828903,0.0952522459238755,0.10613703997747366,0.2126309561876508,-0.022517333289086516,0.21879247838649418,0.027220851825958157,-0.09505817164946444,-0.061135462445027974,-0.11504287277131932,0.10807874565754991,-0.0690075982779007,-0.005305532169468262,-0.08907235401886578,-0.04061409344540124,-0.020175111727289984,-0.00814789965972086,0.06828711552328466,-0.10495217704961748,0.0690664429200637,-0.037982502197983735,-0.0389516982413076,0.23496582686233064,-0.10922623638585935,-0.10435210701537616,0.028029717112572072,0.09808797158867809,-0.1292736847993485,-0.023950457066683348,0.04238431703475632,-0.04633686385685984,0.083121840760946,-0.0753380678345759,-0.01068753272108522,0.03535980647964789,0.07028890464375476,-0.11174587809154816,0.005734375024017995,-0.02395348672930209,-0.07550443817278091,0.04453583803833314,0.27924956250133476,-0.020103833344381157,-0.04859360446339811,0.3364757616881795,-0.10032363476962193,0.12057382556135664,0.25065403785269563,-0.05114358706689456,-0.033815537096848564,0.13953328572867,-0.06708813688463663,0.0037758668217070072,0.02009412798808138,-0.2123705435186295,0.0008693344346237235,-0.009529164717010664,0.01278555761138263,-0.026587930349210073,-0.019627228216595658,-0.0636955563356574,0.02803709841008381,-0.1050258923749552,-0.22458820787085154,-0.0919209395038121,0.07758154731393553,-0.071971382006828,-0.018641917391703657,0.11756431436380545,-0.06448649093056229,0.34860909437427007,0.07089089757086439,-0.14701169579988815,-0.05467928175242679,-0.07624000409533364,-0.08336442954611646,-0.11599956259587928,0.10307379486856776,-0.14336128450055327,-0.0097570737085481,0.06847266013363162,-0.06894583367145471,0.22687640169583953,-0.017347062140620692,0.06922016303759511,0.14584674447871668,-0.0046668359308111975,-0.020783655178806447,0.2164612836857155,-0.016759935628360745,0.009339625430644982,-0.2766077929128858,0.08580565624045189,-0.12141935555190422,0.042096169708107,-0.12390872954836249,0.036140166056963095,0.18581226754481825,-0.09904883518672275,-0.12198431475756467,-0.05876913983612071,-0.07716510533628751,0.07797894818540801,0.28741930068510124,-0.23252733198408565,-0.00538213200332657,0.24130902979072863,-0.09959318707006777,-0.006290266982454379,-0.14145095560924345,-0.05630222377258335,0.02592082751120834,0.2219804712391537,-0.07761000748437374,-0.10892167683790055,-0.04098327710339731,-0.19324899637765283,-0.027165266151240357,-0.060890404697635195,-0.07857238923543985,-0.13670792300249265,-0.18639875565904634,-0.11506350524308254,0.028512530353845628,0.016286163912625808,-0.009558616005511207,0.026411405775950567,0.010763287795169543,0.004344656548209919,0.0287149773068842,0.03340486210473118,-0.042983658788905144,0.23230604614396003,-0.02046048208129041,0.04936263815777132,0.10692538903922824,-0.08268033359489392,0.009616304092071298,0.010513158820151386,-0.22790713044588182,0.06693502003716806,-0.22324731703038025,-0.07110323555740447,0.004101233513115205,0.0018391127921474754,0.05663549691696875,-0.19892921936009647,0.4097601684025269,-0.017488896802236427,-0.1195291989865781,0.07828071131249204,0.07644161397591945,0.12243496535680204,-0.164903570525662,0.2434385575658922,-0.16161573712681382,0.159924449909927,0.08404910896090738,-0.15911119173036206,-0.01773535339525798,-0.07339691882617713,0.242922152020154,0.39898683479076164,-0.06653392811550125,-0.10701729692068424,-0.08571179214377635,-0.11080272714891219,0.1469324803361304,0.10901405058256922,-0.026529680074423096,0.33680720784805324,-0.12529382908362427,-0.09808262863566651,0.1689599892198443,-0.0637374855177955,-0.08022488070465586,0.19871925244473718,-0.02827826003159028,-0.026710359430812285,-0.04799372307205685,0.4414624720422615,0.044641984300465246,-0.05695598810993643,-0.006060124646172484,0.07787468474897433,-0.06429885390288936,0.12355357920105091,0.22287026853714342,0.09041963159668384,-0.1409683407511313,0.02923082899700325,-0.09550929858816014,-0.10766407930833852,0.0635230482112431,-0.04904299216708382,-0.0543709889105153,-0.12070744196901637,0.028978382349947627,-0.03915747306546582,-0.03379847162928284,-0.08802698563410001,0.026363336697862793,-0.017008761214399367,-0.005714906355990986,-0.09860957818717664,-0.13920511352926548,-0.15100140442892224,-0.06652514300673383,-0.042103844937597036,0.0012359704263862309,-0.0010455509146143134,-0.07563488652033602,-0.040112719779792354,-0.04934475745724973,0.036592585263857676,-0.08056119821623536,0.01614510997505423,-0.012181470612831494,-0.09707894500654835,0.024996786599838117,-0.04345484205625946,0.0718487293090252,0.24207162516898087,-0.21061743642658673,-0.0331362893323343,-0.13997806694193354,-0.0706460432278325,0.23371287115181677,-0.04986496534074551,0.011777252711642544,0.1537234051952243,-0.13344389092076486,-0.17739129722253003,0.17410006430086486,-0.04590386007380149,-0.2646511486056955,0.21420438184592466,-0.00032623097718832976,0.00696949139326312,-0.02669239366299092,0.19046331351230994,0.08420677670325688,-0.048042185769647125,-0.13974528341757844,-0.01671107851492968,0.1747788417333347,-0.11595903562191634,-0.03129839746063119,-0.017158350531068088,-0.2773063192021806,0.10982525216417259,0.2980706199954915,0.08368454595499811,-0.02856416789211347,-0.1704321422132905,-0.013732768295018351,-0.00489974448898434,0.08344722106356298,-0.0811249485886324,0.11387414889510801,0.2392716850442583,-0.01609318487200535,-0.03599507340809763,-0.0343017325539821,0.16374030352009233,-0.0026210011482311942,-0.05067348420758335,-0.09389718648219628,0.06507888878831311,-0.27160015844371077,0.009141113257137276,-0.01907741533032851,-0.07515629962509365,-0.22571281526457093,-0.09228089934489586,0.1667861330553612,0.077724504653888,-0.034200920487912484,-0.0876951773425768,-0.0807188094402354,-0.19660938662319377,0.13108833617250173,-0.11309328442691595,-0.06868486174826301,-0.05116996350988689,0.584222411397428,-0.11052959861410001,-0.012620578035178848,-0.13853083240286,-0.07493770092165462,-0.06780442478038524,0.090200720194915,-0.10701688108185159,-0.05604286304739251,0.042207880398070406,-0.0761263054504625,-0.13810373245762522,0.05519529030813077,-0.020504465856706514,0.34569531215199606,-0.044952819664272316,-0.03912014693697803,0.42709009163785766,-0.23918513252622667,0.02208264917066939,-0.01093188343439329,-0.14942352088655175,-0.015708028775991815,-0.05830848057251731,0.04035623647777769,-0.05171860021166635,-0.10392279063512408,0.3011145240678888,0.1559842344947,0.0491664473680141,-0.08245948736667168,-0.06688280599217994,-0.1518222853792963,0.2823306619273791,0.297552356051538,0.2040539604484118,0.038092652353170296,0.12191104623784148,-0.04648527491048945,0.10418285188797953,0.42072516209670546,0.02492300489525815,0.028508431610916457,0.005803741757787247,-0.11569183013470939,0.11424844762046703,-0.01831208580663948,-0.16283113381786685,0.35929466738052945,-0.01581275645181694,-0.023959091024016433,-0.05592316619392952,0.0031768902707174237,0.16275425025123794,-0.039853154531898916,0.016373097875258476,-0.15997652771554105,-0.06837546186426875,0.051896918812637576,0.0940360331381389,-0.010621135944212559,0.0773074807161623,0.6111082057876137,-0.08391337703097755,0.006239190390598002,0.07272496133210975,-0.11691733147935375,-0.02219136556684865,-0.039981062498775845,0.041092310620847644,-0.09795887401065975,0.054505000987293925,-0.10784399152208976,-0.024148429848112062,-0.05108862519588918,-0.008178146755718794,0.07544053104922731,-0.06815095245504832,-0.07188879243077012,-0.042105847925631615,-0.07490568162617449,0.07625597049449791,-0.08913114325399585,0.006636293481953058,0.0675866747466415,-0.08507757151105061,0.041538521907828176,-0.13915293874891774,-0.027721091244541224,0.006380934831032644,-0.16624822843235898,0.03779288159778816,-0.010102813669825884,0.20259143920747982,-0.06379633550962087,-0.0272770178228395,-0.09033559745672319,0.03232955802115943,0.0699080750543381,-0.016551370321046584,0.06711238421809454,0.1373867676389139,-0.047104931587506844,-0.1312537770993481,0.05830638552946057,-0.11067097501203665,0.04174574026963894,0.004720541929402051,0.1721306995908684,-0.12450151589555124,-0.26544903754424853,0.10965914352501067,-0.07125619096803731,0.08747510189520943,-0.15980231443902357,0.08587072910496976,-0.04673528774624649,-0.043792509444855306,-0.07083029532080672,-0.13786370385726357,0.09575333851376276,0.3322832118586785,-0.2738204492655249,-0.016523840189252244,0.6101835374473334,-0.09190696738472222,0.0019259653958062538,-0.027539534503132963,0.06406202957952871,-0.010214112012057108,0.07172264989638127,0.02687573365692992,0.13308455962222743,-0.06815761837005684,-0.05602847553861801,-0.03608989312216563,0.2176660556226808,0.012935883603275541,-0.06127664825903282,0.09542986116534101,0.07399543664163587,-0.020328958077469362,-0.12951019219841708,0.2135750270685146,-0.045410765147771794,0.06507673657419923,0.19088509797646572,-0.010044250892154969,0.059915372521284184,-0.07587865694361023,-0.11646268601223317,-0.24678655934585425,0.06100376069070002,0.11762066048553552,-0.08245925535598699,-0.18607563947077094,-0.14593081935677232,0.009758937278552178,0.26881622961753515,0.07998543578978572,0.023889047819220098,0.0383550308149936,-0.059165069512553584,-0.12768220744563072,-0.1536103637575242,0.08569981999141843,-0.2462941391847617,-0.11143791449109516,-0.016054477697459015,-0.08904039784227859,-0.2722180561163723,-0.014424627631377487,0.0018633202861674187,0.5226343236205658,0.021604501774092816,0.06810417802369471,0.0032722651400430057,0.07493654853253888,0.054774179041761545,0.08998291595883387,-0.040147768374399644,0.28694159224385574,-0.03405808318202646,0.0736523210146739,0.027602814464828224,0.22365445852616253,0.022472754254782526,-0.17483128457835187,-0.0737376395001208,-0.0651182044867019,0.2866586215051004,-0.04011269148772048,-0.01322780797787487,0.050147452350336415,0.0917071668955761,-0.22909428104058982,-0.027091859333641968,0.07664773808533352,0.09002245356357196,-0.046742129355652216,-0.012747538672038547,-0.016033314967495084,-0.04053510545523791,0.07342935281587659,0.02198747285468786,0.12151908585666375,-0.09837011654137305,0.3495810006829508,-0.06651665883283085,-0.030266299914162666,-0.08903149677121898,-0.043364814721726874,-0.057217051393691334,-0.021560735082817507,-0.008126400994701086,0.6094923701339329,-0.18457673479315254,0.06346008175155399,0.22522164398306657,-0.12176521085212348,0.030570176162529276,0.022813022466201503,0.0827108370094498,0.09153197546160671,0.49073423767998087,-0.08013967175996424,-0.03952153287809099,0.10786475825622494,-0.09960216329413536,0.03121902410423781,-0.15233787045003322,-0.006244899228297794,-0.1282664872014127,0.099171690329011,-0.05433205078897877,0.09044898236064276,-0.07676079383270754,-0.10475630423424301,-0.011691005775440175,-0.05010567980550381,-0.09949959460645591,0.04969984657798078,-0.03934673688875413,0.020254539778916288,-0.04357639671996536,-0.05466341867714186,0.14119740714971635,0.11888858593989203,-0.0861058595467887,-0.09079389001667301,-0.0012113963524444387,-0.1592269619231333,-0.04386553331878519,0.015993838469846706,0.048270610015478006,0.08117891532101627,0.30734194606485743,0.034293061681940545,0.24725270890145384,0.10761796485271502,-0.057643288362456885,0.11793445809015154,0.1018032945756503,-0.1362436352624389,-0.033990886358573114,0.10822233508046969,0.007652065120181228,-0.10250114863789221,-0.0616223280318582,-0.008836004272269809,-0.07760077783856538,0.008153523741930475,0.15376724929795477,-0.023501353246270822,-0.006230441243197185,-0.13604220951387358,-0.16649315583498073,0.03933195068728594,-0.09954174335713069,0.21482565411945317,-0.10723651454536529,-0.024943265426907717,0.09834961339501952,0.17828753152389434,-0.10930180012898638,0.21636666811342256,0.29858978768821903,0.020664500772501985,-0.06955517808924955,-0.053887262866352065,0.14867103544774757,0.01196478685610482,-0.010680396048534282,-0.07934033361963773,0.2688305498582745,-0.1712784658880732,-0.014139892758700506,0.06622067054568996,0.03491926406024917,-0.19490993928345926,0.4394350992135084,-0.0910112757196838,0.030269155713853045,-0.04388904792587556,-0.07481061575731908,-0.1789830275142951,-0.11930243992032583,-0.24834268710953392,0.08643830458155508,0.09753821227927162,0.1422950352622746,-0.08574962301452586,0.09392177197713943,0.025101463321701323,-0.09851695235328765,-0.02008444660351568,0.08498776712133395,-0.015875579944514006,0.09236336149483247,-0.05078167204439298,0.21699272088042637,0.03142770896061437,-0.11887278167042332,0.0065108987270085334,0.06859334508628508,-0.24585707131161977,0.09881530589856628,0.08283414738715973,-0.08374115012114809,-0.10887737010028159,-0.013240695627027026,-0.020163979469214784,-0.014463664717617966,-0.07188001085437373,-0.01866344887142275,0.18351396609718473,-0.1149619451262333,-0.05994089475453231,0.12644139588858674,-0.09549733659813617,-0.0501920835097702,-0.07126978721210804,-0.08030366821597562,-0.04848803382352718,0.09128318811686162,0.26277514829666276,0.05753187143300983,-0.1749149695648419,-0.052864226103969646,-0.02256296169162328,0.057386511931126705,0.12730260817107356,0.0328912399256493,-0.06625229765314201,0.034144501151943084,-0.09788851778413366,-0.10008964859669701,0.07233545087921228,-0.13715176860146774,-0.07926172774836869,0.006308066783356168,-0.1000416749220997,-0.07435813079536284,0.12498210782776836,-0.04966367031909646,0.011626450512298736,0.0906917656371373,0.13975486943531384,0.0482797732410783,0.44935421462832703,-0.09572963917078062,-0.014104052684966962,0.11585080054036925,-0.051123278401474695,0.0042751549414492334,-0.037211830822361874,0.015737206326312324,0.32963081966937946,-0.0415558608231389,0.05350300931552112,-0.03479605157199243,0.18763004659825686,0.025701545578400084,0.0162474860509224,0.006990696693347341,-0.06426819654253982,-0.002970918405485342,-0.016039000918939495,0.04559397751006119,-0.17288846065335237,-0.07628423077238618,-0.1075710849075553,0.08739200667534261,-0.20222432004834803,0.006680309808313713,0.06131603443491904,-0.05209185194750097,-0.21258728278606911,0.06607062864298309,-0.09900087610416568,0.0211978134873694,-0.13358248059027156,0.02394681629845733,-0.1948959642482959,0.13974078151828362,-0.040407110642866555,-0.02910810135183244,-0.09050369367042724,0.005905808338693603,-0.047791924571092825,0.06294601889248487,-0.010433825410133594,-0.09225867977332627,0.015394100826467628,0.10472643639219985,0.1749146438162474,-0.057677716956995174,0.21221609554420887,-0.14360665252823132,-0.030909796372525677,0.04412175645405409,-0.060670050685874076,-0.21031617540150388,-0.10804697369655074,-0.10267451871482923,0.020936122693038718,0.6038921745380887,-0.005155281086635208,-0.04798422443340459,0.610268649563699,-0.027357430629495232,-0.05454819619891308,-0.10706205425314858,-0.15005458185160198,-0.10267141575992328,-0.06380954678935433,-0.04947752870799476,-0.021323220189729455,-0.09705491376487768,-0.08939448752023638,-0.008339875658639216,-0.07409325167158884,-0.017476716788793253,0.22336990137329005,-0.22499370849129088,-0.22230060923631179,-0.24275598747787183,-0.04263915191606914,-0.11067732891326416,0.04039285739747393,0.0739497071700563,0.03248983072277774,-0.09120265820913653,0.012696617295331962,-0.10374514759672919,-0.08189331122688598,0.03184990513374451,0.11679248994297572,0.257194201699841,-0.07844838922758283,0.042254751973735474,-0.035681368099550755,0.00409480286314467,-0.07652000159509997,0.0037713390725382288,0.040726954540027394,-0.2480588964668702,-0.06374484150425606,0.08694123527183581,-0.07449741570275391,0.024703930739436407,-0.021659454570258904,0.09663389748329815,-0.11681389542882935,-0.11565017045061046,-0.07062581477829122,-0.04614815683062448,-0.008817193941921885,0.013660933584461823,-0.031195543059180146,-0.05144999081016421,0.038727336926204184,0.06622908105526477,-0.12509030988389855,-0.10882168773426533,0.10830196446244407,0.04592551023921349,-0.036467979616034936,0.07441120209588811,0.11068860076296118,-0.1673625764980557,0.014703142124302074,0.031558309968800914,0.03495520513081022,0.1667195393545653,0.039345884661334686,-0.12115066625901003,0.03569120395254458,-0.05885848244154196,0.11286059451219693,-0.12171932237291701,-0.024271034077071513,-0.024642292178869183,-0.09702743717931044,-0.01782977916089479,0.13082531935402114,0.061649126175468326,0.0075423310336882946,-0.05827651690915312,0.09116456014502144,0.027442841050685476,-0.03877684320217706,-0.019286550506833457,-0.12189487330381,-0.05224458580115084,-0.04858051796884765,0.10019569923773282,-0.062124959374715245,-0.05572935810298084,-0.057315509836606104,-0.22568410197436867,-0.059994850918165504,0.6289497626528823,-0.04081167752806003,-0.09275526214353268,-0.01700487813398807,-0.11807326127154923,0.02689890023480803,0.04301363294462409,-0.06058978480797065,-0.14025965238646174,-0.04471252914072471,0.3108624380480516,-0.044371327510174,0.016364212887817416,0.15325147814280152,-0.08123017368023808,-0.1143984273383695,-0.10722313033456364,-0.22024390080703152,0.4084878189843568,-0.030963385452441195,-0.023594847089217847,0.0030076240043732227,-0.11680941822872212,-0.07473559898080397,-0.0721625983532305,-0.06968766238079753,0.20534637590213087,0.055912220781553446,-0.1453472685720303,-0.0074622820064764095,0.10581420337945632,0.012500218989864073,-0.11777540040764685,-0.14996096099051714,-0.04619141196162301,0.05012671811867746,0.02407832987555526,-0.13698108623738392,-0.1230790451273735,-0.08848971589525052,0.018379000347749242,-0.11560148931080678,-0.007718232337565249,-0.12978998358680263,-0.060123238072592504,-0.16109611787568187,0.031036423846669124,-0.0774312450329789,0.02908964435563951,0.007696912926342483,0.19958915886358958,-0.19326569403870197,-0.256474818854955,-0.08228268539135845,-0.09276187249970909,0.015853371731478105,-0.006244899228297794,0.04316220553699489,-0.005169405894184529,-0.14260900975073376,-0.13215982108353078,-0.06155502558755931,-0.030858169335006264,-0.09991173282605136,0.12620753169443324,-0.06969914798163114,0.1089941791483854,-0.21150992516073644,-0.09448750942603183,-0.0642244811018998,-0.05934366280290136,-0.04261391199765103,-0.1734770567616925,-0.08568287090179018,-0.056559713845818496,-0.11580792314503496,0.005849628363086803,0.12174400104305935,-0.1570581788427076,-0.020196278645626634,-0.12303519636742281,-0.1864500015504019,-0.10346466193903144,0.0644235842953947,0.16731278625724966,-0.059579771788982376,-0.02476173939810135,-0.020139567775739466,-0.09167410836676226,-0.09856367092847411,-0.17635136267478083,-0.14518583527636764,-0.04100906189562137,-0.038356110975361714,0.05906216080601204,-0.054209683276529905,0.19039197504845592,-0.20645035711472892,-0.16744692408038753,-0.1322447933396191,-0.035715985681139206,-0.01027155829204817,0.17913108636562342,-0.20406656600788048,-0.2690083840979022,-0.2041020819880736,-0.03125993375206758,-0.19788348382855284,-0.056692222911860095,0.003197287007050866,-0.1309710173061446,0.17229004946398277,0.014272456326874529,-0.06200736930964225,0.06226812305458973,0.04221558624849219,-0.07884457239240103,-0.08211990208880343,0.5422874296921739,-0.009519157154194201,-0.07407108594910951,-0.18347273322693627,-0.16360025194654815,-0.03935795881954114,-0.01614082473679724,-0.10691633381706699,-0.057260586479401535,-0.0005035307931994308,-0.04922022182489612,-0.06303285389294079,-0.03187091400481135,-0.03501812478013828,0.3467065457848933,-0.09521663081713198,5.087401129619574e-06,0.1662770446411693,-0.034209298074067175,0.048725439692324526,0.08958083619271488,-0.15666873415534188,-0.08167268949563225,0.013194336206354781,0.1268357799123813,-0.04193754085523175,0.061676841012194235,-0.0017656527512397453,-0.07618988246196114,-0.0764773671032155,-0.10735743475936917,0.07354649380729776,0.1714697290972728,-0.14789159848084277,0.010735391369331443,-0.009873509404509212,-0.25193106649626074,0.0260935260132115,0.04092626204796402,-0.11426875289745345,0.19347707578186812,-0.015717128004707804,-0.09093966358159827,0.17747899842469722,-0.034304332270890414,-0.22012947385895382,-0.10940475005043854,-0.16283661287892387,-0.0763271238361744,0.026927000478164112,0.2271279992129397,0.1733255591617299,-0.1074898608408902,0.07032822007585796,0.055437510021926296,-0.03262643072948941,-0.05549784688086191,-0.014665734792001771,0.6137274466389825,0.10689637207024474,-0.2612258368660559,-0.013229672364623747,0.04352440614046148,0.34268934054862166,-0.019306081829905675,-0.026193800141545195,0.10424578468522647,-0.06301053697977382,-0.003513713720406614,-0.15713538172240296,0.41806465323011965,-0.05768437618323753,-0.07122717642976525,0.036873143824391046,-0.012565982718593443,0.6280846232712503,-0.1113997546313171,-0.02682908694228836,-0.08052752796396312,-0.013755881071856556,0.04940244614746897,-0.13245086667570807,-0.14019506363867787,0.4102495616198335,0.018959956237862563,-0.041021829632820334,0.053518625511345806,-0.12228801860139038,-0.09388098454895354,0.0004089750277772054,-0.016081951789546994,-0.14531748894654187,0.2039530132110445,0.11547435495783186,0.37152541665036576,-0.041025340263046384,0.04679090057181802,-0.06936461162938344,-0.08683928460624307,-0.03218975984330296,0.06186074507889537,0.05532205924875986,-0.05110432894714175,0.23141893038230987,-0.004193344705122022,-0.09791059449800589,-0.14860633521697536,-0.0020169542357860127,-0.004087352887416559,-0.018809772191891075,0.01328060173236705,0.10829146472435297,-0.0752708231784284,0.09583990826210997,-0.17634824593527418,0.16553257419828707,0.003743401906846591,0.07606509543639198,-0.032509269875993285,-0.0796813710557612,0.021567991724389866,-0.016589058524397206,-0.06206163921335667,-0.006110165622220442,0.010851740055550898,0.03218777392550829,0.026456207240617372,-0.033502820741656024,0.12460540358288372,-0.15192577262837167,-0.04752585828551109,0.46821048881529953,0.045619817040513745,0.0069456950663661,0.04169630274511525,-0.08468177032703246,0.03637539303983204,-0.02965903765106682,0.20370088480214668,0.41926521425515173,-0.022422742199226176,-0.07560664664145368,-0.26787772131964044,-0.08926618224178808,-0.07875062355913187,0.19140119661622954,0.2145569042927648,-0.07071587735610704,-0.06476285158235218,-0.15635041088084267,-0.13810309290380915,0.04661729207277142,0.01829022584609806,-0.06181290602018538,0.14134041822971544,0.2554759081843771,-0.030674036963665777,-0.0015289642528347293,-0.05523867426045981,-0.19722331312703942,0.2118115251578802,-0.18120012349245543,-0.08815158496770857,0.08672997538262481,-0.07929500715742535,0.06783532714596854,0.147128102422482,0.6281234371145988,-0.06818082122326649,-0.01596191282786778,-0.05860616745521915,-0.08570754213625369,0.05068775944566895,-0.046373334252435665,0.01544619933320168,-0.02895580665739649,-0.06146255081119219,-0.09948137664332217,-0.12218460663056227,-0.009004298288498642,-0.032725727714080076,0.058609849551795136,0.11372733998734676,-0.15386852503504753,-0.13813982332023053,0.10814903643520651,-0.0050921211006209645,-0.11282361497911995,-0.04913754100255473,-0.006565905216979336,0.1108934582437244,0.08651643514364342,-0.03852743084170161,-0.04737785647148425,-0.12201127625570038,-0.20632517758176872,-0.008712084826152848,-0.01196535900862079,-0.06907689534704636,-0.05326328231172283,-0.016140400879868306,0.1873762647050662,0.0037585788150596884,-0.12183138246713883,0.05211364318570326,-0.07917192069340585,-0.031228263006239848,0.07856933556912549,-0.0709979731666661,0.0016803985533831434,0.04850178589463511,0.12522718923780932,0.10405081474560642,-0.10644973317039227,-0.039571406864546375,-0.03726048718690038,-0.08102494683270585,0.5967502205119796,-0.12706688256351878,-0.04513696363658462,0.14132226939333378,0.47541491149727094,-0.011098062313960625,0.39426021509480214,-0.0022852490278204973,0.02336750945814707,-0.05970802061570117,0.11384679066495354,0.13672779986455566,-0.023825270434373182,-0.012726098137764085,-0.03987384311577057,0.0660614050705384,-0.1749528938589587,-0.05839077314639104,0.01649955002505028,-0.09273306369696828,-0.030589937779262964,-0.09439424504115862,0.09811488949647063,0.16550959365529905,0.23504264260175492,0.05524394615583202,-0.10478184894687446,-0.06010777388542227,0.07110569754440514,0.08421042847473802,0.20343422338874767,-0.21705668013289628,-0.1755800476646924,0.017375728808197367,0.2595026374153887,-0.028954010352938982,0.0318314318810726,0.19330009474251125,-0.04434661911746669,0.026454023351387707,0.0961337246485449,-0.004291491536862924,0.017554339082986054,0.2809667121496901,-0.13958367566088792,0.0642278291957248,0.059724425164783435,-0.09176755265887163,-0.03545502951567517,-0.041236663615166855,-0.0206752910487316,0.09462775657664814,-0.06505593467987657,0.004539136933989302,-0.07566919733297968,0.34766128719443234,-0.13765689677305618,-0.08560435338884669,-0.05027579734641571,0.03288621281819307,0.010149395480231197,-0.0724367352106138,0.31535526956479676,-0.02412240648850935,-0.025947909071995748,0.054449568940123684,0.23576763176634977,-0.027907745440311667,0.03528045154880659,-0.029684422220343704,-0.11272015895143526,-0.08040871581555854,0.02606136631980449,0.07534684085958822,0.09835009820873125,0.19121395702296196,0.0678015038636208,0.03690945781110098,0.13091452981914017,0.023466697251393574,0.2509299810041912,-0.009341141777870968,-0.02779425752242316,0.11958732120090237,-0.07843928945216581,0.013420951191005297,0.028553741715175645,0.005749342619194028,0.2947231461683275,-0.00040242066295529224,-0.035895427752198825,0.16026587848039175,0.03042030091274967,0.033574545621413925,0.033799179521147873,-0.07315649385819249,0.03572541550676222,-0.055601588302901434,-0.04991440774034929,0.062382427542546706,-0.06911658090421223,0.4679159270574502,0.07578752359118215,-0.09853490090237013,0.07601232047354939,-0.005110060603323484,0.20647232533343576,-0.016358991699613364,0.04391401083233564,0.23320737784853426,-0.14226823507359385,0.5244132520911303,-0.0592507055859716,-0.011672873886683147,0.2759213341600065,-0.1150764325819338,-0.06880789860303345,0.15043480198296338,0.026097557846854004,0.18285367703070787,-0.10307503476556114,-0.02740309799788661,-0.049047606149807334,0.08867411514661969,0.03216691144358833,0.07893105999757562,-0.024623304783621198,0.14337671734605031,0.6108140178785946,-0.029941592715048922,-0.021780452828736634,0.16804826000244064,0.2444238159796729,-0.12706822217463823,-0.060667344861087304,-0.07242435417875126,0.16686451354603304,0.03757059257510805,-0.06942993305462664,0.06790083913542981,-0.06552657554727298,0.08633466651169781,0.15148108579181505,-0.08662507261881365,-0.08967325840071445,-0.056618202191768975,0.2543911719261526,-0.1587881593703935,-0.27438183665681454,-0.04762803628978099,0.018792188156429773,-0.033237420785605244,0.0175733806973439,0.046767143636456524,-0.09973059762215974,-0.06907795687435932,0.3519852319459079,-0.07105995497287731,0.06416759624410921,-0.00238666069390306,0.04974551899835495,-0.11042844214413917,-0.0058482289927266635,-0.07844863122425104,-0.043808780135901705,0.004895604624194344,-0.21265367889335934,-0.17923340705185425,-0.003978887824951173,0.5998841277456999,-0.133458130516922,-0.014605620383888683,-3.748176469454693e-05,0.25645841437116684,-0.08702632471822566,0.02642069747574581,0.11957202509354783,-0.17032941851585712,-0.061164345934361285,-0.13602836988127964,0.20572203098679406,0.07558567257180199,-0.06102517253983002,-0.056757155835705966,0.3450361439360251,0.07180649974096874,0.14838130052851284,-0.1477646807562229,-0.14270444152532888,-0.10194636801176372,-0.004067534539008261,-0.09996143560780754,-0.04625322058951992,-0.07096909853087235,0.028259160608498986,-0.07691522011217736,0.04900655145061299,-0.07727587398524029,-0.06848743085554074,0.07996147158512033,0.030435759491491802,-0.014092254954711189,0.05428531568484005,-0.05292320055204795,-0.04457144220462251,-0.16558721278515004,-0.06702493509365126,-0.09379721780789382,0.03268815504395919,0.12624777280716698,-0.05561429459513036,-0.07583191252843643,-0.07305681237106404,-0.10438670343461605,0.12163135443910332,-0.045803640160675056,0.15672739404244387,0.07197015145520443,-0.1112758709209145,-0.08613345104709559,-0.10189947322921743,0.15957585350814926,0.027674574614934642,0.022812206691465678,0.05413805843814342,-0.12008412009936821,-0.0030095267578318058,-0.17453945834709864,-0.010263399953282617,0.12116752513650594,0.00044168862663384966,-0.05795438377730334,-0.07569864563456186,-0.1305185834833511,-0.06439252061328968,0.23041050307964528,0.053132790590435956,-0.1068715830983731,0.11088456131665594,-0.028034353288468844,0.029743817746876035,0.16396091191452458,0.19764266009227804,0.02004673214006642,-0.07544749363348495,0.21881332326399822,-0.059490080139597265,-0.023093313229168414,-0.0712648777206189,-0.05744223075611782,0.09339649907798335,-0.013199622610961328,0.268678034936501,0.23465411772752431,0.12283460869234085,0.05426963261115595,-0.0030673232997055267,0.25615955087112596,0.025215566272395618,-0.04225501517943164,-0.0023194292734111067,0.02926523341487046,-0.037908389882680395,0.13747006283266738,0.09191744221232415,-0.011753743369190089,0.15239445368090176,-0.012709016407710367,-0.06410607214880766,0.08866097422792436,0.04249741062139346,0.059266654373207474,-0.06201022261596286,-0.09506645538352226,0.06556555133384867,0.10635135785431399,0.015463111710168087,-0.0033122677292226426,0.052386457652054824,-0.0033781441840002144,-0.0165426866735014,0.07253823988487534,-0.02629144109479545,0.2935621123350339,-0.033649682617073816,-0.016806242941137875,-0.02933868015628638,0.5365113446906882,0.22977049218898946,-0.06158656023497116,-0.05265564607785206,0.22940613758550374,-0.06339269816286139,0.04723125257075411,-0.026472210756836225,0.04638013949241693,-0.11596984486920056,-0.03795174784754809,-0.04994014453665558,0.045967238905403426,0.37511402333843474,-0.10978639961514522,0.09890136688762673,0.33894338717796213,-0.06237323980502913,0.01569235855286566,0.03840027616450139,-0.012342239315459205,0.1020428584353465,0.04725583202279345,0.07904123382432358,-0.11922676724948358,0.1577705194141644,-0.17673906787182778,-0.09811689027289501,-0.036008899916206656,-0.17254890127284472,0.2459914325180473,-0.06852658265648597,0.018930529804213333,0.039901037926432015,0.044791402485613196,-0.10096445954126515,-0.24123070636739374,0.07852598262444277,0.05037818974240978,0.0774626698286219,-0.15816301087354712,-0.03593619208369128,0.023789365661797013,0.249904025712171,-0.10589819422024459,0.2143489328813505,0.10342624153701219,0.21888962978849671,-0.0833962791554566,-0.0722684969026153,-0.1114286761256957,-0.11280272441193916,-0.2679994416696856,-0.11769235309728618,-0.0970719575369304,0.02271439084078576,-0.12276582526229501,-0.17736220977603462,-0.01629838161532018,-0.12822729101145258,-0.0757237768015377,-0.023989111285710876,0.024738827407750585,-0.07923327596563161,0.5062591113367161,-0.003370168145994971,0.1253129621357438,-0.01928181550209058,-0.09681345959306482,-0.06827174204753265,-0.0307259112245358,-0.10230793874719936,0.16139506274149454,-0.07313034824753774,0.19629332578425007,0.13599504174748248,0.026845765775580933,-0.0995984869067982,0.39365756156052534,0.001807835084360275,-0.03732739346074499,-0.008148280494910884,0.02910008164939618,-0.22046205433442012,0.0344627225292641,-0.006270760364236537,-0.021771340198104346,-0.05955877844374831,0.05524752188676269,-0.07866214468061825,0.002941362712763975,-0.1006887632287564,-0.21062731552479436,0.1780862702389289,0.15563891971686794,0.031012651830464696,-0.03540975558518975,-0.03784822857427998,0.017691712667569085,-0.12543169295702752,-0.08830854539987835,-0.10983408420841338,-0.06240979851214109,0.0964273262187264,0.09574912380447864,0.05474510123040221,0.10463493644603053,0.19478284095366916,0.0344532409290218,-0.24100058072508093,0.1632315825925936,0.16288491851440767,-0.0858235763136195,-0.13826302629901516,-0.06997205572365083,-0.05457482433492637,-0.08535532087728662,0.043097139810621986,-0.17307713377841275,-0.1354968392301035,-0.16616760925694202,0.16771165119965808,0.13205481839499295,-0.06702461737144673,-0.06928397529541273,-0.02234207613412695,-0.048787930521151705,0.09748261736669198,0.2655660372296307,-0.08119360469875563,-0.08146377398161887,0.034826625310513594,0.20674502174002551,0.04335428445649729,0.27427755520600294,0.011253215956892507,-0.008464879116602578,-0.10781526004819132,-0.06390919527288251,-0.09953199014711543,0.3251477249668608,-0.09815830092635455,0.012796125183116935,0.19432012291789044,-0.006743546418534927,-0.13452183825444938,0.24490184955581815,-0.028157581279228214,0.05379620865000377,0.041052378710856464,-0.11454435263228616,-0.09886261649403522,0.13215518935891024,0.07708816609051797,-0.12265153483615021,-0.02307062468433644,0.5341170128260665,-0.008004527921444389,0.03434172032830665,0.10790470984644034,-0.14711529188011443,0.07918823970900682,-0.06860075019004272,-0.03556731892591434,0.14826671211835096,-0.2683830366406636,-0.037144349375123606,0.11723489823183819,-0.1953677536661382,0.007811233193573914,0.13447227693274402,0.0751748360562943,-0.18480544261894497,0.01937373522784524,-0.06536717725500303,-0.022467688709650453,-0.22905037086354527,-0.0070351324772592585,0.1801795363903063,-0.22389076806950226,0.25142294480962013,0.2249665450252866,-0.018766101958172805,0.1763181238870138,-0.132166606291388,-0.12877211918222176,-0.1539593817795283,0.026868071508551868,0.21115896853769905,0.09933780864556808,0.1822252559320023,-0.05382223414178605,-0.048640449959459105,0.22942282936127817,-0.1703037235591326,0.014125944788252874,-0.0386018061892334,0.0712721880425036,0.05463528554301149,0.4599443937260076,-0.06811386483226542,0.018864117522232902,-0.049550999538606866,-0.03197517868709599,0.04189178100569859,-0.0951456753551457,0.1509158148791851,-0.021458123247758528,-0.07579704776034552,-0.10394762522187809,-0.1287186545998392,0.03550654740993641,-0.044288059841237154,-0.020905775218462246,-0.005033225478231815,0.08252159981505118,0.12336267431859364,-0.16157387233039283,-0.08827508196682933,0.109225240851905,-0.1904416082530575,-0.011635414558998261,-0.16701979871000186,0.15214609776444185,0.0019593136549146276,0.09033380434740511,-0.050860395435600426,-0.08791724811278755,0.15793734519340508,0.027076505753948963,-0.05638139187799753,0.011865203576803138,-0.023173903493134293,0.08180040503619522,-0.07357283957314902,0.026512578262425875,0.04969880273346284,0.08348042033375576,-0.008860728171453421,0.03240221641047285,0.03061229881582473,-0.00016112876015815294,-0.08538590085359715,-0.001629790441874818,-0.1606927915865475,0.0018486431705351438,-0.042936739245653666,0.09709584810134422,-0.07449928376895978,0.01988589733893874,-0.1294253915938227,0.09782224872042,0.6115609855994835,-0.13347323982947856,0.0356446744108496,0.1798105479522146,-0.008469438788653786,0.014611690925791199,-0.06066371911038089,-0.10847159601897456,0.10428197537749338,-0.10157388486926565,-0.049290930136233244,0.06426264694881746,-0.06732176747933018,-0.022082092241154663,0.1219358373288324,-0.09662436663335928,-0.11939698814385803,0.2706744924127649,0.15980453452746352,-0.029858158366800544,-0.054013080649541766,-0.03293564137010493,-0.1813637947965306,0.048700370246577994,-0.256474818854955,-0.006174404624279704,0.023482368074200184,0.02128331334324303,-0.10203568051450743,-0.19812799604589637,-0.0013458506268442109,0.04466664341515927,0.12498074244794831,-0.11530093513581163,0.05451585160208923,-0.08388942283406374,-0.04791603595892646,0.007445638157392281,0.17589173356112164,-0.08057184758057995,0.5697573018899446,-0.06425712031032484,-0.08585686983822717,-0.016217141993606122,0.14855311786984987,-0.0026816610433650794,0.27307711445229294,-0.004719338089625625,-0.003366094123223122,-0.09074570261256126,0.02376715349097687,-0.01941433972781419,0.030906536592104433,0.3316918516709207,0.02445335908974798,0.1228405765090379,-0.09071907462869919,-0.08218622478680576,0.5427073400461495,0.08174891095140346,-0.10230358363610934,-0.04100165884262901,-0.016311200750737515,-0.012812020216899502,-0.08889692193370571,0.08158009911285524,0.43055616946820574,-0.03452702890745488,-0.09717029972949862,-0.0736830231239959,-0.08460057248445452,0.022212589915386216,-0.05845577878576307,-0.05214688654977124,0.21650642540951656,-0.1442142674697651,0.04737650261669049,0.07524817475978286,0.0013802646684051897,-0.16272288252426928,0.03800275839779155,0.13456679370712293,-0.1486502478145614,0.3416219968601962,0.2262908221329633,-0.0023560543236434968,0.21664990826112723,-0.018342066555326393,0.23066227500598885,-0.20301452499333494,-0.0292103175366257,0.15129823142558207,-0.03364274099606386,-0.11974281970434757,-0.03484757590725598,0.009129069587284004,-0.11756097359375717,-0.07893199883617512,0.46968604183204593,-0.008818627137659959,0.11232092937262705,0.006370348870362042,0.025120294985075228,-0.08165123344427327,0.02355394969468012,-0.06304446736012817,-0.0929347241913863,0.13581653453302722,0.2395153184299212,-0.06549114473068404,0.21936362383568572,0.008032844978832736,0.048696158331723506,-0.1277034301144852,-0.14239924843118987,-0.1444812013005668,0.024540891778070995,0.05208128810895139,0.03104250928037058,-0.15250486313745443,-0.023668996515195787,-0.012370196764350128,0.14846532052772463,0.0761445727869985,-0.19213961833478388,0.02308990510295567,0.35263708622996753,-0.09427579564060075,-0.006112556250863518,-0.09217282193150618,0.08013393962643806,0.017848712990187678,-0.08211857337226387,-0.0921434663367259,-0.0009573764675325462,-0.03408426511574947,-0.0863400181512194,0.06108221821678278,-0.12233083336578766,-0.050839081604431444,0.13882124178017352,0.0025400863064846987,-0.2448104212622858,-0.07193342174230569,0.15782503617304786,-0.2597905937509908,-0.05541319555788962,-0.08354904905961667,-0.012741750928024915,0.08261594147945445,-0.0872216609480492,0.007536143144980211,-0.06053158192489193,-0.13075528282574855,-0.0926421775360016,-0.025233693019811886,0.03128237173644288,-0.08580902916883044,-0.02220705795252105,-0.10512335828964846,-0.12988687052073913,-0.04990017705269114,0.039624174322321515,-0.02775041754727103,-0.0319909760865492,0.014039834541621917,-0.08556802102923873,-0.06622017217438793,-0.023670696810257033,-0.010006203433696851,0.10193500895397195,-0.06579682614438914,-0.042481082649529295,0.018081505246107396,-0.21946248087239834,0.05195524768038816,-0.21177754109128394,-0.0776186024496866,0.08584588919017525,-0.18065590487963232,0.07033794839536449,-0.1257809037915313,-0.030171752825850613,-0.1462831482740912,-0.02116016031989116,0.08431320718230671,0.03683514437863794,-0.09453905535923392,-0.08341249711111581,-0.14526434565824806,-0.09806132867351522,-0.002833648830950637,-0.006287386047337577,-0.125094764548499,-0.12937870843319643,-0.04239951281553559,-0.007694879387574186,0.1424637133665009,0.02284096357819485,-0.03817297626661806,0.01575708815774274,0.18539903179798858,-0.041063811201444776,-0.05850960013905973,-0.09911116295615865,0.10357189880049163,-0.08293031404206719,0.06906557045591649,-0.03570154203866646,-0.004881123842765997,0.1772097443439414,-0.09773475368192809,-0.03473375375946751,0.08718030293272672,0.11477737546864757,-0.13408875486276517,0.08723944642038946,0.16165381544192473,0.06956862926072004,-0.035566397675397425,-0.029296783879582917,-0.059152128454551874,-0.26172189575096344,0.046323342765539455,0.020131206972967167,0.005605163129143881,0.0695327654760121,0.06120785460415883,-0.02188159015289917,-0.019032414028467146,-0.07541529085976988,-0.10080179621465969,-0.11924973471148538,-0.009700660867031967,0.4567924735238502,-0.10557118319740787,-0.18170892240861947,0.01563163921747964,-0.024979660673932886,0.23772829101424367,-0.05867939356283747,0.06442085927006924,-0.028450177083264996,0.052141542954173656,0.15845003177814224,-0.06982989643679945,0.056581515682799115,0.3346405591373509,-0.2211958767753819,-0.043237370114058905,0.31390008900496913,-0.002367149703843342,0.059397704690007186,-0.013376263448434206,0.07933903381598784,-0.14444408416785953,-0.02680942135379204,-0.08027911521169162,-0.03399094761671296,0.10326857567417212,0.13252463904808917,0.05318200686715391,0.14236869856968895,0.005741045124599767,-0.05722732562524164,-0.08762038292967599,-0.07730590332499836,0.03225916946485696,-0.06684155589233133,-0.0533173224377889,-0.06443558062501031,-0.11488265250977744,0.017678938426395473,-0.1489094358466414,0.02575680558765729,-0.012380036868835032,0.04028480528448156,-0.004721427032410387,-0.032059980596829504,-0.09214114057802177,-0.11391276273215042,-0.11213358598179547,0.12003088338515082,-0.08356428549682075,-0.0600272293744759,0.11075808180941973,0.021809576204813885,-0.03681836307997121,0.06148039622620238,-0.04769145022976654,0.07717797622617294,-0.07077554695370127,0.1129137447586746,0.06344504960098027,0.030838527354396194,0.019602830226293142,-0.11005953786345142,0.2236235115331209,-0.09454850831880526,-0.06785032031694702,-0.1388317336678337,-0.0270999838459453,-0.022993459803253256,0.14752120551066242,-0.13189229190403312,0.06362777115653565,-0.10072592245360101,-0.12332651496295415,-0.19730142990847357,-0.053712449363487416,-0.10003501621256478,0.05350807269348296,0.03647451705192412,-0.10179762686613449,-0.1456917649794017,-0.2829649251474566,0.09797480095327214,-0.056645469157610116,-0.04855489494671386,-0.09453027220358771,-0.09755821154971217,-0.06316062036546759,0.12347469948481345,0.16718793966424372,0.029904526984040763,-0.0844662691463008,0.19363255169327181,-0.05964890556301888,-0.02431857556713628,0.21429002513004414,-0.11234209281546993,-0.01906585508160282,0.15572309216944463,0.6278978531334458,-0.006211468411967908,0.1842957849169642,-0.009678054190221537,-0.11787974623045873,0.05601180324021951,-0.03820326202932224,-0.02077048363600741,-0.02755498780486442,-0.18905929065972138,0.16549401485805754,-0.0821154442704864,0.079069512578714,-0.07829127078757946,-0.04874569501228018,0.0024438429772001722,0.17554093712903865,0.11638636748798069,-0.0861069329301319,-0.14124428642954986,0.28122414288312453,-0.05370074031703708,0.18669401240147968,0.04709228668760648,0.10344291409935304,-0.051900829361603666,-0.03164142737450386,-0.006820930309875983,0.4111566559332844,-0.056736242563821815,-0.03201895650550357,0.018166968419816305,0.02686653880292634,-0.08043907881590454,-0.043347658642855946,0.06826753281574659,0.0026870184455148106,-0.07914546137347352,0.1312886263594675,-0.08522535418567763,0.0588647857644579,0.10270682058200914,-0.0039035016748914524,-0.016853736875747625,0.01725967303162842,0.15558114618332752,-0.22083867358974027,-0.21221544790757274,0.05939169978035177,-0.024165069837675076,0.2467558404944942,0.029743817746876035,0.12362540355983892,-0.0014393133658605246,-0.03542007521218521,0.004090506810084924,0.013139286625804998,-0.04434987694404638,0.02684939626314328,0.24588315159222202,-0.28163795932050856,-0.11415109641163437,-0.025583457452470183,-0.050869622633138274,-0.16965163815561965,0.08089470344497073,0.008561208677481495,-0.1855067921154031,-0.2354973159414904,0.05205359057939641,-0.1353746689733282,-0.07381741248271398,0.020429683613945336,-0.0782339046252,-0.040877719800068366,0.07744981808275565,-0.05580147407824005,0.11265047785528613,0.27410325445482747,-0.05022719150688908,-0.011676257822392382,-0.0949378894519846,-0.09063172168110332,0.07521452129204863,0.45035256765539783,0.05515930864648085,0.5505052888157628,0.0050253608368300805,-0.06853113114963248,-0.0640088333598509,-0.007339692191754092,0.07714550555198113,-0.03111150934209795,0.007763893967526198,0.03924941925105573,0.30962302647306295,-0.023590251601110654,0.23019871589642155,-0.027002543074490834,0.2907547630836803,-0.12604127788033334,0.15783542697011355,0.058334403204727485,-0.019494461360446837,0.005249925108929456,0.07146279875094058,0.0467487871468981,0.013848235825527088,0.261388048316055,-0.09049854527293558,-0.14377312710394843,0.15289467929698372,-0.01992922292255608,0.16209305466966392,-0.006068218035869196,0.10759448234925269,-0.10847532324895783,-0.17756251252404923,-0.08168696699902649,-0.013786465590537204,-0.2612258368660559,0.22971153550764614,-0.2649571154137385,0.1926436561491183,0.09610217334893495,0.382426471975913,-0.10917466580063408,-0.01820695185086168,-0.029425602723086794,-0.1180631497241026,-0.08696590791900513,0.07514830099077832,-0.01842787692864785,-0.15084314906310212,0.32839853066372304,-0.05274778788588676,-0.04062002584374187,0.017423604359283782,0.006699531339285917,-0.09628012139024382,0.08717678727293016,-0.06828661164000782,-0.12067291582576202,0.039744209569200765,0.39685299247310657,0.08691775281103378,0.17415749646127507,0.4208574830813173,-0.06559483959480039,0.04354989564196153,0.12200751448015375,0.10171105685688796,0.02267449889339336,0.20721999008182238,0.15257076047586013,-0.10578870191853865,0.06398818137015107,0.2481215833255527,0.08436824196802767,0.11425979065217079,-0.04292988917683418,0.29980638188371245,0.2220226947615882,0.0606581672798744,0.12755495178535342,-0.07158248166875578,-0.054376040588765044,-0.07961262454975734,-0.0414295471759399,0.1020428584353465,0.09874883997971033,0.13452873651813949,-0.1728201682600471,-0.12918725629450784,-0.09084962902702873,0.17703720015173635,-0.0666635110427937,-0.0032455168558373133,0.06636580099297253,0.07924283600266854,-0.13813881038439388,-0.13311932460127446,-0.02819103375895872,0.06661587027796456,0.26497718135201853,0.07904138319342963,0.0983595913409734,-0.010833211908212258,-0.010779268029345796,-0.00967805858497239,-0.1001092668161482,-0.0033655450728908485,0.05957646256298742,-0.008200869919065903,0.010869366666461438,-0.040727536922527036,-0.11886884079624876,-0.04502322754941937,-0.020441805614008147,0.07176766097570139,-0.0781912530877243,-0.0712689935069178,0.06306011799760704,-0.015358449520136715,0.0429285928333638,-0.09657390563408559,0.04658790209354514,0.04686992710000818,0.126795808598802,-0.1028109997293461,-0.1492239159211274,0.058850894352903416,-0.08399889880249088,0.0009483757635338184,-0.07245248388690206,-0.013167345534936802,0.11634053375383147,0.1480693123011589,-0.1114724893858973,0.13898580465502394,-0.14899313672508055,-0.0524408036185152,-0.23241189765647696,0.03265595362658139,-0.05982651230313835,0.1769676798401569,-0.03580550899575945,-0.038568939324110256,0.002078103825280767,-0.04965855079387451,0.1494182584043357,0.15633447625169647,-0.12044128419399176,-0.21023910013549124,0.05012009447902271,0.2643863329535215,-0.21217417351458706,-0.010523516688805433,-0.13422972514893133,-0.02644391467559214,-0.17876056113636113,-0.1307389400765031,-0.04026539197427574,-0.03510390789699714,-0.034366282706988105,-0.05985678211674729,0.04717939517267439,0.1905419068415525,0.3148439692255314,0.41127745300999247,-0.018106734035860154,0.1712946579583111,0.03693341448074769,0.1197165421013246,-0.1056253450426234,0.006265827460999339,-0.006244899228297794,0.14660756168667866,0.04567818762784387,0.12049738067704695,-0.011131402686288493,-0.005283738021836569,0.13694377235513597,0.0023405224929129364,-0.05807238252280656,0.04322113008744026,-0.10554856502618647,-0.044874313166843705,-0.1635388658941611,-0.15799355677367602,-0.009269609195582626,0.22983082268861585,-0.22230495993750413,-0.19689622733120613,0.3494027170936645,-0.18525261894873452,0.015120412110960685,-0.036832173701146755,-0.09610050304436546,-0.055257929421467095,0.14153434665663783,-0.0020757300465189268,-0.07791643314891973,-0.14053979810692377,0.0954212658960929,0.02156892727207203,-0.06941273096346917,-0.16941268950975416,0.021838643443647925,-0.08703108505224433,0.047886608410731175,-0.11488718374694974,-0.023666321360973297,-0.1383031789170463,-0.16594325106670657,0.041563172365115704,-0.22112353259816844,0.05334420218735489,0.16467718366215878,-0.020035314103151937,-0.13638319975076976,-0.06347394939631072,-0.03173368579700913,-0.11590143292262302,-0.12992030749984476,0.014048322912058885,-0.07648600164648824,-0.04579420390530066,0.021602496961483305,0.2859314255578155,-0.05082426244495934,-0.08853640323258943,0.055239272682558196,-0.04581819975105312,0.46747945516293593,-0.031705644102080564,-0.1462007374147126,-0.11413588418514378,-0.05710442934241477,-0.06032781599856121,-0.10367472802545213,-0.045528161671965245,0.44329639406918736,0.06238227032615762,0.04260003663042336,0.054643324905632694,0.0018476904125541772,0.039923554658622615,0.09023482394200652,0.040789118176003746,-0.057955488010925704,0.06627274362255019,-0.07990653208502707,0.03062085834907048,0.10034042820394937,-0.05230781895588887,-0.08347893893994796,-0.07917739793837006,-0.11338121623560525,0.07623920734209748,-0.06678655355557515,-0.09762905345049791,-0.21611048422366386,-0.1418915665602164,0.18260976025696118,-0.11269936919953819,0.06459993295794086,-0.07979188501189007,-0.2738204492655249,0.004014376368296173,-0.03699855003727413,-0.015900583578819048,-0.02086124888797336,0.5959201311637752,-0.013040995760902226,-0.0944016609287062,-0.04387016436495142,-0.2570568918859029,0.12189881952569208,-0.054945808880115014,-0.05153991559890343,0.11719341831843526,-0.05191985495761126,-0.029600860858636235,-0.07937050698090904,-0.23402884701995724,0.002731236701077992,-0.165767640515084,0.04766372678255841,-0.0007798615572534997,-0.013261676586197382,-0.048787930521151705,-0.15641049283340308,0.02537750785307743,-0.10992336623077278,-0.08663491546020248,-0.15249268663251503,0.3353642275719224,-0.2149385563484278,0.1095534477026551,-0.09108465237398675,-0.17767688900119552,-0.07471704459673797,-0.046867593456904416,-0.02453791213137605,0.12480208733699295,-0.09629267384280342,0.24399772102611827,-0.1424693277543589,0.25868075958907505,-0.08073923749036235,-0.12275344195179085,-0.1167248959963726,0.009585112179807,0.020190942966359687,-0.16727791692843333,-0.2295200676199232,0.1147185476171444,-0.10876983330858395,-0.019577677317207234,-0.10256950796475674,-0.020123321674487476,-0.014718402469998814,-0.027461908017896695,-0.020549436347566688,-0.06400389913018119,-0.04523299881454724,0.1843289383726694,-0.10622876770292133,0.04424668514003912,-0.15923694977037256,0.034255982633708604,-0.020480275905322462,-0.214553389911493,0.10168269925537468,-0.20322088727380644,-0.09805685879888937,-0.12306752839122775,-0.08393684942492058,-0.06371059015684316,-0.06855739490032893,0.11110513567421743,-0.029171159621607577,-0.12189607787753509]}],                        {"coloraxis":{"colorbar":{"title":{"text":"color"}},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"legend":{"tracegroupgap":0},"margin":{"t":60},"scene":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"xaxis":{"title":{"text":"x"}},"yaxis":{"title":{"text":"y"}},"zaxis":{"title":{"text":"z"}}},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('d3a29ddc-778c-4439-bd8d-f19917f79026');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
#Final sgmentation
Segmentation = pd.DataFrame()
Segmentation['CustomerID'] = IDs
Segmentation['Group'] = kmeans.labels_
Segmentation.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12347</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12348</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12375</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12405</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12428</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12582</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12587</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>12588</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12630</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12631</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12664</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12704</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>12349</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>12514</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



<a class="anchor" id="con"></a>
# 6. Conclusions


```python
customers['sum'] = sum1
customers['label'] = kmeans.labels_
customers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>QuantityCanceled</th>
      <th>TotalPrice</th>
      <th>categ_0</th>
      <th>categ_1</th>
      <th>categ_2</th>
      <th>categ_3</th>
      <th>categ_4</th>
      <th>country</th>
      <th>sum</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4310.00</td>
      <td>896.70</td>
      <td>484.32</td>
      <td>327.74</td>
      <td>1464.44</td>
      <td>1136.80</td>
      <td>18</td>
      <td>784420.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1797.24</td>
      <td>0.00</td>
      <td>683.24</td>
      <td>0.00</td>
      <td>360.00</td>
      <td>754.00</td>
      <td>26</td>
      <td>55714.44</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>455.42</td>
      <td>268.32</td>
      <td>25.50</td>
      <td>31.80</td>
      <td>129.80</td>
      <td>0.00</td>
      <td>26</td>
      <td>7742.14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1710.39</td>
      <td>354.61</td>
      <td>213.72</td>
      <td>360.40</td>
      <td>625.70</td>
      <td>155.96</td>
      <td>26</td>
      <td>92361.06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>7877.20</td>
      <td>1248.69</td>
      <td>886.99</td>
      <td>1317.95</td>
      <td>3418.88</td>
      <td>1004.69</td>
      <td>26</td>
      <td>2315896.80</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers.groupby('label').sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>QuantityCanceled</th>
      <th>TotalPrice</th>
      <th>categ_0</th>
      <th>categ_1</th>
      <th>categ_2</th>
      <th>categ_3</th>
      <th>categ_4</th>
      <th>country</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35934</td>
      <td>2535970.291</td>
      <td>1032916.21</td>
      <td>237005.831</td>
      <td>281170.70</td>
      <td>461461.650</td>
      <td>525120.62</td>
      <td>38766</td>
      <td>8.266032e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>108184</td>
      <td>3334606.942</td>
      <td>524936.29</td>
      <td>256101.711</td>
      <td>806473.92</td>
      <td>1200659.741</td>
      <td>553564.57</td>
      <td>59569</td>
      <td>2.304810e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23491</td>
      <td>2513530.941</td>
      <td>464593.33</td>
      <td>497506.691</td>
      <td>208337.22</td>
      <td>447375.700</td>
      <td>899023.24</td>
      <td>58404</td>
      <td>1.318368e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers.groupby('label').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>QuantityCanceled</th>
      <th>TotalPrice</th>
      <th>categ_0</th>
      <th>categ_1</th>
      <th>categ_2</th>
      <th>categ_3</th>
      <th>categ_4</th>
      <th>country</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33.868049</td>
      <td>2390.169926</td>
      <td>973.530829</td>
      <td>223.379671</td>
      <td>265.005372</td>
      <td>434.930867</td>
      <td>494.929896</td>
      <td>36.537229</td>
      <td>7.790794e+05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65.092659</td>
      <td>2006.382035</td>
      <td>315.846143</td>
      <td>154.092486</td>
      <td>485.243032</td>
      <td>722.418617</td>
      <td>333.071342</td>
      <td>35.841757</td>
      <td>1.386769e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.572581</td>
      <td>1559.262370</td>
      <td>288.209262</td>
      <td>308.626980</td>
      <td>129.241452</td>
      <td>277.528350</td>
      <td>557.706725</td>
      <td>36.230769</td>
      <td>8.178460e+05</td>
    </tr>
  </tbody>
</table>
</div>



 - __Label 0__: Customers that generally spend more in our ecommerce and they're more likely to be looking for vintage or decoration items

- __Label 1__: Customers that are more likely to cancel an order. They can be dissatisfied with the service. It is interesting to think about sending a NPS to collect some information on how to attract more people like them.

- __Label 2__: Customers that spend less than label 0, but they're satisfied with the service/products. They're looking for smaller items, from categ_0 (and that explains the less amount of money spent)
