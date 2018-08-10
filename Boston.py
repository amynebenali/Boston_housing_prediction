
# coding: utf-8

# #  Boston Housing Prices

# In[21]:


# Importing a few necessary libraries
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import kurtosis, skew
import warnings
warnings.simplefilter('ignore')

# Make matplotlib show plots inline
get_ipython().run_line_magic('matplotlib', 'inline')

# Create our client's feature set for 
# which we will be predicting a selling price
CLIENT_FEATURES = [[13.44, 0.00, 18.587, 0, 0.7858, 4.8880, 82.00,                     1.655, 23, 566.0, 18.65, 400.05, 14.58]]

# Load the Boston Housing dataset into the boston variable : i didn't use the csv provided because it 
# contains all data in one column 
boston = datasets.load_boston()  

# Initialize the housing prices and housing features
housing_prices = boston.target
housing_features = boston.data
#Store in DataFrame
attributes = boston.feature_names #feature names
df_data = pd.DataFrame(housing_features, columns = attributes)
df_target = pd.DataFrame(housing_prices, columns =['MEDV'])
df_boston = pd.concat([df_data, df_target,], axis = 1) #concat data/target
print (boston.DESCR)

print ("*****************************")
print ("Summary\n")
feats = df_boston.shape[1]
obs = df_boston.shape[0]
print ("Number of Housing Features: ", feats)
print ("Number of Houses: ", obs)



print("Boston Housing dataset loaded successfully!")


# 
# # Statistical Analysis and Data Exploration
# 

# In[2]:


df_boston.head()


# In[3]:


df_boston.describe()


# In[4]:


pl.title("Histogram",fontsize=16)
sns.distplot(housing_prices, kde=True,bins=20,color='b',
             kde_kws={"color": "r", "lw": 2, "label": "KDE"})


# In[5]:


pl.figure()
fig,axes = pl.subplots(4, 4, figsize=(14,18))
fig.subplots_adjust(wspace=.4, hspace=.4)
img_index = 0
for i in range(boston.feature_names.size):
    row, col = i // 4, i % 4
    axes[row][col].scatter(boston.data[:,i],boston.target)
    axes[row][col].set_title(boston.feature_names[i] + ' and MEDV')
    axes[row][col].set_xlabel(boston.feature_names[i])        
    axes[row][col].set_ylabel('MEDV')
pl.show()


# In[6]:


fig, ax = pl.subplots(figsize=(10,10))
pl.title("Correlation Plot",fontsize=16)
sns.heatmap(df_boston,cmap=sns.diverging_palette(220, 20, sep=10, as_cmap=True))


# In[7]:


#Correlation between attributes
pd.set_option('display.width', 100)
pd.set_option('precision', 3)
correlations = df_boston.corr(method='pearson')
print(correlations)


# In[8]:


#Skewness
df_boston.skew() #The skew result show a positive (right) or negative (left) skew. Values closer to zero show less skew.


# In[9]:


# Number of houses and features in the dataset
total_houses, total_features = boston.data.shape

# Minimum housing value in the dataset
minimum_price = housing_prices.min()

# Maximum housing value in the dataset
maximum_price = housing_prices.max()

# Mean house value of the dataset
mean_price = housing_prices.mean()

# Median house value of the dataset
median_price = np.median(housing_prices)

# Standard deviation of housing values of the dataset
std_dev = housing_prices.std()

# Show the calculated statistics
print("Boston Housing dataset statistics (in $1000's):\n")
print("Total number of houses:", total_houses)
print("Total number of features:", total_features)
print("Minimum house price:", minimum_price)
print("Maximum house price:", maximum_price)
print("Mean house price: {0:.3f}".format(mean_price))
print("Median house price:", median_price)
print("Standard deviation of house price: {0:.3f}".format(std_dev))


# In[10]:


print(CLIENT_FEATURES)


# In[11]:


print('Client CRIM = ' + str(CLIENT_FEATURES[0][0]))
print('Client RAD = ' + str(CLIENT_FEATURES[0][8])) 
print('Client B = ' + str(CLIENT_FEATURES[0][11]))


# 
# ## Picking evaluation method 
# 

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def shuffle_split_data(X, y):
    """ 
    Shuffles and splits data into 70% training and 30% testing subsets,
    then returns the training and testing subsets. 
    """
    # Shuffled data
    X_s, y_s = shuffle(X, y, random_state=0)

    # Split the data into training (70%) and testing (30%)
    X_train, X_test, y_train, y_test = train_test_split(X_s, y_s,
                                                        test_size=0.3,
                                                        random_state=0)

    # Return the training and testing data subsets
    return X_train, y_train, X_test, y_test


# Test shuffle_split_data
try:
    X_train, y_train, X_test, y_test = shuffle_split_data(housing_features, 
                                                          housing_prices)
    print("Successfully shuffled and split the data!")
except:
    print("Something went wrong with shuffling and splitting the data.")
    
print("Shape of training data: ", X_train.shape)
print("Shape of training target: ", y_train.shape)
print("Shape of testing data: ", X_test.shape)
print("Shape of testing target: ", y_test.shape)


# In[13]:


from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def performance_metric(y_true, y_predict):
    """ 
    Calculates and returns the total error between true 
    and predicted values
    based on a performance metric chosen by the student. 
    """
    error = MAE(y_true, y_predict)
    return error

# Test performance_metric
try:
    total_error = performance_metric(y_train, y_train)
    print("Successfully performed a metric calculation!")
except:
    print("Something went wrong with performing a metric calculation.")


# In[14]:


from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def fit_model(X, y):
    """
    Tunes a decision tree regressor 
    model using GridSearchCV on the input data X 
    and target labels y and returns this optimal model.
    """

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Set up the parameters we wish to tune
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    # Make an appropriate scoring function
    scoring_function = make_scorer(performance_metric, 
                                   greater_is_better=False)

    # Make the GridSearchCV object
    reg = GridSearchCV(regressor, parameters, 
                       scoring_function)

    # Fit the learner to the data to obtain the optimal 
    # model with tuned parameters
    reg.fit(X, y)

    # Return the optimal model
    return reg.best_estimator_

# Test fit_model on entire dataset
try:
    reg = fit_model(housing_features, housing_prices)
    print("Successfully fit a model!")
except:
    print("Something went wrong with fitting a model.")


# 
# ## Analyzing Model Performance
# 

# In[15]:


def learning_curves(X_train, y_train, X_test, y_test):
    """
    Calculates the performance of several models with 
    varying sizes of training data. The learning and testing 
    error rates for each model are then plotted. 
    """
    
    print("Creating learning curve graphs for max_depths of 1, 3, 6, and 10. . .")
    
    # Create the figure window
    fig = pl.figure(figsize=(10,8))

    # We will vary the training set size so that 
    # we have 50 different sizes
    sizes = np.rint(np.linspace(1, len(X_train), 50)).astype(int)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    # Create four different models based on max_depth
    for k, depth in enumerate([1,3,6,10]):
        
        for i, s in enumerate(sizes):
            
            # Setup a decision tree regressor so that 
            # it learns a tree with max_depth = depth
            regressor = DecisionTreeRegressor(max_depth = depth)
            
            # Fit the learner to the training data
            regressor.fit(X_train[:s], y_train[:s])

            # Find the performance on the training set
            train_err[i] = performance_metric(y_train[:s], 
                                              regressor.predict(X_train[:s]))
            
            # Find the performance on the testing set
            test_err[i] = performance_metric(y_test, 
                                             regressor.predict(X_test))

        # Subplot the learning curve graph
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, test_err, lw = 2, label = 'Testing Error')
        ax.plot(sizes, train_err, lw = 2, label = 'Training Error')
        ax.legend()
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Data Points in Training Set')
        ax.set_ylabel('Total Error')
        ax.set_xlim([0, len(X_train)])
    
    # Visual aesthetics
    fig.suptitle('Decision Tree Regressor Learning Performances', 
                 fontsize=18, y=1.03)
    fig.tight_layout()
    pl.show()


# In[16]:


def model_complexity(X_train, y_train, X_test, y_test):
    """ 
    Calculates the performance of the model 
    as model complexity increases. The learning and 
    testing errors rates are then plotted. 
    """
    
    print("Creating a model complexity graph. . . ")

    # We will vary the max_depth of a decision tree 
    # model from 1 to 14
    max_depth = np.arange(1, 14)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns 
        # a tree with depth d
        regressor = DecisionTreeRegressor(max_depth = d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, 
                                          regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, 
                                         regressor.predict(X_test))

    # Plot the model complexity graph
    pl.figure(figsize=(7, 5))
    pl.title('Decision Tree Regressor Complexity Performance')
    pl.plot(max_depth, test_err, lw=2, label = 'Testing Error')
    pl.plot(max_depth, train_err, lw=2, label = 'Training Error')
    pl.legend()
    pl.xlabel('Maximum Depth')
    pl.ylabel('Total Error')
    pl.show()


# In[17]:


learning_curves(X_train, y_train, X_test, y_test)


# In[18]:


model_complexity(X_train, y_train, X_test, y_test)


# 
# ## Model Prediction
# 

# In[19]:


max_depths = []
for _ in range(100):
    reg = fit_model(housing_features, housing_prices)
    max_depths.append(reg.get_params()['max_depth'])
print("GridSearchCV max_depth result for DecisionTreeRegression model: ")
print("Median:", np.median(max_depths))
print("Mean:", np.mean(max_depths), ", Standard deviation:", np.std(max_depths))


# In[20]:


sale_price = reg.predict(CLIENT_FEATURES)
print("Predicted value of client's home: {0:.3f}".format(sale_price[0]))

