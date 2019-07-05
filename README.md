# machine-learnin-2

#load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length','sepal-width', 'petal-length' , 'petal-width' , 'class']
dataset = pandas.read_csv(url, names=names)
#description
print(dataset.describe())
#description
#class distribution
print( dataset.groupby('class').size())

#box and whisker plots
dataset.plot(kind='box', subplots= True ,layout =(2,2),sharex= False, sharey= False)
plt.show()
#histogram
dataset.hist()
plt.show()
scatter_matrix(dataset)
plt.show()
array= dataset.values
X= array[:,0:4]
Y=array[:,4]
validation_size=0.20
seed=7
X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y, test_size=validation_size, random_state=seed)   
        
models=[]
models.append(('LR',LogisticRegression(solver='liblinear', multi_class='ovr')))

                                      
