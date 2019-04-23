# Neural network implementation
from sklearn.neural_network import MLPClassifier
# Function for loading the breast cancer data set
from sklearn.datasets import load_breast_cancer
# Function for splitting data into training & testing sets
from sklearn.model_selection import train_test_split 

data = load_breast_cancer()

attributes = data.data
labels = data.target

#print(attributes)
#print(labels)

attributes_train, attributes_test, labels_train, labels_test = train_test_split(attributes, labels, test_size = 0.33)


#neuralnet = MLPClassifier()
neuralnet = MLPClassifier(solver="lbfgs", activation = "logistic" , alpha = 10.0)
neuralnet.fit(attributes_train, labels_train)
accuracy = neuralnet.score(attributes_test, labels_test)

print(str(accuracy*100) + "% accuracy")

##############################################################################

#Trying another algorithm

from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()

randomforest.fit(attributes_train, labels_train)
rfaccuracy = randomforest.score(attributes_test, labels_test)

print(str(rfaccuracy*100)+"% accuracy for randomforest")






