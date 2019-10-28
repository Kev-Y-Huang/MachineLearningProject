import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from flask import Flask, request

dataset = pd.read_csv("sportsref_download.csv")

include = ['Rk', 'Age', 'Wt', '40YD', 'Vertical', 'BenchReps', 'Broad Jump', '3Cone', 'Shuttle', 'Drafted (tm/rnd/yr)']
dataset_ = dataset[include]

dataset_["Drafted (tm/rnd/yr)"] = dataset_["Drafted (tm/rnd/yr)"].str.extract(r'(\d{1})', expand=False)
dataset_['Drafted (tm/rnd/yr)'].fillna('7', inplace=True)
dataset_.fillna(dataset_.mean(), inplace=True)
print(dataset_)

# Check how many of each species we have
dataset_.groupby('Drafted (tm/rnd/yr)').size()

# splitting up the labels and the values for each species:
feature_columns = ['Age', 'Wt', '40YD', 'Vertical', 'BenchReps', 'Broad Jump', '3Cone', 'Shuttle']
X = dataset_[feature_columns].values
Y = dataset_['Drafted (tm/rnd/yr)'].values

# Data Visualization:
plt.figure(figsize=(15,10))
parallel_coordinates(dataset_.drop("Rk", axis=1), "Drafted (tm/rnd/yr)")
plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
plt.xlabel('Features', fontsize=15)
plt.ylabel('Features values', fontsize=15)
plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
plt.show()

# Splitting into training and test datasets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)

# Creating the learning model
knn_classifier = KNeighborsClassifier(n_neighbors=10)

# Fitting the model with the training data
knn_classifier.fit(X_train, Y_train)

# Making predictions with the test data (This line is also where we would potentially classify new data)
Y_pred = knn_classifier.predict(X_test)
print(Y_pred)

# Finding Accuracy:
accuracy = accuracy_score(Y_test, Y_pred)*100
print('Accuracy of model: ' + str(round(accuracy, 2)) + ' %.')
cm = confusion_matrix(Y_test, Y_pred)
cm

# creating list of cv scores
cv_scores = []
k_list = [i for i in range(1, 50, 2)]

# perform 10-fold cross validation
for k in range(1, 50, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())



# Displaying results visually
plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.plot(k_list, cv_scores)

plt.show()

# Set up Flask App
app = Flask(__name__)


@app.route("/", methods=['GET'])
def classify():
    # array mapping numbers to flower names
    # classes = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]

    # get values for each component, return error message if not a float
    try:
        values = [[float(request.args.get(component)) for component in ['Age', 'Wt', '40YD', 'Vertical', 'BenchReps', 'BroadJump', '3Cone', 'Shuttle']]]
    except TypeError:
        return "An error occured\nUsage: 127.0.0.1:5000?Age=NUM&Wt=NUM&40YD=NUM&Vertical=NUM&BenchReps=NUM&BroadJump=NUM&3Cone=NUM&Shuttle"

    # Otherwise, return the prediction.
    prediction = knn_classifier.predict(values)[0]
    return prediction


# Run the app.
app.run()

# try 127.0.0.1:5000?Age=22&Wt=199&40YD=4.58&Vertical=40&BenchReps=13&BroadJump=124&3Cone=7.22&Shuttle=4.28