import matplotlib.pyplot as plt #visualisation des donnees
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer # remplir les données manquantes
from sklearn.metrics import accuracy_score
import seaborn as sns # création des graphiques statistiques
import numpy as np
import pandas as pd #manipulation et l'analyse des données

#classifiers
from sklearn.neighbors import KNeighborsClassifier  #K-NN  plus proche voisin
from sklearn.svm import LinearSVC  #SVM
from sklearn.svm import SVC        #SVM
from sklearn.linear_model import LogisticRegression  #Logistic Regression
from sklearn.ensemble import RandomForestClassifier  #Random Forests

train_data = pd.read_csv("train.csv")
train_data.head()

test_data = pd.read_csv("test.csv")
test_data.head()

all_data = pd.concat([train_data, test_data])
all_data.head()

#count combien de points de données sont présents dans chaque ensemble
print(f"The training set has {len(train_data)} datapoints")
print(f"The test set has {len(test_data)} datapoints")
print(f"Overall the whole dataset has {len(all_data)} datapoints")

all_data.info()

all_data.drop(columns=["Survived"]).describe()

#calcul le % de personne surviver et les nombre de pers
survival_rate = train_data["Survived"].mean()
print(f"Survival rate: {survival_rate}")
print(f"Death rate: {1-survival_rate}")

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("pourcentage of women who survived:", rate_women)
print("pourcentage of women who dead:", 1-rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("pourcentage of men who survived:", rate_women)
print("pourcentage of men who dead:", 1-rate_women)

to_show_features = "Sex"
fig, axs = plt.subplots(1, figsize=(10,5))
sns.countplot(x=train_data[to_show_features], hue='Survived', data=train_data.replace({"Survived": {0:"No", 1:"Yes"}}))
plt.show()

#determine s'il existe une forte corrélation entre les fonctionnalités
corr = train_data.corr()
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(corr, annot=True, linewidth=0.5)
ax.set_title("Correlation matrix")
plt.show()

#voir la relation entre les variables et les différentes distributions de données est à travers les paires
ax = sns.pairplot(train_data.drop(columns=["PassengerId"]).replace({"Survived": {0:"No", 1:"Yes"}}),
                                                                    hue="Survived",
                                                                    plot_kws={"alpha":0.5},
                                                                    diag_kind="hist")

fig, ax = plt.subplots(1, 2, figsize=(20,10))
sns.countplot(x=train_data.loc[train_data["Sex"] == "male"]["Pclass"], hue='Survived', data=train_data.replace({"Survived": {0:"No", 1:"Yes"}}), ax=ax[0])
ax[0].set_title("Male Survied Pclass")
sns.countplot(x=train_data.loc[train_data["Sex"] == "female"]["Pclass"], hue='Survived', data=train_data.replace({"Survived": {0:"No", 1:"Yes"}}), ax=ax[1])
ax[1].set_title("female Survied Pclass")

#preparation des donnees
features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]
X_train = pd.get_dummies(train_data[features])
# Remplir les valeurs manquantes
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
X_train = imputer.fit_transform(X_train)
# Standardiser les données
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
y_train = np.array(train_data["Survived"])
print(f"Shape of the training set: {X_train.shape}")

#entrainement
def trainClassifier(X_train, y_train, model_name, classifier, params, score, verbose=False, num_folds=10):

    kf = sklearn.model_selection.StratifiedKFold(num_folds)


    train_scores = []

    best_score = 0

    for config in sklearn.model_selection.ParameterGrid(params):
        train_scores_run = []
        counts = []
        for train_indices, valid_indices in kf.split(X_train, y_train):
            counts.append(len(train_indices))
            X_train_kf = X_train[train_indices]
            y_train_kf = y_train[train_indices]
            X_valid_kf = X_train[valid_indices]
            y_valid_kf = y_train[valid_indices]
            model = classifier(**config)
            model.fit(X_train_kf, y_train_kf)
            y_hat = model.predict(X_valid_kf)
            train_score = score(y_valid_kf, y_hat)
            train_scores_run.append(train_score)

        if np.average(train_scores_run, weights=counts) > best_score:
            best_score = np.average(train_scores_run, weights=counts)
            best_config = config
            if(verbose):
                print("New best score obtained")
                print(f"Training with: {config}")
                print(f"Total Score obtained with cross validation: {best_score}\n")

        train_scores.append(np.average(train_scores_run, weights=counts))

    output_df = pd.DataFrame(data = [[model_name, best_config ,best_score]], \
        columns=["model_name", "parameters", "training_score"])

    return output_df

#creation d'un vide data Frame pour le refactue les resultas
results = pd.DataFrame() 

#K-NN
params = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15]
}
classifier = KNeighborsClassifier

classifier_df = trainClassifier(X_train, y_train, "k-NN", classifier, params, accuracy_score)
results = results.append(classifier_df)

#SVM
params = {
    "C": [1e-3, 1e-2, 1e-1, 1],
    "max_iter": [30000]
}
classifier = LinearSVC
classifier_df = trainClassifier(X_train, y_train, "LinearSVC", classifier, params, accuracy_score)
results = results.append(classifier_df)
#
params = {
    "kernel" : ["rbf"],
    "C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
    "gamma": [1e-3, 1e-2, 1e-1, 1, 10]
}
classifier = SVC
classifier_df = trainClassifier(X_train, y_train, "SVC", classifier, params, accuracy_score)
results = results.append(classifier_df)

#Logistic Regression
params = {
    "C": [1e-3, 1e-2, 1e-1, 1, 10]
}
classifier = LogisticRegression
classifier_df = trainClassifier(X_train, y_train, "LogisticRegression", classifier, params, accuracy_score)
results = results.append(classifier_df)

#Random Forests
params = {"max_depth": [3, 5, 7, 10, None],
          "n_estimators":[3, 5,10, 25, 50],
          "max_features": [1, 2, "auto"]}
classifier = RandomForestClassifier
classifier_df = trainClassifier(X_train, y_train, "RandomForests", classifier, params, accuracy_score)
results = results.append(classifier_df)

#comparaison des modeles
results = results.set_index("model_name")
results


