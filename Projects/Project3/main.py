from img2vec_pytorch import Img2Vec #type: ignore
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna
import pickle

# Preparing the data 
img2vec = Img2Vec(cuda=True) # cuda = True if you want to use GPU

train_dir = './training_set'
test_dir = './test_set'

# Function to extract features from images in a directory
def extract_features_n_labels_from_directory(directory):
    
    features = []
    labels = []
    for category in os.listdir(directory):
        for i,image in enumerate(os.listdir(os.path.join(directory, category))):
            img = Image.open(os.path.join(directory, f'cat.{i+1}.jpg')) # open image
# why is it showing 'Image' is not defined?
# 
            img_features = img2vec.get_vec(img) # extract features
            features.append(img_features)
            labels.append(category) # append label

    return features, labels

train_features, train_labels = extract_features_n_labels_from_directory(train_dir)
test_features, test_labels = extract_features_n_labels_from_directory(test_dir)


# LETS TRAIN A MODEL

def objective(trial):
    # invoke suggest method of a Trial Object to generate hyperparamets 
    regressor_name = trial.suggest_categorical('regressor', ['RandomForest', 'SVC', 'XGBClassifier'])
    if regressor_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    elif regressor_name == 'SVC':
        C = trial.suggest_float('C', 1e-6, 1e+1, log=True)
        gamma = trial.suggest_float('gamma', 1e-6, 1e+1, log=True)
        model = SVC(C=C, gamma=gamma)
    else:
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    model.fit(train_features, train_labels)
    preds = model.predict(test_features)
    return accuracy_score(test_labels, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('Best trial:')
trial = study.best_trial

# best hyperparameters
print('  Value: {}'.format(trial.value))

# best score
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# Train the best model with the best hyperparameters
best_regressor_name = trial.params['regressor']
if best_regressor_name == 'RandomForest':
    best_model = RandomForestClassifier(n_estimators=trial.params['n_estimators'], max_depth=trial.params['max_depth'])
elif best_regressor_name == 'SVC':
    best_model = SVC(C=trial.params['C'], gamma=trial.params['gamma'])
else:
    best_model = XGBClassifier(learning_rate=trial.params['learning_rate'], n_estimators=trial.params['n_estimators'], max_depth=trial.params['max_depth'])

best_model.fit(train_features, train_labels)
best_preds = best_model.predict(test_features)
print('Accuracy of the best model:', accuracy_score(test_labels, best_preds))

# save the best model 
with open('best_image_classifier.pkl', 'wb') as f:
    pickle.dump(best_model, f)