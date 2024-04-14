import numpy as np
from utils.helper import parse_args, load_data, split_data, train_model, get_predictions, evaluate_model, store_predictions

config = parse_args()

# Step 1: Load the data
print("Loading dataset...")
features = load_data(feature_path=config['path']['features'])

# Step 2: Split the dataset using 10-fold cross validation
print(f"Splitting dataset into train set and test set with test size as {config['test_size']} ...")
X_train, X_test, y_train, y_test, id_train, id_test, label_encoder = split_data(features=features,test_size=config['test_size'],seed=config['seed'])

#  Step 3: Train a machine learning model and predict the subjects of the scientific papers
print(f"Training the {config['classifier']['type']} Model...")
clf = train_model(X_train=X_train, y_train=y_train,classifier= config['classifier']['type'], kernel= config['classifier']['kernel'],seed=config['seed'],kfold=config['kfold'])
train_predictions = get_predictions(clf=clf,X=X_train)
test_predictions = get_predictions(clf=clf,X=X_test)

# Step 4: Store predictions in a file
print(f"Storing the predictions in file '{config['path']['predictions']}' ...")
predictions = np.concatenate([train_predictions,test_predictions], axis=0)
paper_ids = np.concatenate([id_train,id_test], axis=0)
store_predictions(predictions=predictions, path=config['path']['predictions'], paper_ids=paper_ids, label_encoder=label_encoder)

# Step 5: Evaluates the approach in terms of accuracy 
print(f"Evaluating the quality of the {config['classifier']['type']} Model...")
print("Train Set:")
evaluate_model(predictions=train_predictions, y=y_train, evaluation_metric=config['results']['metric'])
print("Test Set:")
evaluate_model(predictions=test_predictions, y=y_test, evaluation_metric=config['results']['metric'])