from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance



def parameter_tuning(X_train, y_train):
  cl = RandomForestClassifier()
  param_grid = {
      'n_estimators': [100, 150, 200], #[100, 150, 200, 500]
      'max_features': ['auto', 'sqrt'], #['auto', 'sqrt', 'log2']
      'criterion': ['gini', 'entropy'] #['gini', 'entropy']
      }
  print('Using Random Forest')

  # Create a GridSearchCV object to find the best hyperparameters
  grid_search = GridSearchCV(cl, param_grid, scoring='accuracy', cv=5, verbose=10)
  # Fit the GridSearchCV object to the training data
  grid_search.fit(X_train, y_train)
  best_params = grid_search.best_params_
  cv_results = grid_search.cv_results_
  best_score = grid_search.best_score_
  print("Best hyperparameters:", best_params)

  return best_params, best_score, cv_results