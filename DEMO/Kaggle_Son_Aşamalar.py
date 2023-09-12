# Kaggle da işimiz kolaylasın diye iki fonksiyon
#Gerekenler:

#prediction //// model: model nesnesi, param_grid: dictionary, X: train X, y train Y, X_test test X

#save //// test_predictions: output of the prediction function , ids: dataframe["ID"] gibi bir yapı , outputcolumn1name, outputcolumn2name, outputfilename





def save(test_predictions, ids , outputcolumn1name, outputcolumn2name, outputfilename):
  import pandas as pd

  # Create a DataFrame by zipping passenger IDs and predictions
  result_df = pd.DataFrame(list(zip(ids, test_predictions)), columns=[outputcolumn1name, outputcolumn2name])

  # Save the result to a CSV file
  result_df.to_csv(outputfilename, index=False)



def prediction(model, param_grid, X, y, X_test, n_splits=10):
  from sklearn.model_selection import GridSearchCV, StratifiedKFold


  grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=n_splits) , n_jobs=-1)

  grid_search.fit(X, y)


  best_model = grid_search.best_estimator_
  best_params = grid_search.best_params_
  test_predictions = best_model.predict(X_test)

  return test_predictions



def larger_row():
  import pandas as pd
  pd.options.display.max_rows = None  

def larger_column():
  import pandas as pd
  pd.options.display.max_column = None  