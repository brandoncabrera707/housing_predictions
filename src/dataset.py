from sklearn.model_selection import train_test_split
import pandas as pd
def split_training_val_test(X, y, random_state = 42, test_size = 0.33):
  """ Returns 7 DataFrames: X_train, X_val, X_test, y_train, y_val, y_test, train_df
  
  Args:
      X (DataFrame): features used for predicting
      y (DataFrame): predictor target
      random_state (Int):
      test_size (Float):
  """
  X_main, X_test, y_main, y_test = train_test_split(X,y, random_state = random_state, test_size = test_size)
  X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, random_state = random_state, test_size = test_size)
  train_df = pd.concat([X_train, y_train], axis = 1)
  return X_train, X_val, X_test, y_train, y_val, y_test, train_df

  
  
