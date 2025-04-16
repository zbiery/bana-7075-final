from src.pipeline import pipeline
from src.training import train_lr_model, train_nn_model, train_rf_model

# Train & Register Neural Network Model
x_train, x_test, y_train, y_test, scaler = pipeline(model="nnet")
train_nn_model(x_train, 
               y_train, 
               x_test, 
               y_test, 
               scaler=scaler,
               lr=0.001,
               hidden_dims=(64,32,16),
               epochs=25)

# # Train & Register Logistic Regression Model
# x_train, x_test, y_train, y_test = pipeline(model="glm")
# train_lr_model(x_train, 
#                y_train, 
#                x_test, 
#                y_test,
#                C=1,
#                max_iter=100,
#                solver='lbfgs',
#                penalty='l2')

# # Train & Register Random Forest Model
# x_train, x_test, y_train, y_test = pipeline(model="tree")
# train_rf_model(x_train, 
#                y_train, 
#                x_test, 
#                y_test,
#                n_estimators=100,
#                max_depth=None,
#                min_samples_split=2,
#                min_samples_leaf=1, 
#                max_features='sqrt')
