from src.pipeline import pipeline
from src.training import train_lr_model

x_train, x_test, y_train, y_test = pipeline(model="glm")
train_lr_model(x_train, y_train, x_test, y_test)