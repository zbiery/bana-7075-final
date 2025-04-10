from src.pipeline import pipeline

x_train, x_test, y_train, y_test = pipeline(model="nnet")
print(x_train)