from src.pipeline import pipeline
from src.training import train_nn_model
from sklearn.preprocessing import StandardScaler

# Load pipeline data
x_train, x_test, y_train, y_test, scaler = pipeline(model="nnet")

# Define some parameter combinations to try
experiment_configs = [
    {"epochs": 20, "lr": 0.001, "hidden_dims": (64, 32), "batch_size": 32},
    {"epochs": 30, "lr": 0.0005, "hidden_dims": (128, 64, 32), "batch_size": 32},
    {"epochs": 25, "lr": 0.0003, "hidden_dims": (64, 64), "batch_size": 16},
    {"epochs": 40, "lr": 0.0007, "hidden_dims": (128, 32), "batch_size": 64},
]

# Run experiments
for config in experiment_configs:
    run_name = f"nn_e{config['epochs']}_lr{config['lr']}_h{'-'.join(map(str, config['hidden_dims']))}_bs{config['batch_size']}"
    print(f"Running: {run_name}")
    train_nn_model(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        scaler=scaler,
        epochs=config["epochs"],
        lr=config["lr"],
        hidden_dims=config["hidden_dims"],
        batch_size=config["batch_size"],
        run_name=run_name
    )
