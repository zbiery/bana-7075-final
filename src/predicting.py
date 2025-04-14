def lr_predict(model, X_new, threshold=0.5):
    """
    Generates predictions on new input data using the trained logistic regression model.

    Args:
        model (LogisticRegression): Trained sklearn logistic regression model
        X_new (np.ndarray): New input data (must match training features)
        threshold (float): Decision threshold for converting probabilities to 0/1

    Returns:
        probs (np.ndarray): Raw predicted probabilities
        preds (np.ndarray): Binary predictions (0 or 1)
    """

    # Get probabilities and binary predictions
    probs = model.predict_proba(X_new)[:, 1]
    preds = (probs >= threshold).astype(int)

    return probs, preds
