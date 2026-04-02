# Models Module

## Purpose
Train, evaluate, and save ML models for drift detection experiments.

## Usage

```python
from models.trainer import ModelTrainer

# Train a model
trainer = ModelTrainer(model_type='rf')
trainer.train(X_train, y_train)

# Evaluate
accuracy = trainer.evaluate(X_test, y_test)

# Make predictions
predictions = trainer.predict(X_new)

# Save model for later use
trainer.save('models/credit_model.pkl')

# Load saved model
loaded_model = ModelTrainer.load('models/credit_model.pkl')
```

## Available Models

### RandomForest (`model_type='rf'`)
- **Pros**: Simple, interpretable, no tuning needed, handles mixed features
- **Cons**: Slower inference than LR
- **Used for**: All baseline experiments
- **Config**: n_estimators=100, random_state=42

## Model Selection Rationale

We use **only RandomForest** throughout to:
- Isolate drift detection effects (not model type differences)
- Keep training time fast (~2 seconds per model)
- Maintain reproducibility (no hyperparameter tuning needed)

## Future Extensions

To add another model (e.g., LightGBM):

```python
class ModelTrainer:
    def __init__(self, model_type='rf'):
        if model_type == 'rf':
            # ... existing code
        elif model_type == 'lgb':
            self.model = LGBMClassifier(...)
```

## Training Tips

- **Always use `random_state=42`** for reproducibility
- **Don't tune hyperparameters** - not the focus of this project
- **Log training time** for resource consumption analysis
- **Save trained models** - reuse across experiments

