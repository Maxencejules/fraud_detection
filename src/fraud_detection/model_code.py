"""Entry point for MLflow "models from code".

MLflow executes this file when the logged model is loaded, instead of unpickling a
Python object. Artifacts and the feature contract are injected through
``FraudModel.load_context``.
"""

from mlflow.models import set_model

from fraud_detection.modeling import FraudModel

set_model(FraudModel())
