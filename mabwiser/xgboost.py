from typing import Dict, List, Union, Optional, NoReturn, NamedTuple
from mabwiser.utils import argmax, Arm, Num, _BaseRNG
from mabwiser.base_mab import BaseMAB
from sklearn.calibration import LabelEncoder
import xgboost as xgb
import numpy as np


class _XGBMab(BaseMAB):
    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str]) -> None:
        super().__init__(rng, arms, n_jobs, backend)
        self.xgb_model = xgb.XGBClassifier(colsample_bytree=0.8, subsample=1, reg_lambda=0, reg_alpha=2, n_estimators=60, gamma=2, learning_rate=0.2, min_child_weight=10, max_depth=4, max_delta_step=10)
        self.label_encoder = LabelEncoder()
        self.arms = arms
        self._is_initial_fit = False
    

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
        
        #encode arm names to numbers
        encoded_decisions = self.label_encoder.fit_transform(decisions)

        #remove all rows where response is != 1.0
        positive_decisions = encoded_decisions[rewards == 1.0]
        positive_contexts = contexts[rewards == 1.0]

        #split the data into train and test
        train_indices = np.random.choice(len(positive_decisions), len(positive_decisions)//2)

        y_train = positive_decisions[train_indices]
        X_train = positive_contexts[train_indices]

        mask = np.ones(len(positive_decisions), np.bool)
        mask[train_indices] = 0

        y_test = positive_decisions[mask]
        X_test = positive_contexts[mask]

        evalset = [(X_train, y_train), (X_test,y_test)]

        # Train the model
        self.xgb_model.fit(X_train, y_train, eval_set=evalset, early_stopping_rounds=100)
    
        self._is_initial_fit = True


    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
        print("Partial fit not implemented yet")
        pass

    def predict(self, contexts: Optional[np.ndarray] = None) -> Union[Arm, List[Arm]]:
        # Return the arm with maximum expectation
        expectations = self.predict_expectations(contexts)
        if isinstance(expectations, dict):
            return argmax(expectations)
        else:
            return [argmax(exp) for exp in expectations]

    def predict_expectations(self, contexts: Optional[np.ndarray] = None) -> Union[Dict[Arm, Num], List[Dict[Arm, Num]]]:
        probabilities = self.xgb_model.predict_proba(contexts[0])

        # Sort the probabilities and class labels in descending order of probability
        sorted_indices = np.argsort(-probabilities, axis=1)

        original_class_labels = self.label_encoder.inverse_transform(self.xgb_model.classes_)
        sorted_original_labels = original_class_labels[sorted_indices].tolist()
        sorted_probabilities = np.take_along_axis(probabilities, sorted_indices, axis=1).tolist()

        ## TODO also do explore! here
        
        return { arm: score for (arm, score) in zip(sorted_original_labels, sorted_probabilities) }
        

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):
        pass

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
        pass

    def is_contextual(self):
        return True