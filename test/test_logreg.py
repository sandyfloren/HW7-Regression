"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
#from sklearn
from regression import LogisticRegressor, loadDataset
# (you will probably need to import more things here)



X_train, X_val, y_train, y_val = loadDataset(
        features=[
			'GENDER',
			'Body Mass Index',
			'Triglycerides',
			'Carbon Dioxide',
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )


model = LogisticRegressor(num_feats=10, learning_rate=0.00001, tol=0.0001, max_iter=100, batch_size=5)

def test_prediction():

	X_padded = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	y_pred = model.make_prediction(X_padded)
	
	assert len(y_pred) == X_padded.shape[0], '`y_pred` should be the same length as the number of observations in `X`.'
	assert np.min(y_pred) > 0, 'Predictions should be in (0, 1)'
	assert np.max(y_pred) < 1, 'Predictions should be in (0, 1)'

def test_loss_function():
	loss_same = model.loss_function(y_train, y_train)
	loss_diff = model.loss_function(y_train, 1 - y_train)

	assert np.abs(loss_same) < 1e-5, 'Loss was calculated incorrectly.'
	assert loss_diff < 0, 'Loss was calculated incorrectly.'

def test_gradient():
	X_padded = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

	assert len(model.calculate_gradient(y_train, X_padded)) == model.num_feats + 1, 'Gradient vector has incorrect length.'

def test_training():
	W_start = model.W.copy()
	model.train_model(X_train, y_train, X_val, y_val)
	W_end = model.W.copy()

	assert np.not_equal(W_start, W_end).any(), 'Weights were not updated during training.'
