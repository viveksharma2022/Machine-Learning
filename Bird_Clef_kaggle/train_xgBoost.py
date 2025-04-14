import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, classification_report
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import logging
import gc
import torch

class AppLogger:
    _logger = None

    @staticmethod
    def init(log_file=str(Path(__file__).resolve().parent) + '/xgboostTrain_log.txt', level=logging.DEBUG, to_console=True):
        if AppLogger._logger is not None:
            return  # Already initialized

        logger = logging.getLogger('AppLogger')
        logger.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_path = Path(log_file).resolve()

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        AppLogger._logger = logger

    @staticmethod
    def info(message):
        if AppLogger._logger:
            AppLogger._logger.info(message)

    @staticmethod
    def warning(message):
        if AppLogger._logger:
            AppLogger._logger.warning(message)

    @staticmethod
    def error(message):
        if AppLogger._logger:
            AppLogger._logger.error(message)

    @staticmethod
    def debug(message):
        if AppLogger._logger:
            AppLogger._logger.debug(message)

class Model:
    def __init__(self):
        # Initialize the LabelEncoders for categorical columns
        self.trainXEncoder = LabelEncoder()
        self.trainYEncoder = LabelEncoder()

        # Initialize the CatBoostClassifier with necessary parameters
        self.model = None 

        # Specify categorical columns (adjust if needed)
        self.categoricalColsTrainY = "primary_label"

        # Variable to store the model in-memory (as a byte buffer)
        self.model_buffer = io.BytesIO()

        self.params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',  # Use "hist" method
            'device': 'cuda',       # Set device to GPU (CUDA)
            'max_depth': 3,
            'eta': 0.1,
            'num_class': 206,
            'verbosity': 1 
            }

    def _EncodeCategorical(self, trainX, trainY):
        # # Only encode the categorical column in trainX (input features)
        # trainX.loc[:, self.categoricalColsTrainX] = self.trainXEncoder.fit_transform(trainX[self.categoricalColsTrainX])

        # Encode the target column in trainY (target labels)
        trainY.loc[:, self.categoricalColsTrainY] = self.trainYEncoder.fit_transform(trainY[self.categoricalColsTrainY])
        trainY.loc[:, self.categoricalColsTrainY] = trainY.loc[:, self.categoricalColsTrainY].astype(int)

    def _GetMetricScores(self, y_true, y_pred):
        # Assuming y_true and y_pred are your true labels and predicted labels
        mcc = matthews_corrcoef(y_true, y_pred)
        AppLogger.debug(classification_report(y_true, y_pred, zero_division=0))
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # or 'macro' or 'micro'
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        return metrics
    
    def Train(self, trainX, trainY):

        # Encode categorical columns
        self._EncodeCategorical(trainX, trainY)

        # Split to train and validation sets
        xTrain, xVal, yTrain, yVal = train_test_split(trainX, trainY, test_size=0.1, random_state=42)

        # Function to load data in chunks
        def LoadDataInChunks(chunkSize=50000):
            for start in range(0, len(xTrain), chunkSize):  # iterate by chunk size
                end = min(start + chunkSize, len(xTrain))
                AppLogger.info(f'Train Range: {start}, {end}')
                yield xTrain.iloc[start:end], yTrain.iloc[start:end]

        # Training in chunks
        for i, (xtrainChunk, yTrainChunk_df) in enumerate(LoadDataInChunks()):
            yTrainChunk = yTrainChunk_df[self.categoricalColsTrainY].astype(int)
        
            # Filter and prepare validation set
            yVal_filtered_df = yVal[yVal[self.categoricalColsTrainY].isin(yTrainChunk.unique())]
            xVal_filtered = xVal.loc[yVal_filtered_df.index]
            yVal_filtered = yVal_filtered_df[self.categoricalColsTrainY].astype(int)
        
            dtrain = xgb.DMatrix(xtrainChunk, label=yTrainChunk)
            dval = xgb.DMatrix(xVal_filtered, label=yVal_filtered)
            
            evals = [(dtrain, 'train'), (dval, 'eval')]

            if i == 0:
                self.model = xgb.train(self.params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10)
            else:
                self.model = xgb.train(self.params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10, xgb_model=self.model)

            # check metrics in every batch
            y_proba_val = self.model.predict(xgb.DMatrix(xVal_filtered), output_margin=False)  # shape: (n_samples, n_classes)
            y_pred_val = np.argmax(y_proba_val, axis=1)
            
            y_proba_train = self.model.predict(xgb.DMatrix(xtrainChunk), output_margin=False)  # shape: (n_samples, n_classes)
            y_pred_train = np.argmax(y_proba_train, axis=1)

            AppLogger.info("Train metrics")
            val_scores_train = self._GetMetricScores(yTrainChunk, y_pred_train)
            AppLogger.debug(f"""Train scores: Accuracy = {val_scores_train['accuracy']},
            Precision = {val_scores_train['precision']}, Recall = {val_scores_train['recall']},
            MCC = {val_scores_train['mcc']}""")

            AppLogger.info("Validation metrics")
            val_scores_val = self._GetMetricScores(yVal_filtered, y_pred_val)
            AppLogger.debug(f"""validation scores: Accuracy = {val_scores_val['accuracy']}, \
            Precision = {val_scores_val['precision']}, Recall = {val_scores_val['recall']},
            MCC = {val_scores_val['mcc']}""")

            # Save model from time to time
            if i % 5 == 0:
                self.model.save_model(str(Path(__file__).resolve().parent) + '/data/xgBoostModel.json')
            
            # Clear memory for processed chunk data (not needed for next iteration)
            torch.cuda.empty_cache()  # This will free up GPU memory if using CUDA
            del xtrainChunk, yTrainChunk_df, yVal_filtered_df, xVal_filtered, dtrain, dval, evals, y_proba_val, y_pred_val, y_proba_train, y_pred_train
            gc.collect()
            
if __name__ == "__main__":
    # Assuming trainX and trainY are defined as your training data and labels

    data = pd.read_csv(str(Path(__file__).resolve().parent) + '/data/dataMFCC.csv', dtype={1: str})
    data.head()
    
    trainX = data.drop(columns = ['Unnamed: 0', 'primary_label', 'secondary_labels', 'type', 'filename',
       'collection', 'rating', 'url', 'latitude', 'longitude',
       'scientific_name', 'common_name', 'author', 'license', 'class_name'])

    trainY = data[['primary_label']]

    AppLogger.init()

    model = Model()
    model.Train(trainX, trainY)
