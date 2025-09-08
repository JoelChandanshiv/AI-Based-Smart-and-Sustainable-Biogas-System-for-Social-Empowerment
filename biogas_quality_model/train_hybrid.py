# train_hybrid.py
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

from utils_preprocess import read_csv_time_series, make_windows, fit_scalers, transform_windows, encode_labels

# CONFIG
DATA_CSV = "data/sensor_timeseries.csv"  # provide data CSV with timestamp, sensors, label
SCALER_DIR = "saved/scalers"
MODEL_DIR = "saved/models"
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

WINDOW_SIZE = 20     # number of timesteps per window (tweak)
STEP = 5             # sliding step
TEST_SIZE = 0.2
RANDOM_STATE = 42
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 50

# 1) Read data
df = read_csv_time_series(DATA_CSV, timestamp_col="timestamp")
print("Loaded data:", df.shape)

# 2) Create windows
Xw, Xs, y_raw = make_windows(df, window_size=WINDOW_SIZE, step=STEP)
y, label_map = encode_labels(y_raw)
print("Windows:", Xw.shape, Xs.shape, y.shape)

# 3) Fit scalers based on data and save
sensor_scaler, stat_scaler = fit_scalers(Xw, Xs, SCALER_DIR)

# 4) Transform windows with scalers
Xw_scaled, Xs_scaled = transform_windows(Xw, Xs, SCALER_DIR)

# 5) Train/val split
Xw_train, Xw_val, Xs_train, Xs_val, y_train, y_val = train_test_split(
    Xw_scaled, Xs_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 6) Build LSTM model (returns embedding vector)
timesteps = Xw_train.shape[1]
n_sensors = Xw_train.shape[2]

def build_lstm_model(timesteps, n_sensors, units=64):
    inp = Input(shape=(timesteps, n_sensors), name="timeseries_input")
    x = LSTM(units, return_sequences=False, name="lstm_layer")(inp)  # last hidden state
    x = Dropout(0.2)(x)
    # optional direct LSTM prediction head (not necessary)
    out = Dense(1, activation='sigmoid', name="aux_out")(x)
    model = tf.keras.Model(inputs=inp, outputs=[out, x], name="lstm_with_embedding")
    return model

lstm_model = build_lstm_model(timesteps, n_sensors, units=LSTM_UNITS)
# The model has two outputs: aux_out (sigmoid) and embedding (dense vector)
lstm_model.compile(optimizer=Adam(1e-3), loss={'aux_out':'binary_crossentropy'}, metrics=['accuracy'])
lstm_model.summary()

# 7) Prepare y for aux output: binary mapping of classes; if multiclass, you can adapt
# For binary label mapping: assume 0/1 in y; if multiclass, derive a binary like "low vs rest" or use embedding only.
y_train_aux = (y_train > 0).astype(int)
y_val_aux = (y_val > 0).astype(int)

# 8) Train LSTM
checkpoint_path = os.path.join(MODEL_DIR, "lstm_best.h5")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
]
history = lstm_model.fit(
    Xw_train, {'aux_out': y_train_aux},
    validation_data=(Xw_val, {'aux_out': y_val_aux}),
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2
)

# 9) Build a smaller keras model that outputs embedding only (load best weights)
lstm_model.load_weights(checkpoint_path)
# Create embedding extractor model
embedding_model = tf.keras.Model(inputs=lstm_model.input, outputs=lstm_model.get_layer("lstm_layer").output)
embedding_model.summary()
# Save embedding model
embedding_model.save(os.path.join(MODEL_DIR, "lstm_embedding_model"))

# 10) Extract embeddings for train and val
emb_train = embedding_model.predict(Xw_train, batch_size=128)
emb_val = embedding_model.predict(Xw_val, batch_size=128)

# 11) Combine embeddings with stat features
Xgb_train = np.concatenate([Xs_train, emb_train], axis=1)
Xgb_val = np.concatenate([Xs_val, emb_val], axis=1)
print("XGB input shapes:", Xgb_train.shape, Xgb_val.shape)

# 12) Train XGBoost (multiclass or binary depending on y)
# Detect number of classes
num_classes = len(np.unique(y))
if num_classes == 2:
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss'
    )
else:
    xgb_model = XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

xgb_model.fit(Xgb_train, y_train, eval_set=[(Xgb_val,y_val)], early_stopping_rounds=20, verbose=True)

# 13) Save XGBoost and label map and scalers
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_model.joblib"))
joblib.dump(label_map, os.path.join(MODEL_DIR, "label_map.joblib"))
print("Saved models to", MODEL_DIR)
