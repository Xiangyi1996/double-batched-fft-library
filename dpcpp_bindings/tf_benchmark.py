import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Dense
import time
import numpy as np

print(tf.config.list_physical_devices())
# TF_ENABLE_ONEDNN_OPTS=1
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)
times = []
throughputs = []


for exp in range(20, 21):
    # for exp in range(8, 9):
    n_samples = 2**exp
    X_train = np.random.rand(n_samples, 64)
    X_train = X_train.astype(np.float16)
    y_train = np.random.rand(n_samples, 64)
    y_train = y_train.astype(np.float16)
    print(y_train.shape)
    with tf.device("/XPU:0"):
        arch = [Dense(64, activation="relu", input_dim=64), Dense(64)]
        model = tf.keras.models.Sequential(arch)
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        start_time = time.perf_counter()
        model.fit(X_train, y_train, epochs=1000, batch_size=n_samples, verbose=0)

        end_time = time.perf_counter()
        times.append(end_time - start_time)

        print(f"Batchsize {n_samples}: {times[-1]}s")

# for x in times:
#     print(times)
