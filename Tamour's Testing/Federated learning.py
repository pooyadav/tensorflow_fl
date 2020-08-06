import tensorflow as tf
import tensorflow_federated as tff

READINGS_TYPE = tff.FederatedType(tf.float32, tff.CLIENTS)
THRESHOLD_TYPE = tff.FederatedType(tf.float32, tff.SERVER)

@tff.federated_computation(READINGS_TYPE)
def get_average_tempreature(sensor_readings):
    return tff.federated_mean(sensor_readings)

@tff.tf_computation
def exceeds_threshold_fn(reading, threshold):
    return tf.to_float(reading > threshold)

@tff.tf_computation(tf.float32, tf.float32)
def exceeds_threshold_fn(reading,threshold):
    return tf.to_float(reading > threshold)

@tff.federated_computation(READINGS_TYPE, THRESHOLD_TYPE)
def get_fraction_over_threshold(readings, threshold):
    return tff.federeated_mean(
        tff.federated_map(
            exceeds_threshold_fn,
            [readings, tff.federated_broadcast(threshold)]))

