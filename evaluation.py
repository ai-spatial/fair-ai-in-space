import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ---------------------------------------------------------------------
import numpy as np
from tensorflow import keras
from keras import backend as K
import pickle

# ---------------------------------------------------------------------
# Global Variable
model_dir = 'results'
dir_ckpt = model_dir + '/' + 'checkpoints'

MAX_ROW_PARTITION = 5
MAX_COL_PARTITION = 5

BATCH_SIZE = 256 * 256

INPUT_SIZE = 10  # number of features
NUM_LAYERS_DNN = 8  # num_layers is the number of non-input layers, one more LSTM
NUM_CLASS = 23
EPOCH_TRAIN = 300

CKPT_FOLDER_PATH = dir_ckpt
LEARNING_RATE = 0.001

CLASSIFICATION_LOSS = 'categorical_crossentropy'
REGRESSION_LOSS = 'mean_squared_error'

# ---------------------------------------------------------------------
# Read the X training dataset and the Y training dataset
X = np.load('X_test.npy')
y = np.load('y_test.npy')

# read sample indices for all partitions within each candidate partitioning
with open('test_id.pickle', 'rb') as handle:
    test_id = pickle.load(handle)


# ---------------------------------------------------------------------
# DNN model
class DenseNet(tf.keras.Model):

    def __init__(self, layer_size=INPUT_SIZE, num_class=NUM_CLASS, ckpt_path=CKPT_FOLDER_PATH):
        super(DenseNet, self).__init__()

        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.5)  # mean=0.0, seed=None
        self.dense = []
        for i in range(7):
            self.dense.append(tf.keras.layers.Dense(layer_size, activation=tf.nn.relu, kernel_initializer=initializer))

        self.out = tf.keras.layers.Dense(num_class, kernel_initializer=initializer)

        self.ckpt_path = ckpt_path
        self.model_name = 'dnn'
        print('check ckpt path: ' + ckpt_path)

    def call(self, inputs):  # , num_layers = NUM_LAYERS_DNN

        for i in range(7):
            if i == 0:
                layer = self.dense[i](inputs)
            else:
                layer = self.dense[i](layer)

        out_layer = tf.nn.softmax(self.out(layer))

        return out_layer

    def save_branch(self, branch_id):
        # save the current branch
        # branch_id should include the current branch (not after added to X_branch_id)
        self.save_weights(self.ckpt_path + '/' + self.model_name + '_ckpt_' + branch_id)
        return

    def load_base_branch(self, branch_id):
        # load the base branch before further fine-tuning
        self.load_weights(self.ckpt_path + '/' + self.model_name + '_ckpt_' + branch_id)
        return


def dice(y_true, y_pred):
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))

    intersect = K.sum(y_pred * y_true, axis=0) + K.epsilon()
    denominator = K.sum(y_pred, axis=0) + K.sum(y_true, axis=0)
    dice_scores = K.constant(2) * intersect / (denominator + K.epsilon())
    return 1 - dice_scores


def custom_loss(y_true, y_pred):
    loss = dice(y_true, y_pred)
    return loss


optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)


def model_compile(model):
    # optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
    global optimizer

    model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])


# ---------------------------------------------------------------------
def get_class_wise_accuracy(y_true, y_pred, prf=False):
    num_class = y_true.shape[1]
    stat = tf.keras.metrics.categorical_accuracy(y_true, y_pred)

    true_pred_w_class = y_true * np.expand_dims(stat, 1)
    true = np.sum(true_pred_w_class, axis=0).reshape(-1)
    total = np.sum(y_true, axis=0).reshape(-1)

    if prf:
        pred_w_class = tf.math.argmax(y_pred, axis=1)
        pred_w_class = tf.one_hot(pred_w_class, depth=NUM_CLASS).numpy()
        pred_total = np.sum(pred_w_class, axis=0).reshape(-1)
        return true, total, pred_total
    else:
        return true, total


def get_overall_accuracy(y_true, y_pred):
    stat = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    true = np.sum(stat)
    total = stat.shape[0]

    return true, total


def get_class_wise_list(y_true, y_pred):
    stat = tf.keras.metrics.categorical_accuracy(y_true, y_pred)

    true_pred_w_class = y_true * np.expand_dims(stat, 1)
    pred_w_class = tf.math.argmax(y_pred, axis=1)
    pred_w_class = tf.one_hot(pred_w_class, depth=NUM_CLASS).numpy()

    return true_pred_w_class, pred_w_class


def get_prf(true_class, total_class, pred_class):
    pre = true_class / pred_class
    rec = true_class / total_class

    pre_fix = np.nan_to_num(pre, nan=np.nanmean(pre))
    rec_fix = np.nan_to_num(rec, nan=np.nanmean(rec))
    f1 = 2 / (pre_fix ** (-1) + rec_fix ** (-1))
    return pre, rec, f1


def get_avg_f1(f1, total_class):
    avg_f1 = np.sum(f1 * total_class / np.sum(total_class))

    return avg_f1


def get_fairness_loss_all(y_test, y_pred, partition_list, all_partitioning_data_list):
    global MAX_ROW_PARTITION, MAX_COL_PARTITION, GLOBAL_MEAN

    fairness_loss_list = np.zeros((MAX_ROW_PARTITION * MAX_COL_PARTITION - 1), dtype='float')

    true_pred_w_class, pred_w_class = get_class_wise_list(y_test, y_pred)

    for (index1, index2) in partition_list:

        f1_list = np.zeros(index1 * index2, dtype='float')
        data_list = all_partitioning_data_list[(index1, index2)]

        for i in range(index1 * index2):
            true_class_part = np.sum(true_pred_w_class[data_list[i]], axis=0).reshape(-1)
            total_class_part = np.sum(y_test[data_list[i]], axis=0).reshape(-1)
            total_pred_part = np.sum(pred_w_class[data_list[i]], axis=0).reshape(-1)

            pre, rec, f1 = get_prf(true_class_part, total_class_part, total_pred_part)

            f1_list[i] = get_avg_f1(f1, total_class_part)

            fairness_loss_list[(index1 - 1) * MAX_COL_PARTITION + index2 - 1 - 1] = np.mean(
                np.abs(GLOBAL_MEAN - f1_list))

    return fairness_loss_list


def get_partition_data(index1, index2, X_data, y_data, all_partitioning_data_list):
    X_test = []
    y_test = []

    data_list = all_partitioning_data_list[(index1, index2)]
    for i in range(index1 * index2):
        X_test.append(X_data[data_list[i]])
        y_test.append(y_data[data_list[i]])

    return X_test, y_test


ROW_LIST = list(range(1, MAX_ROW_PARTITION + 1))
COL_LIST = list(range(1, MAX_COL_PARTITION + 1))

PARTITIONINGS = []

for r in ROW_LIST:
    for c in COL_LIST:
        if r == 1 and c == 1:
            continue

        PARTITIONINGS.append((r, c))

print(PARTITIONINGS)

# -----------------------------------------------------------------------------
base_model = DenseNet()
model_compile(base_model)
base_model.load_base_branch('base_model')

y_pred = base_model.predict(X, batch_size=BATCH_SIZE)
true_part, total_part = get_overall_accuracy(y, y_pred)

true_class_part, total_class_part, total_pred_part = get_class_wise_accuracy(y, y_pred, prf=True)
pre, rec, f1 = get_prf(true_class_part, total_class_part, total_pred_part)

loss = dice(y.astype('float32'), y_pred)
loss = tf.reduce_mean(loss).numpy()

with np.printoptions(precision=4, suppress=True):
    print('te_accuracy = {:.4f}\nloss = {}\nf1 = {}'.format(true_part / total_part, loss, f1))

GLOBAL_MEAN = get_avg_f1(f1, total_class_part)
fairness_loss_iter = get_fairness_loss_all(y, y_pred, PARTITIONINGS, test_id)
base_fairness = np.sum(fairness_loss_iter)

print('base', GLOBAL_MEAN, base_fairness)

# -----------------------------------------------------------------------------
model = DenseNet()
model_compile(model)
model.load_base_branch('model_final')

y_pred = model.predict(X, batch_size=BATCH_SIZE)
true_part, total_part = get_overall_accuracy(y, y_pred)

true_class_part, total_class_part, total_pred_part = get_class_wise_accuracy(y, y_pred, prf=True)
pre, rec, f1 = get_prf(true_class_part, total_class_part, total_pred_part)

loss = dice(y.astype('float32'), y_pred)
loss = tf.reduce_mean(loss).numpy()

with np.printoptions(precision=4, suppress=True):
    print('te_accuracy = {:.4f}\nloss = {}\nf1 = {}'.format(true_part / total_part, loss, f1))

GLOBAL_MEAN = get_avg_f1(f1, total_class_part)
fairness_loss_iter = get_fairness_loss_all(y, y_pred, PARTITIONINGS, test_id)
base_fairness = np.sum(fairness_loss_iter)

print('SPAD', GLOBAL_MEAN, base_fairness)
