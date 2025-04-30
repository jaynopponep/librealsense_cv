import argparse
import math
import h5py
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import socket
import importlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import glob
from pathlib import Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=20, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))
os.system('cp train.py %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 1

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

def prepare_pcd_data():
    data_dir = 'data_pcd'
    pcd_files = glob.glob(os.path.join(data_dir, '*.pcd'))
    
    if not pcd_files:
        print("No PCD files found in the data_pcd directory")
        exit(1)
    
    print(f"Found {len(pcd_files)} PCD files for training")
    
    # Split files into train/test sets (80/20 split)
    split_idx = int(len(pcd_files) * 0.8)
    train_files = pcd_files[:split_idx]
    test_files = pcd_files[split_idx:]
    
    # Create train and test file lists
    train_file_list = os.path.join(data_dir, 'train_files.txt')
    test_file_list = os.path.join(data_dir, 'test_files.txt')
    
    with open(train_file_list, 'w') as f:
        for file_path in train_files:
            f.write(f"{file_path}\n")
            
    with open(test_file_list, 'w') as f:
        for file_path in test_files:
            f.write(f"{file_path}\n")
    
    return train_file_list, test_file_list

def load_pcd_file(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('VERSION') or line.startswith('FIELDS') or \
               line.startswith('SIZE') or line.startswith('TYPE') or line.startswith('COUNT') or \
               line.startswith('WIDTH') or line.startswith('HEIGHT') or line.startswith('VIEWPOINT') or \
               line.startswith('POINTS') or line.startswith('DATA'):
                continue
            values = line.strip().split()
            if len(values) >= 3:
                x, y, z = float(values[0]), float(values[1]), float(values[2])
                points.append([x, y, z])
    
    if not points:
        return None, None
    
    # Convert to numpy array and reshape
    points = np.array(points)
    
    # Sample/pad to ensure we have exactly NUM_POINT points
    if len(points) > NUM_POINT:
        # Randomly sample points
        idx = np.random.choice(len(points), NUM_POINT, replace=False)
        points = points[idx, :]
    elif len(points) < NUM_POINT:
        # Pad by duplicating points
        idx = np.random.choice(len(points), NUM_POINT - len(points))
        extra_points = points[idx, :]
        points = np.vstack((points, extra_points))
    
    # All files are treated as the same class (0)
    labels = np.zeros(1, dtype=np.int32)
    
    # Add batch dimension
    points = np.expand_dims(points, 0)
    
    return points, labels

def load_batch_pcd_files(file_list):
    all_points = []
    all_labels = []
    
    with open(file_list, 'r') as f:
        for line in f:
            file_path = line.strip()
            points, labels = load_pcd_file(file_path)
            if points is not None:
                all_points.append(points)
                all_labels.append(labels)
    
    # Stack all data
    if all_points:
        all_points = np.vstack(all_points)
        all_labels = np.hstack(all_labels)
        return all_points, all_labels
    else:
        return None, None

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,
                        batch * BATCH_SIZE,
                        DECAY_STEP,
                        DECAY_RATE,
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    # Prepare data
    train_file_list, test_file_list = prepare_pcd_data()
    
    with tf.Graph().as_default():
        with tf.device('/cpu:0'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            MODEL.NUM_CLASSES = NUM_CLASSES
            
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            saver = tf.train.Saver()
            
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer, train_file_list)
            eval_one_epoch(sess, ops, test_writer, test_file_list)
            
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer, train_file_list):
    is_training = True
    
    current_data, current_label = load_batch_pcd_files(train_file_list)
    if current_data is None:
        log_string('No training data found')
        return
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
   
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx+1) * BATCH_SIZE, file_size)
        
        batch_data = current_data[start_idx:end_idx, :, :]
        batch_label = current_label[start_idx:end_idx]
        
        # Augment batched point clouds by rotation and jittering
        rotated_data = provider.rotate_point_cloud(batch_data)
        jittered_data = provider.jitter_point_cloud(rotated_data)
        
        feed_dict = {ops['pointclouds_pl']: jittered_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}
        
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (end_idx - start_idx)
        loss_sum += loss_val
    
    if num_batches > 0:
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))
    else:
        log_string('No batches processed')
        
def eval_one_epoch(sess, ops, test_writer, test_file_list):
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    current_data, current_label = load_batch_pcd_files(test_file_list)
    if current_data is None:
        log_string('No test data found')
        return
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx+1) * BATCH_SIZE, file_size)
        
        batch_data = current_data[start_idx:end_idx, :, :]
        batch_label = current_label[start_idx:end_idx]
        
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}
        
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (end_idx - start_idx)
        loss_sum += (loss_val * (end_idx - start_idx))
    
    if total_seen > 0:
        log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
        log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    else:
        log_string('No test samples evaluated')

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
