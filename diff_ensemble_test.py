import tensorflow as tf
import res18
import vgg16
import data_csv as data
import configure as cf
import csv
import numpy as np
import os
import random

curr_mode = cf.TES_CSV
curr_path = cf.TES_IMG_PATH

f_log = open(cf.TES_LOG, 'w', newline = '')
log_wr = csv.writer(f_log)

test_sz = 0
f_temp = open(curr_mode, 'r')
rdr_temp = csv.reader(f_temp)
for line in rdr_temp:
  test_sz = test_sz + 1
f_temp.close()

print("test sz : " + str(test_sz))

sess = tf.Session()

#model1, model2 build
images_model1 = tf.placeholder(tf.float32, [None, 224, 224, 3])
images_model2 = tf.placeholder(tf.float32, [None, 224, 224, 3])
answers_model1 = tf.placeholder(tf.float32, [None, cf.CLS_NUM])
answers_model2 = tf.placeholder(tf.float32, [None, cf.CLS_NUM])
train_mode_model1 = tf.placeholder(tf.bool)
train_mode_model2 = tf.placeholder(tf.bool)
#model1, test
net_model1 = res18.Network(cf.MODEL1_TES_CKPT_PATH, trainable=False, fine_tuning=False)
net_model1.build(images_model1, train_mode_model1)
#model2, train
net_model2 = vgg16.Network(cf.MODEL2_TRA_CKPT_PATH, trainable=False, fine_tuning=False)
net_model2.build(images_model2, train_mode_model2)

dt_model1 = data.Data(curr_path, curr_mode, oversampling=False, batch_size = cf.BATCH_SIZE)

sess.run(tf.global_variables_initializer())

#restore weights

var_list1 = net_model1.build_tensors()
var_list2 = net_model2.build_tensors()

loader1 = tf.train.Saver(var_list1)
loader2 = tf.train.Saver(var_list2)

loader1.restore(sess, cf.MODEL1_TES_CKPT_PATH)
loader2.restore(sess, cf.MODEL2_TES_CKPT_PATH)

#first test start

end = int(test_sz/cf.BATCH_SIZE) 
bunmo = end * cf.BATCH_SIZE
print("test case")
print(bunmo)

second_sz = 0

total_cnt = 0

# inference and label begua

for i in range(0, end):
    print(str(i) + " batch")
    im_list, la_list, nm_list = dt_model1.get_batch()
    #print(im_list.shape)
    r1, r2 = sess.run([net_model1.prob, net_model1.fc8_d1], feed_dict={images_model1: im_list, train_mode_model1:False})
    infer_list = [-1] * cf.BATCH_SIZE
    idx_list_next = []
    im_list_next = []
    batch_sz_next = 0
    cnt = 0
    for j in range(0, cf.BATCH_SIZE): #make batch for second model1
        infer = np.argmax(r1[j])
        if r1[j][infer] < cf.RANK1_THRESH_HOLD: #too low accurcacy.
            idx_list_next.append(j)
            im_list_next.append(im_list[j].reshape(1, 224, 224, 3))
            batch_sz_next = batch_sz_next + 1
        else: #
            infer_list[j] = infer
    if batch_sz_next!=0:
      im_list_next = np.concatenate(im_list_next)
      print(im_list_next.shape)
      r3, r4 = sess.run([net_model2.prob, net_model2.fc8_d1], feed_dict={images_model2: im_list_next, train_mode_model2:False})
      for j in range(0, batch_sz_next):
        infer = np.argmax(r3[j])
        infer_list[idx_list_next[j]] = np.argmax(r3[j])
    for j in range(0, cf.BATCH_SIZE):
        ans = np.argmax(la_list[j])
        print("answer : " + str(ans) +", inference : " + str(infer_list[j]))
        if ans == infer_list[j]:
            cnt = cnt+1
    print(str(i)+" batch accuracy : "+str(cnt*100/cf.BATCH_SIZE))
    total_cnt = total_cnt + cnt

print("accuracy : " + str(total_cnt*100 / bunmo))
print("total correct : " + str(total_cnt))
