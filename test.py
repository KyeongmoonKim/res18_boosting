import tensorflow as tf
import res18
import data_csv as data
import configure as cf
import numpy as np
import utils as ut
import csv

curr_mode = cf.TES_CSV
curr_path = cf.TES_IMG_PATH
fd = open(cf.TES_LOG, 'w', newline='')
wr = csv.writer(fd)

cnt_matrix = []
for i in range(0, cf.CLS_NUM):
    cnt_matrix.append([0]*cf.CLS_NUM)

ckpt_path = cf.MODEL1_TES_CKPT_PATH 
sess = tf.Session()
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
answers = tf.placeholder(tf.float32, [None, cf.CLS_NUM])
train_mode = tf.placeholder(tf.bool)

test_sz = 0
f_temp = open(curr_mode, 'r')
rdr_temp = csv.reader(f_temp)
for line in rdr_temp:
  test_sz = test_sz + 1
f_temp.close()
print("test sz : " + str(test_sz))
dt = data.Data(curr_path, curr_mode, batch_size = cf.BATCH_SIZE)
curr_net = res18.Network(ckpt_path, trainable=False, fine_tuning=False)
curr_net.build(images, train_mode)

sess.run(tf.global_variables_initializer())

var_list = curr_net.build_tensors()
#print(var_list)
loader = tf.train.Saver(var_list)

loader.restore(sess, ckpt_path)
print("start testing")
end = int(test_sz/cf.BATCH_SIZE)
bunmo = int(test_sz/cf.BATCH_SIZE)*cf.BATCH_SIZE
ret = []
total_cnt = 0
for i in range(0, end):
    im_list, la_list, nm_list = dt.get_batch()
    #print(im_list.shape)
    #print(la_list.shape)
    r1, r2 = sess.run([curr_net.prob, curr_net.fc8_d1], feed_dict={images: im_list, train_mode:False})
    cnt = 0
    for j in range(0, cf.BATCH_SIZE):
      lab= np.argmax(la_list[j])
      infer = np.argmax(r1[j])
      print("answer : " + str(lab)+", inference : "+str(infer))
      if lab == infer:
        cnt = cnt+1
    print(str(i)+" batch accuracy : "+str(cnt/cf.BATCH_SIZE))
    total_cnt = total_cnt + cnt
print("total accuracy : "+str(total_cnt * 100/bunmo))