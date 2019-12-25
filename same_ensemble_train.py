import tensorflow as tf
import res18
import data_csv as data
import configure as cf
import csv
import numpy as np
import os
import random

#test all the training set using model1, to make next beg.
f_log = open(cf.TRA_LOG, 'w', newline = '')
log_wr = csv.writer(f_log)
f_beg_pool = open('beg_pool.csv', 'w', newline='')
beg_pool_wr = csv.writer(f_beg_pool)

#test_sz set
test_sz = 0
f_temp = open(cf.TRA_CSV, 'r')
rdr_temp = csv.reader(f_temp)
for line in rdr_temp:
  test_sz = test_sz + 1
  beg_pool_wr.writerow(line)
f_temp.close()
print("test sz : " + str(test_sz))


#graph def
model1_graph = tf.Graph()
model2_graph = tf.Graph()
#model1
with model1_graph.as_default():
    images_model1 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    answers_model1 = tf.placeholder(tf.float32, [None, cf.CLS_NUM])
    train_mode_model1 = tf.placeholder(tf.bool)
    net_model1 = res18.Network(cf.MODEL1_TES_CKPT_PATH, trainable=False, fine_tuning=False)
    net_model1.build(images_model1, train_mode_model1)

#model2
with model2_graph.as_default():
    images_model2 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    answers_model2 = tf.placeholder(tf.float32, [None, cf.CLS_NUM])
    train_mode_model2 = tf.placeholder(tf.bool)
    net_model2 = res18.Network(cf.MODEL2_TRA_CKPT_PATH, trainable=cf.MODEL2_TRAINABLE, fine_tuning=cf.MODEL2_FINE_TUNING)
    net_model2.build(images_model2, train_mode_model2)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=answers_model2, logits=net_model2.fc8_d1)
    loss = tf.reduce_mean(cross_entropy) 
    optimizer = tf.train.AdamOptimizer(cf.LEARNING_RATE).minimize(loss)

dt_model1 = data.Data(cf.TRA_IMG_PATH, cf.TRA_CSV, oversampling=False, batch_size = cf.BATCH_SIZE)

#sess def
model1_sess = tf.Session(graph=model1_graph)
model2_sess = tf.Session(graph=model2_graph)

#restore model1
with model1_sess.as_default():
    with model1_graph.as_default():
        model1_sess.run(tf.global_variables_initializer())
        var_list1 = net_model1.build_tensors()
        loader1 = tf.train.Saver(var_list1)
        loader1.restore(model1_sess, cf.MODEL1_TES_CKPT_PATH)
#restore model2
with model2_sess.as_default():
    with model2_graph.as_default():
        model2_sess.run(tf.global_variables_initializer())
        var_list2 = net_model2.build_tensors()
        loader2 = tf.train.Saver(var_list2)
        loader2.restore(model2_sess, cf.MODEL2_TRA_CKPT_PATH)
        
end = int(test_sz/cf.BATCH_SIZE)
with model1_sess.as_default():
    with model1_graph.as_default():
        for i in range(0, end):
            im_list, la_list, nm_list = dt_model1.get_batch()
            r1, r2 = model1_sess.run([net_model1.prob, net_model1.fc8_d1], feed_dict={images_model1: im_list, train_mode_model1:False})
            l, clsnum = la_list.shape 
            cnt = 0
            cnt_passed = 0
            for j in range(0, l): #make batch for next model train, the 
                #print(nm_list[j] + ", ans : " + str(np.argmax(la_list[j])) +", inference : " + str(np.argmax(r1[j])))
                lab = np.argmax(la_list[j])
                inf = np.argmax(r1[j])
                if lab!=inf: #inference isn't correct => to batch. this may be changed for machine perfomance. (ex) thresh hold.
                    cnt_passed = cnt_passed+1
                    for k in range(0, cf.BEG_WEIGHT-1):
                        beg_pool_wr.writerow([nm_list[j], lab])
                else: #correct!
                    if r1[j][inf] < cf.RANK1_THRESH_HOLD: #too small
                      cnt_passed = cnt_passed+1
                      for k in range(0, cf.BEG_WEIGHT-1):
                        beg_pool_wr.writerow([nm_list[j], lab])
                    cnt = cnt+1
            print(str(i)+" batch accuracy : "+str(cnt/cf.BATCH_SIZE)+", passed cnt : "+str(cnt_passed))

f_beg_pool.close()
print("beg_pool is made completely!")

#set for beg making
f_beg_pool = open('beg_pool.csv', 'r')
beg_pool_rdr = csv.reader(f_beg_pool)
f_beg = open('beg.csv', 'w', newline='')
f_beg_wr = csv.writer(f_beg)
#start
num = test_sz
name_list = []
label_list = []

for line in beg_pool_rdr:
    name_list.append(line[0])
    label_list.append(int(line[1]))
f_beg_pool.close()

rand_idx = random.sample(range(len(label_list)), len(label_list))
limit = int(num*cf.BEG_RATIO)
cnt = 0
for i in range(0, len(name_list)):
    if cnt == limit:
        break
    idx = rand_idx[i]
    label = int(label_list[idx])
    cnt = cnt + 1
    f_beg_wr.writerow([name_list[idx], label])

del name_list
del rand_idx
del dt_model1
del label_list 

f_beg.close()
print("beg is made completely!, and model2 start training!")
dt_model2 = data.Data(cf.TRA_IMG_PATH, 'beg.csv', oversampling=False, batch_size = cf.BATCH_SIZE)
dt2_model2 = data.Data(cf.VAL_IMG_PATH, cf.VAL_CSV, oversampling=False, batch_size = cf.BATCH_SIZE)

for i in range(0, cf.ITER):
    if(i%cf.SAVE_ITER == 0)and(i!=0):
        print("trying to saving...")
        with model2_sess.as_default():
          with model2_graph.as_default():
            net_model2.save_model(model2_sess, i)
    im_list, la_list, nm_list = dt_model2.get_batch()
    with model2_sess.as_default():
      with model2_graph.as_default():
        r1, r2, r0= model2_sess.run([optimizer, loss, net_model2.prob], feed_dict={images_model2: im_list, answers_model2: la_list, train_mode_model2:True})
    tra_cnt = 0
    for j in range(0, cf.BATCH_SIZE):
      inf = np.argmax(r0[j])
      lab = np.argmax(la_list[j])
      if inf == lab:
        tra_cnt = tra_cnt + 1
    im_list_val, la_list_val, name_list_val = dt2_model2.get_batch()
    with model1_sess.as_default():
      with model1_graph.as_default():
        vr0, vr1 = model1_sess.run([net_model1.fc8_d1, net_model1.prob], feed_dict={images_model1: im_list_val, train_mode_model1:False})
    infer_list = [-1] * cf.BATCH_SIZE
    idx_list_next = []
    im_list_next = []
    batch_sz_next = 0
    cnt = 0
    for j in range(0, cf.BATCH_SIZE): #model 1 inference
      infer = np.argmax(vr1[j])
      if vr1[j][infer] < cf.RANK1_THRESH_HOLD:
        idx_list_next.append(j)
        im_list_next.append(im_list_val[j].reshape(1, 224, 224, 3))
        batch_sz_next = batch_sz_next + 1
      else:
        infer_list[j] = infer
    if batch_sz_next != 0: #model 2 in
      im_list_next = np.concatenate(im_list_next)
      print(im_list_next.shape)
      with model2_sess.as_default():
        with model2_graph.as_default():
          vr3, vr4 = model2_sess.run([net_model2.prob, net_model2.fc8_d1], feed_dict={images_model2: im_list_next, train_mode_model2:False})
      for j in range(0, batch_sz_next):
          infer = np.argmax(vr3[j])
          infer_list[idx_list_next[j]] = np.argmax(vr3[j])
    for j in range(0, cf.BATCH_SIZE):
      ans = np.argmax(la_list_val[j])
      if ans==infer_list[j]:
        cnt = cnt+1
    acr = (cnt * 100) / cf.BATCH_SIZE
    print("iteration "+ str(i)+" => loss : " + str(r2)+", val accuraccy : " +str(acr)+", model2 tra accuracy : "+str(tra_cnt*100/cf.BATCH_SIZE))
    log_wr.writerow([i, r2, acr, tra_cnt*100/cf.BATCH_SIZE])
    
f_log.close()
#remove all created additional file
os.remove('beg_pool.csv')
os.remove('beg.csv')
