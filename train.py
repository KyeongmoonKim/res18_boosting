import tensorflow as tf
import res18 as net
import data_csv as data
import configure as cf
import csv
import numpy as np

'''BATCH_SIZE = 32
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
DROP_OUT = 0.5
CLS_NUM = 102
ITER = 50000
FINE_TUNING = True'''

log = cf.TRA_LOG
f = open(log,'w')
wr = csv.writer(f)

ckpt_path = cf.MODEL1_TRA_CKPT_PATH
sess = tf.Session()
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
answers = tf.placeholder(tf.float32, [None, cf.CLS_NUM])
train_mode = tf.placeholder(tf.bool)

dt = data.Data(cf.TRA_IMG_PATH, cf.TRA_CSV, oversampling=cf.OVER)
dt2 = data.Data(cf.VAL_IMG_PATH, cf.VAL_CSV, oversampling=False)
curr_net = net.Network(ckpt_path, fine_tuning=cf.FINE_TUNING, hard=cf.HARD)
curr_net.build(images, train_mode)


    
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=answers, logits=curr_net.fc8_d1)
loss = tf.reduce_mean(cross_entropy) 
optimizer = tf.train.AdamOptimizer(cf.LEARNING_RATE).minimize(loss)
#graph finish

sess.run(tf.global_variables_initializer())

var_list = curr_net.build_tensors()
#print(var_list)
loader = tf.train.Saver(var_list)

loader.restore(sess, ckpt_path)
print("start training")
for i in range(0, cf.ITER):
    if(i%cf.SAVE_ITER == 0)and(i!=0):
        print("trying to saving...")
        curr_net.save_model(sess, i)
    im_list, la_list, nm_list = dt.get_batch()
    r1, r2, r0= sess.run([optimizer, loss, curr_net.prob], feed_dict={images: im_list, answers: la_list, train_mode:True})
    im_list_val, la_list_val, nm_list_val = dt2.get_batch()
    l, clsnum = la_list_val.shape
    r4, r3 = sess.run([curr_net.fc8_d1, curr_net.prob], feed_dict={images: im_list_val, answers: la_list_val, train_mode:False})
    ans_cnt = 0
    ans_tra_cnt = 0
    for j in range(0, l):
      lab_tra = np.argmax(la_list[j])
      inf_tra = np.argmax(r0[j])
      lab = np.argmax(la_list_val[j])
      inf = np.argmax(r3[j])
      #print("val ans : "+str(lab)+", val infer : "+str(inf)+", tra ans : " +str(lab_tra) + ", tra inf : "+str(inf_tra))
      if lab_tra == inf_tra:
        ans_tra_cnt = ans_tra_cnt+1  
      if lab==inf:
        ans_cnt = ans_cnt+1
    acr_tra = (ans_tra_cnt * 100.0) /cf.BATCH_SIZE
    acr = (ans_cnt * 100.0) / cf.BATCH_SIZE
    line = [i, r2, acr, acr_tra]
    wr.writerow(line)
    print("iteration "+ str(i)+" => loss : " + str(r2)+", val acuraccy : " +str(acr)+", tra accuracy : "+str(acr_tra))
f.close()
    
    