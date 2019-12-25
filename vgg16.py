import inspect
import os
import numpy as np
import tensorflow as tf
import configure as cf

VGG_MEAN = [103.939, 116.779, 123.68]
'''
    fine tuning 할때는 네트워크 2개를 분리하면됨. feature map이랑 그거로.
'''
class Network:
    def __init__(self, save_path=None, trainable=True, fine_tuning=True, hard=True):
        if save_path is not None: #load
            self.ckpt_reader = tf.train.NewCheckpointReader(save_path)
            self.var_to_shape_map =  self.ckpt_reader.get_variable_to_shape_map()
            for key in self.var_to_shape_map:
                print("tensor_name: ", key)
        else:
            self.ckpt_reader=None
        self.trainable = trainable
        self.restore_name_list = []
        self.save_name_list = []
        self.fine_tuning = fine_tuning
        self.hard = hard
    def build_tensors(self):
        self.var_list = list()
        for name in self.restore_name_list:
            try:
                tensor_aux = tf.get_default_graph().get_tensor_by_name(name+':0')
            except:
                print('Not found: '+name)
                assert False
            self.var_list.append(tensor_aux)
        return self.var_list
        
    def set_var(self, initial_value, var_name):
        self.save_name_list.append(var_name)
        if self.ckpt_reader is not None: #reading parametername
            if var_name in self.var_to_shape_map:
                self.restore_name_list.append(var_name)
                print(var_name)
            else:
                print(var_name + " is not found")
                assert False
        if self.trainable:
            if not self.fine_tuning: #all training
                var = tf.Variable(initial_value, name=var_name)
            elif self.hard: 
                var = tf.Variable(initial_value, name=var_name)
            else:
                var = tf.Variable(initial_value, name=var_name, trainable=False)
        else:
            var = tf.Variable(initial_value, name=var_name, trainable=False)
            #var = tf.constant(initial_value, dtype=tf.float32, name=var_name)
        # print var_name, var.get_shape().as_list()
        assert var.get_shape().as_list() == initial_value.get_shape().as_list()

        return var
  
    def conv_layer(self, bottom, in_channels, out_channels, name):
      #with tf.variable_scope(name):
        filt, conv_biases = self.set_conv_var(3, in_channels, out_channels, name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(bias)

        return relu
        
    def set_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.set_var(initial_value, name + "/weights")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.set_var(initial_value, name + "/biases")

        return filters, biases

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
   



    def build(self, images, train_mode=None): #trainable : dropout.
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=images)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        
        self.conv1_1 = self.conv_layer(bgr, 3, 64, "vgg_16/conv1/conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "vgg_16/conv1/conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'vgg_16/pool1')
    
        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "vgg_16/conv2/conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "vgg_16/conv2/conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'vgg_16/pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "vgg_16/conv3/conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "vgg_16/conv3/conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "vgg_16/conv3/conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'vgg_16/pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "vgg_16/conv4/conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "vgg_16/conv4/conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "vgg_16/conv4/conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'vgg_16/pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "vgg_16/conv5/conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "vgg_16/conv5/conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "vgg_16/conv5/conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'vgg_16/pool5')
        #fc layer
        print(self.pool5.get_shape)
        
        #fc6
        init = tf.truncated_normal([7, 7, 512, 4096], 0.0, 0.001)
        name = "vgg_16/fc6"
        var_name = name + "/weights"
        weights_fc6 =tf.Variable(init, name=var_name)
        self.save_name_list.append(var_name)
        if (self.ckpt_reader is not None) and (not self.fine_tuning): #reading parametername
            if var_name in self.var_to_shape_map:
                self.restore_name_list.append(var_name)
                print(var_name)
            else:
                print(var_name + " is not found")
                assert False
        
        init = tf.truncated_normal([4096], .0, .001)
        var_name = name+ "/biases"
        biases_fc6 = tf.Variable(init, name=var_name)
        self.save_name_list.append(var_name)
        if (self.ckpt_reader is not None) and (not self.fine_tuning): #reading parametername
            if var_name in self.var_to_shape_map:
                self.restore_name_list.append(var_name)
                print(var_name)
            else:
                print(var_name + " is not found")
                assert False
        
        conv = tf.nn.conv2d(self.pool5, weights_fc6, [1, 7, 7, 1], padding='SAME')
        self.fc6 = tf.nn.bias_add(conv, biases_fc6)
        print(self.fc6.get_shape)
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, cf.DROP_OUT), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, cf.DROP_OUT)
        
        print(weights_fc6.get_shape)
        #fc 7
        init = tf.truncated_normal([1, 1, 4096, 4096], 0.0, 0.001)
        name = "vgg_16/fc7"
        var_name = name + "/weights"
        weights_fc7 =tf.Variable(init, name=var_name)
        self.save_name_list.append(var_name)
        if (self.ckpt_reader is not None) and (not self.fine_tuning): #reading parametername
            if var_name in self.var_to_shape_map:
                self.restore_name_list.append(var_name)
                print(var_name)
            else:
                print(var_name + " is not found")
                assert False
        
        init = tf.truncated_normal([4096], .0, .001)
        var_name = name+ "/biases"
        biases_fc7 = tf.Variable(init, name=var_name)
        self.save_name_list.append(var_name)
        if (self.ckpt_reader is not None) and (not self.fine_tuning): #reading parametername
            if var_name in self.var_to_shape_map:
                self.restore_name_list.append(var_name)
                print(var_name)
            else:
                print(var_name + " is not found")
                assert False
        print(self.relu6.get_shape)
        conv = tf.nn.conv2d(self.relu6, weights_fc7, [1, 1, 1, 1], padding='SAME')
        self.fc7 = tf.nn.bias_add(conv, biases_fc7)
        print(self.fc7.get_shape)
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, cf.DROP_OUT), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, cf.DROP_OUT)
        
        #fc8
        init = tf.truncated_normal([1, 1, 4096, cf.CLS_NUM], 0.0, 0.001)
        name = "vgg_16/fc8"
        var_name = name + "/weights"
        weights_fc8 =tf.Variable(init, name=var_name)
        self.save_name_list.append(var_name)
        if (self.ckpt_reader is not None) and (not self.fine_tuning): #reading parametername
            if var_name in self.var_to_shape_map:
                self.restore_name_list.append(var_name)
                print(var_name)
            else:
                print(var_name + " is not found")
                assert False
        
        init = tf.truncated_normal([cf.CLS_NUM], .0, .001)
        var_name = name+ "/biases"
        biases_fc8 = tf.Variable(init, name=var_name)
        self.save_name_list.append(var_name)
        if (self.ckpt_reader is not None) and (not self.fine_tuning): #reading parametername
            if var_name in self.var_to_shape_map:
                self.restore_name_list.append(var_name)
                print(var_name)
            else:
                print(var_name + " is not found")
                assert False
        conv = tf.nn.conv2d(self.relu7, weights_fc8, [1, 1, 1, 1], padding='SAME')
        self.fc8 = tf.nn.bias_add(conv, biases_fc8)
        print(self.fc8.get_shape)
        self.fc8_d1 = tf.reshape(self.fc8, [-1, cf.CLS_NUM])
        print(self.fc8_d1)
        self.prob = tf.nn.softmax(self.fc8_d1, name="vgg_16/prob")

    def save_model(self, sess, iter):
        #print([n.name for n in tf.get_default_graph().as_graph_def().node])
        temp_var_list = list()
        for name in self.save_name_list:
            try:
                tensor_aux = tf.get_default_graph().get_tensor_by_name(name+':0')
            except:
                print('Not found: '+name)
                assert False
            temp_var_list.append(tensor_aux)
        print(temp_var_list)
        temp_saver = tf.train.Saver(temp_var_list)
        ckpt_path = temp_saver.save(sess, cf.VGG_SAVE_PATH+cf.VGG_SAVE_NAME+str(iter)+'.ckpt')
        print("save ckpt file:", ckpt_path)
