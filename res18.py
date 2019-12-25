import inspect
import os
import numpy as np
import tensorflow as tf
import configure as cf

VGG_MEAN = [103.939, 116.779, 123.68]
'''
     res_block code need!
     fill build
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
    
    def batch_norm(self, inputs, name, train_mode):
        n_out = inputs.get_shape().as_list()[3]
        #save_list_save_name
        self.save_name_list.append(name+'gamma')
        self.save_name_list.append(name+'beta')
        self.save_name_list.append(name+'mu')
        self.save_name_list.append(name+'sigma')
        
        beta = tf.Variable(tf.zeros(shape=[n_out], dtype=tf.float32), name=name+'beta')
        gamma = tf.Variable(tf.ones(shape=[n_out], dtype=tf.float32), name=name+'gamma')
        mu = tf.Variable(tf.zeros(shape=[n_out], dtype=tf.float32), name=name+'mu', trainable=False)
        sigma = tf.Variable(tf.ones(shape=[n_out], dtype=tf.float32), name=name+'sigma', trainable=False)
        #save store name 
        a = [name+'gamma', name+'beta', name+'mu', name+'sigma']
        for var_name in a:
            if var_name in self.var_to_shape_map:
                self.restore_name_list.append(var_name)
                print(var_name)
            else:
                print(var_name + " is not found")
                assert False
        #doesn't update moving mean, var.
        '''batch_mean, batch_var = tf.nn.moments(inputs, [0,1,2])
        def temp():
            print("h")
            train_mean = tf.assign(mu, mu*0.9+batch_mean*0.1)
            train_var = tf.assign(sigma, sigma*0.9+batch_var*0.1)
            with tf.control_dependencies([train_mean, train_var]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        
        def temp2():
            print("x")
            return tf.identity(mu), tf.identity(sigma)
        mean, var = tf.cond(train_mode, temp, temp2)
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)'''
        normed = tf.nn.batch_normalization(inputs, mu, sigma, beta, gamma, 1e-3)
        return normed 
    
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
        
    def set_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.set_var(initial_value, name + "/weights")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.set_var(initial_value, name + "/biases")

        return filters, biases
    
    def conv_layer(self, bottom, in_channels, out_channels, name):
      #with tf.variable_scope(name):
        filt, conv_biases = self.set_conv_var(3, in_channels, out_channels, name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(bias)

        return relu
    
    def avg_pool(self, bottom, name): #global average pooling
        return tf.nn.avg_pool(bottom, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        
    
    def building_block(self, inputs, in_channels, out_channels, projection_shortcut, name, train_mode): #name format (ex) conv3_1, no batch norm.
        #shortcuts calculations
        shortcuts = None
        if projection_shortcut: #True, needs to change dimensions
            init = tf.truncated_normal([1, 1, in_channels, out_channels], 0.0, 0.001)
            shortcut_filter = self.set_var(init, name+'/shortcut/kernel')
            shortcuts = tf.nn.conv2d(inputs, shortcut_filter, [1, 2, 2, 1], padding='SAME')
        else:
            shortcuts = inputs
        print(name + "/shortcut")
        print(shortcuts.get_shape)
        #conv1
        conv1 = None
        if projection_shortcut: #True, need to change dimensions
            init = tf.truncated_normal([3, 3, in_channels, out_channels], 0.0, 0.001)
            filter1 = self.set_var(init, name+'/conv_1/kernel')
            conv1 = tf.nn.conv2d(inputs, filter1, [1, 2, 2, 1], padding='SAME')
        else:
            init = tf.truncated_normal([3, 3, in_channels, out_channels], 0.0, 0.001)
            filter1 = self.set_var(init, name+'/conv_1/kernel')
            conv1 = tf.nn.conv2d(inputs, filter1, [1, 1, 1, 1], padding='SAME')
        print(name + "/conv1")
        print(conv1.get_shape)
        conv1 = self.batch_norm(conv1, name+'/bn_1/', train_mode)
        print("after norm")
        print(conv1.get_shape)
        relu1 = tf.nn.relu(conv1)
        #conv2
        init = tf.truncated_normal([3, 3, out_channels, out_channels], 0.0, 0.001)
        filter2 = self.set_var(init, name+'/conv_2/kernel')
        conv2 = tf.nn.conv2d(relu1, filter2, [1, 1, 1, 1], padding='SAME')
        print(name + "/conv2")
        print(conv2.get_shape)
        conv2 = self.batch_norm(conv2, name+'/bn_2/', train_mode)
        sum = conv2 + shortcuts
        relu2 = tf.nn.relu(sum)
        return relu2
    
    def build(self, images, train_mode=None): #trainable : dropout.
        if train_mode is None:
            assert False
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
        filter1 = self.set_var(tf.truncated_normal([7, 7, 3, 64], 0.0, 0.001), 'conv_1/kernel')
        #conv1
        self.conv1 = tf.nn.conv2d(bgr, filter1, [1, 2, 2, 1], padding='SAME')
        print("conv1")
        print(self.conv1.get_shape)
        self.pool1 = self.max_pool(self.conv1, 'pool1')
        print("pool1")
        print(self.pool1.get_shape)
        #conv2, 2 blocks
        self.build1 = self.building_block(self.pool1, 64, 64, False, 'conv2_1', train_mode)
        self.build2 = self.building_block(self.build1, 64, 64, False, 'conv2_2', train_mode)
        #conv3, 2 blocks
        self.build3 = self.building_block(self.build2, 64, 128, True, 'conv3_1', train_mode)
        self.build4 = self.building_block(self.build3, 128, 128, False, 'conv3_2', train_mode)
        #conv4, 2 blocks
        self.build5 = self.building_block(self.build4, 128, 256, True, 'conv4_1', train_mode)
        self.build6 = self.building_block(self.build5, 256, 256, False, 'conv4_2', train_mode)
        #conv5, 2 blocks
        self.build7 = self.building_block(self.build6, 256, 512, True, 'conv5_1', train_mode)
        self.build8 = self.building_block(self.build7, 512, 512, False, 'conv5_2', train_mode)
        #global avg. pool
        self.pool2 = self.avg_pool(self.build8, 'pool2')
        print("global average pool")
        print(self.pool2.get_shape)
        self.pool2_d1 = tf.reshape(self.pool2, [-1, 512])
        #fully connected layers
        weights = tf.Variable(tf.truncated_normal([512, cf.CLS_NUM], 0.1, 0.001), name = 'logits/fc/weights')
        var_name = 'logits/fc/weights'
        self.save_name_list.append(var_name)
        if (self.ckpt_reader is not None) and (not self.fine_tuning): #reading parametername
            if var_name in self.var_to_shape_map:
                self.restore_name_list.append(var_name)
                print(var_name)
            else:
                print(var_name + " is not found")
                assert False
        biases = tf.Variable(tf.truncated_normal([cf.CLS_NUM], 0.1, 0.001), name = 'logits/fc/biases')
        var_name = 'logits/fc/biases'
        self.save_name_list.append(var_name)
        if (self.ckpt_reader is not None) and (not self.fine_tuning): #reading parametername
            if var_name in self.var_to_shape_map:
                self.restore_name_list.append(var_name)
                print(var_name)
            else:
                print(var_name + " is not found")
                assert False
                
        self.fc8_d1 = tf.nn.bias_add(tf.matmul(self.pool2_d1, weights), biases)
        print("fc out")

        print(self.fc8_d1.get_shape)
        self.prob = tf.nn.softmax(self.fc8_d1, name="resnet/prob")
        
        
        
    def save_model(self, sess, iter):
        print("variables in graph")
        print([n.name for n in tf.get_default_graph().as_graph_def().node])
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
        ckpt_path = temp_saver.save(sess, cf.RES_SAVE_PATH+cf.RES_SAVE_NAME+str(iter)+'.ckpt')
        print("save ckpt file:", ckpt_path)