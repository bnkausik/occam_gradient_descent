############################################################
#occam gradient descent for CIFAR10
# BN Kausik May 22 2024
############################################################






import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import sys

from termcolor import colored
def my_print(s):
    return print(colored(s,"black"))

#https://www.tensorflow.org/tutorials/images/cnn

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

N_units=64     #model scale
n_epochs=12     #number of epochs
v_flag=0        #verbose flag
occam_flag=1    #enable/disable occam
p_flag=1        #plot flag
lr =0.4         #initial learning rate
t_frac=0.0      #holdback fraction

opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
for i in range (0,len(opts)):
    if opts[i]=="-p": p_flag=int(args[i]);
    if opts[i]=="-v": v_flag=int(args[i]);
    if opts[i]=="-o": occam_flag=int(args[i]);
    if opts[i]=="-n_epochs": n_epochs=int(args[i]);
    if opts[i]=="-N_units": N_units=int(args[i]);
    if opts[i]=="-lr": lr=float(args[i]);
    if opts[i]=="-tf": t_frac=float(args[i]);

if occam_flag==0: t_frac=0.0
print("N_units",N_units,"n_epochs",n_epochs);
print("v_flag",v_flag,"occam_flag",occam_flag,"p_flag",p_flag);
print("lr",lr)
print("t_frac",t_frac)

"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
"""
class quant_0(tf.keras.constraints.Constraint):
     def __call__(self, w):
         if occam_flag:return tf.where(Q_0 ==0.0,0.0,w)
         else: return w

class quant_2(tf.keras.constraints.Constraint):
     def __call__(self, w):
         if occam_flag:return tf.where(Q_2 ==0.0,0.0,w)
         else: return w

class quant_4(tf.keras.constraints.Constraint):
     def __call__(self, w):
         if occam_flag:return tf.where(Q_4 ==0.0,0.0,w)
         else: return w

class quant_6(tf.keras.constraints.Constraint):
     def __call__(self, w):
         if occam_flag:return tf.where(Q_6 ==0.0,0.0,w)
         else: return w

class quant_7(tf.keras.constraints.Constraint):
     def __call__(self, w):
         if occam_flag:return tf.where(Q_7 ==0.0,0.0,w)
         else: return w


model = models.Sequential()
model.add(layers.Conv2D(N_units, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_constraint=quant_0()))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(2*N_units, (3, 3), activation='relu', kernel_constraint=quant_2()))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(2*N_units, (3, 3), activation='relu', kernel_constraint=quant_4()))
model.add(layers.Flatten())
model.add(layers.Dense(2*N_units, activation='relu', kernel_constraint=quant_6()))
model.add(layers.Dense(10, kernel_constraint=quant_7()))

model.summary(print_fn=my_print)

W_0=model.layers[0].get_weights()[0]
Q_0=tf.Variable(np.ones(W_0.shape))
W_2=model.layers[2].get_weights()[0]
Q_2=tf.Variable(np.ones(W_2.shape))
W_4=model.layers[4].get_weights()[0]
Q_4=tf.Variable(np.ones(W_4.shape))
W_6=model.layers[6].get_weights()[0]
Q_6=tf.Variable(np.ones(W_6.shape))
W_7=model.layers[7].get_weights()[0]
Q_7=tf.Variable(np.ones(W_7.shape))
print("shapes", W_0.shape,W_2.shape,W_4.shape)
N_bias=0
for i in [0,2,4,6,7]:
    N_bias+=(model.layers[i].get_weights()[1]).size
max_N=W_0.size+W_2.size+W_4.size+W_6.size+W_7.size+N_bias
print("total_parameters", max_N)


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

lr_0=lr
err_log=np.zeros((2,n_epochs,2))
wts=np.zeros(n_epochs,dtype=int)

c_log=np.zeros((n_epochs,2))
t_span=len(train_labels)
if t_frac>0:
    t_span=int(t_span*(1.0-t_frac))


for i in range(n_epochs):
    model.fit(train_images[0:t_span], train_labels[0:t_span], epochs=1,verbose=v_flag)
    err_log[0,i]=model.evaluate(train_images[0:t_span], train_labels[0:t_span], verbose=v_flag)
    err_log[1,i]=model.evaluate(test_images, test_labels, verbose=v_flag)
    if t_frac>0:
        c_log[i]=model.evaluate(train_images[t_span:], train_labels[t_span:], verbose=v_flag)
    else:
        c_log[i,:]=err_log[0,i,:]
    if occam_flag:
        wts[i]=N_bias
        wts[i]+=int(tf.math.reduce_sum(Q_0).numpy())
        wts[i]+=int(tf.math.reduce_sum(Q_2).numpy())
        wts[i]+=int(tf.math.reduce_sum(Q_4).numpy())
        wts[i]+=int(tf.math.reduce_sum(Q_6).numpy())
        wts[i]+=int(tf.math.reduce_sum(Q_7).numpy())
        if i>1:
            zeta=(c_log[i,0]-c_log[i-1,0])/(c_log[i-1,0]-c_log[i-2,0])
            print("zeta",zeta)
            lr=np.clip(zeta*lr,0.1*lr_0,lr_0)
        W_t=np.abs(model.layers[0].get_weights()[0])
        q=np.quantile(W_t[W_t!=0],lr)
        W_t=np.where(W_t<q,0,1)
        Q_0.assign(W_t)

        W_t=np.abs(model.layers[2].get_weights()[0])
        q=np.quantile(W_t[W_t!=0],lr)
        W_t=np.where(W_t<q,0,1)
        Q_2.assign(W_t)

        W_t=np.abs(model.layers[4].get_weights()[0])
        q=np.quantile(W_t[W_t!=0],lr)
        W_t=np.where(W_t<q,0,1)
        Q_4.assign(W_t)

        W_t=np.abs(model.layers[6].get_weights()[0])
        q=np.quantile(W_t[W_t!=0],lr)
        W_t=np.where(W_t<q,0,1)
        Q_6.assign(W_t)

        W_t=np.abs(model.layers[7].get_weights()[0])
        q=np.quantile(W_t[W_t!=0],lr)
        W_t=np.where(W_t<q,0,1)
        Q_7.assign(W_t)


    else: wts[i]=max_N
    print("#### ",i,wts[i],max_N,err_log[0,i,0],err_log[1,i,0],err_log[0,i,1],err_log[1,i,1],c_log[i,0],c_log[i,1],lr)

opt_epoch=err_log[1,:,0].argmin()
opt_N=wts[opt_epoch]
if occam_flag: print("opt_epoch",opt_epoch,"opt_N",opt_N)



if p_flag:
    plt.figure(1)
    plt.plot(err_log[0,:,0], label='train_loss',marker="o")
    plt.plot(err_log[1,:,0], label='test_loss',marker="s")
    plt.text(0.03, 0.85, "loss: " +str(N_units)+" units,"+"{:.1f}".format(100*opt_N/max_N)+"% opt      wts" ,transform=plt.gca().transAxes,  bbox=dict(facecolor='white', alpha=0.5))
    plt.figure(2)
    plt.plot(err_log[0,:,1], label='train_acc',marker="o")
    plt.plot(err_log[1,:,1], label='test_acc',marker="s")
    plt.text(0.03, 0.85, "accuracy: " +str(N_units)+" units,"+"{:.1f}".format(100*opt_N/max_N)+"% opt  wts" ,transform=plt.gca().transAxes,  bbox=dict(facecolor='white', alpha=0.5))
    plt.show();



