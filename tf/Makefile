# https://www.tensorflow.org/extend/adding_an_op
# fixme: how impl. gradient? only in Python
TF_INC=`python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`


all:
	echo $(TF_INC)
	g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $(TF_INC) -O2