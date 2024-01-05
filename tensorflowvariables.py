import tensorflow as tf
import numpy as np


tensor_three_d = ([
    [[2,3,4],
    [23,23,12]],
    [[35,98,6],
    [34,2,4]],
    [[9,3,4],
    [4,5,6]]
])

x_reshape = ([
    [3,5,6,6],
    [4,6,1,2]
])

t1 = [[[1, 2, 3], [4, 5, 6]]]
t2 = [[[7, 8, 9], [10, 11, 12]]]

print(tf.constant(t1).shape)
print(tf.constant(t2).shape)
print(tf.concat([t1, t2, t2], 2))

t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[2, 1,], [1, 1]])
print(tf.pad(t, paddings, "CONSTANT"))

params = tf.constant(['p0', 'p1', 'p2', 'p3', 'p4', 'p5'])
print(params[1:3+1])
print(tf.gather(params, tf.range(1,4)))
print(tf.gather(params, [0,5,3]))
print(tf.gather(params, [0]))

indices_1 = [[[0,1],[1,0]],
[[0,0],[1,1]]]
params_1 = [[['a0','b0'],
['c0','d0']],

[['a1','b1'],
['c1', 'd1']]]

print(tf.gather_nd(params_1, indices_1, batch_dims=0))

tensor_two_d = [
    [1,2,0],
    [3,],
    [1,5,6,5,6],
    [2,3]
]
tensor_ragged = tf.ragged.constant(tensor_two_d)
print(tensor_ragged.shape)

tensor_sparse = tf.sparse.SparseTensor(
    [[1,1], [3,4]], [11,56], [5,6]
)
print(tensor_sparse)
tf.sparse.to_dense(tensor_sparse)

tensor_string = tf.constant(["hello", "I am" , "a string"])
print(tf.strings.join(tensor_string, separator=" "))

x= tf.constant([1,2])
x_var = tf.Variable(x)

