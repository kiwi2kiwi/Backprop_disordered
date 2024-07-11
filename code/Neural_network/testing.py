import numpy as np

def activation_function(z):
    return (1. / (1 + np.exp(-z)))


def deri_activation_function():
    return activation_function() * (1 - activation_function())

lr = 0.5

x_train = [[0.05,0.1],[0.1,0.05]]
y_train = [[0.01,0.99],[0.99,0.01]]

w1 = 0.15
w2 = 0.2
w3 = 0.25
w4 = 0.3
w5 = 0.4
w6 = 0.45
w7 = 0.5
w8 = 0.55

b1 = 0.35
b2 = 0.60

# inputs
i1 = x_train[0][0]
i2 = x_train[0][1]

o1_out = 0
o2_out = 0

def forward():
    # hiddens
    h1_in = i1 * w1 + i2 * w2 + b1
    h1_out = activation_function(h1_in)
    h2_in = i1 * w3 + i2 * w4 + b1
    h2_out = activation_function(h2_in)

    # outputs
    o1_in = h1_out * w5 + h2_out * w6 + b2
    o1_out = activation_function(o1_in)
    o2_in = h1_out * w7 + h2_out * w8 + b2
    o2_out = activation_function(o2_in)



# Errors
o1_err = .5*((y_train[0][0] - o1_out)**2)
o2_err = .5*((y_train[0][1] - o2_out)**2)

# derivations
delta_Error_through_delta_o1_out = (o1_out - y_train[0][0])
delta_Error_through_delta_o2_out = (o2_out - y_train[0][1])

delta_o1_out_through_delta_o1_in = o1_out*(1-o1_out)
delta_o2_out_through_delta_o2_in = o2_out*(1-o2_out)

delta_o1_in_through_delta_w5 = h1_out
delta_o1_in_through_delta_w6 = h2_out
delta_o2_in_through_delta_w7 = h1_out
delta_o2_in_through_delta_w8 = h2_out

delta_Error_o1_through_delta_w5 = delta_Error_through_delta_o1_out * delta_o1_out_through_delta_o1_in * delta_o1_in_through_delta_w5
delta_Error_o1_through_delta_w6 = delta_Error_through_delta_o1_out * delta_o1_out_through_delta_o1_in * delta_o1_in_through_delta_w6
delta_Error_o2_through_delta_w7 = delta_Error_through_delta_o2_out * delta_o2_out_through_delta_o2_in * delta_o2_in_through_delta_w7
delta_Error_o2_through_delta_w8 = delta_Error_through_delta_o2_out * delta_o2_out_through_delta_o2_in * delta_o2_in_through_delta_w8

# new weights
w5_new = w5 - lr * delta_Error_o1_through_delta_w5
w6_new = w6 - lr * delta_Error_o1_through_delta_w6
w7_new = w7 - lr * delta_Error_o2_through_delta_w7
w8_new = w8 - lr * delta_Error_o2_through_delta_w8

delta_Error_o1_through_delta_o1_in = delta_Error_through_delta_o1_out * delta_o1_out_through_delta_o1_in
delta_Error_o2_through_delta_o2_in = delta_Error_through_delta_o2_out * delta_o2_out_through_delta_o2_in

delta_o1_in_through_h1_out = w5
delta_o2_in_through_h1_out = w7


delta_Error_o1_through_delta_h1_out = delta_Error_o1_through_delta_o1_in * delta_o1_in_through_h1_out
delta_Error_o2_through_delta_h1_out = delta_Error_o2_through_delta_o2_in * delta_o2_in_through_h1_out


delta_Error_total_through_delta_h1_out = delta_Error_o1_through_delta_h1_out + delta_Error_o2_through_delta_h1_out
delta_Error_total_through_delta_h2_out = delta_Error_o1_through_delta_h1_out + delta_Error_o2_through_delta_h1_out

delta_h1_out_through_delta_h1_in = h1_out*(1-h1_out)
delta_h2_out_through_delta_h2_in = h2_out*(1-h2_out)

delta_h1_in_through_delta_w1 = i1
delta_h1_in_through_delta_w2 = i2
delta_h1_in_through_delta_w3 = i1
delta_h1_in_through_delta_w4 = i2

delta_Error_total_through_delta_w1 = delta_Error_total_through_delta_h1_out * delta_h1_out_through_delta_h1_in * delta_h1_in_through_delta_w1
delta_Error_total_through_delta_w2 = delta_Error_total_through_delta_h1_out * delta_h2_out_through_delta_h2_in * delta_h1_in_through_delta_w2
delta_Error_total_through_delta_w3 = delta_Error_total_through_delta_h2_out * delta_h1_out_through_delta_h1_in * delta_h1_in_through_delta_w3
delta_Error_total_through_delta_w4 = delta_Error_total_through_delta_h2_out * delta_h2_out_through_delta_h2_in * delta_h1_in_through_delta_w4

# new weights
w1_new = w1 - lr * delta_Error_total_through_delta_w1
w2_new = w2 - lr * delta_Error_total_through_delta_w2
w3_new = w3 - lr * delta_Error_total_through_delta_w3
w4_new = w4 - lr * delta_Error_total_through_delta_w4



print("end")

