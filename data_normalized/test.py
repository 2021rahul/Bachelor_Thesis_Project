import theano
import numpy
layer1_w = theano.shared(value=numpy.zeros((1,10)),borrow = True)
layer1_b = theano.shared(value=numpy.zeros((1,10)),borrow = True)
layer1_params = [layer1_w,layer1_b]
layer2_w = theano.shared(value=numpy.zeros((1,10)),borrow = True)
layer2_b = theano.shared(value=numpy.zeros((1,10)),borrow = True)
layer2_params = [layer2_w,layer2_b]
params = layer1_params + layer2_params
for param_i in params:
	print param_i.get_value()  


new_layer1_w = theano.shared(value=numpy.ones((1,10)),borrow = True)
new_layer1_b = theano.shared(value=numpy.ones((1,10)),borrow = True)
new_layer1_params = [new_layer1_w,new_layer1_b]
new_layer2_w = theano.shared(value=numpy.ones((1,10)),borrow = True)
new_layer2_b = theano.shared(value=numpy.ones((1,10)),borrow = True)
new_layer2_params = [new_layer2_w,new_layer2_b]
new_params = new_layer1_params + new_layer2_params
for param_i in new_params:
	print param_i.get_value()  

updates = [
    (param_i, new_param_i)
    for param_i, new_param_i in zip(params, new_params)
] 

for update_i in updates:
	print update_i[0].get_value()
	print update_i[1].get_value()

train_model = theano.function(
    [],
    updates=updates,
    allow_input_downcast=True
)

train_model()
for param_i in params:
	print param_i.get_value()  
