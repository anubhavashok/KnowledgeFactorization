import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
 
def replace_layers(model, i, indexes, layers):
	if i in indexes:
		return layers[indexes.index(i)]
	return model[i]

def prune_vgg16_conv_layer(model, layer_index, filter_index):
	_, conv = model.features._modules.items()[layer_index]
	next_conv = None
	offset = 1
        bn_offset = 1
        
        while (layer_index + bn_offset) <  len(model.features._modules.items()):
                res =  model.features._modules.items()[layer_index+bn_offset]
                if isinstance(res[1], torch.nn.modules.conv.Conv2d):
                    # If we reach a conv first, theres no bn for this layer, so quit
                    bn_offset = len(model.features._modules.items()) + 1
                    break
                if isinstance(res[1], torch.nn.modules.BatchNorm2d):
                    break
                bn_offset = bn_offset + 1
        next_bn = (layer_index + bn_offset) < len(model.features._modules.items()) 
	while layer_index + offset <  len(model.features._modules.items()):
		res =  model.features._modules.items()[layer_index+offset]
		if isinstance(res[1], torch.nn.modules.conv.Conv2d):
			next_name, next_conv = res
			break
		offset = offset + 1
        if next_bn:
            _, bn_old = model.features._modules.items()[layer_index+bn_offset]
            bn = torch.nn.BatchNorm2d(bn_old.num_features-1)
            old_w = bn_old.weight.data.cpu().numpy()
            old_b = bn_old.bias.data.cpu().numpy()
            old_m = bn_old.running_mean.cpu().numpy()
            old_v = bn_old.running_var.cpu().numpy()
            new_w = bn.weight.data.cpu().numpy()
            new_b = bn.bias.data.cpu().numpy()
            new_m = bn.running_mean.cpu().numpy()
            new_v = bn.running_var.cpu().numpy()
            new_w[: filter_index] = old_w[: filter_index]; new_w[filter_index :] = old_w[filter_index+1 :] 
            new_b[: filter_index] = old_b[: filter_index]; new_b[filter_index :] = old_b[filter_index+1 :] 
            new_m[: filter_index] = old_m[: filter_index]; new_m[filter_index :] = old_m[filter_index+1 :] 
            new_v[: filter_index] = old_v[: filter_index]; new_v[filter_index :] = old_v[filter_index+1 :] 
            bn.weight.data = torch.from_numpy(new_w)
            bn.bias.data = torch.from_numpy(new_b)
            bn.running_mean = torch.from_numpy(new_m)
            bn.running_var = torch.from_numpy(new_v)

	def is_bias(x):
		print x
		if x is None:
			return False
		else:
			return True

	#print conv.bias

	new_conv = \
		torch.nn.Conv2d(in_channels = conv.in_channels, \
			out_channels = conv.out_channels - 1,
			kernel_size = conv.kernel_size, \
			stride = conv.stride,
			padding = conv.padding,
			dilation = conv.dilation,
			groups = conv.groups,
			bias = is_bias(conv.bias))

	old_weights = conv.weight.data.cpu().numpy()
	new_weights = new_conv.weight.data.cpu().numpy()

	new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
	new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
	new_conv.weight.data = torch.from_numpy(new_weights)#.cuda()

	bias_numpy = conv.bias.data.cpu().numpy()

	bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
	bias[:filter_index] = bias_numpy[:filter_index]
	bias[filter_index : ] = bias_numpy[filter_index + 1 :]
	new_conv.bias.data = torch.from_numpy(bias)#.cuda()

	if not next_conv is None:
		#print conv.bias
		#print model

		next_new_conv = \
			torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
				out_channels =  next_conv.out_channels, \
				kernel_size = next_conv.kernel_size, \
				stride = next_conv.stride,
				padding = next_conv.padding,
				dilation = next_conv.dilation,
				groups = next_conv.groups,
				bias = is_bias(conv.bias))

		old_weights = next_conv.weight.data.cpu().numpy()
		new_weights = next_new_conv.weight.data.cpu().numpy()

		new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
		new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
		next_new_conv.weight.data = torch.from_numpy(new_weights)#.cuda()

		next_new_conv.bias.data = next_conv.bias.data

	if not next_conv is None and next_bn:
	 	features = torch.nn.Sequential(
	            *(replace_layers(model.features, i, [layer_index, layer_index+bn_offset, layer_index+offset], \
	            	[new_conv, bn, next_new_conv]) for i, _ in enumerate(model.features)))
	 	del model.features
	 	del conv
	 	model.features = features
        elif not next_conv is None and not next_bn:
                features = torch.nn.Sequential(
                    *(replace_layers(model.features, i, [layer_index,  layer_index+offset], \
                        [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
                del model.features
                del conv
                model.features = features
	else:
		#Prunning the last conv layer. This affects the first linear layer of the classifier.
                if next_bn:
                    model.features = torch.nn.Sequential(
	                *(replace_layers(model.features, i, [layer_index, layer_index+bn_offset], \
	                    [new_conv, bn]) for i, _ in enumerate(model.features)))
                else:
                    model.features = torch.nn.Sequential(
                        *(replace_layers(model.features, i, [layer_index], \
                            [new_conv]) for i, _ in enumerate(model.features)))
	 	layer_index = 0
	 	old_linear_layer = None
	 	for _, module in model.classifier._modules.items():
	 		if isinstance(module, torch.nn.Linear):
	 			old_linear_layer = module
	 			break
	 		layer_index = layer_index  + 1

	 	if old_linear_layer is None:
	 		raise BaseException("No linear laye found in classifier")
		params_per_input_channel = old_linear_layer.in_features / conv.out_channels

	 	new_linear_layer = \
	 		torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel, 
	 			old_linear_layer.out_features)
	 	
	 	old_weights = old_linear_layer.weight.data.cpu().numpy()
	 	new_weights = new_linear_layer.weight.data.cpu().numpy()	 	

	 	new_weights[:, : filter_index * params_per_input_channel] = \
	 		old_weights[:, : filter_index * params_per_input_channel]
	 	new_weights[:, filter_index * params_per_input_channel :] = \
	 		old_weights[:, (filter_index + 1) * params_per_input_channel :]
	 	
	 	new_linear_layer.bias.data = old_linear_layer.bias.data

	 	new_linear_layer.weight.data = torch.from_numpy(new_weights)#.cuda()

		classifier = torch.nn.Sequential(
			*(replace_layers(model.classifier, i, [layer_index], \
				[new_linear_layer]) for i, _ in enumerate(model.classifier)))

		del model.classifier
		del next_conv
		del conv
		model.classifier = classifier
	print model
	return model

if __name__ == '__main__':
	model = models.vgg16(pretrained=True)
	model.train()

	t0 = time.time()
	model = prune_conv_layer(model, 28, 10)
	print "The prunning took", time.time() - t0