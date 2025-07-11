#from modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
import os
import torch.nn as nn
import shutil
import numpy as np
import torch
import pytorch_quantization.cim.modules.macro as macro



def dec2bin(x, num_bits):
    x = x.int()

    output = torch.zeros(x.shape[0], x.shape[1], num_bits, device=x.device)
    
    for i in range(num_bits):
        bit = (x >> (num_bits - 1 - i)) & 1
        output[:,:, i] = bit
    
    output = output.view(x.shape[0], x.shape[1]*num_bits) 
    return output   

def dec2val(x, num_bits, cells_per_weight, bitcell, base):
    x = x.int()

    output = torch.zeros(x.shape[0], x.shape[1], num_bits, device=x.device)
    
    for i in range(cells_per_weight):
        val = (x >> bitcell*i) & (base-1)
        output[:,:, i] = val
    
    output = output.view(x.shape[0], x.shape[1]*num_bits)  

    return output

def write_layer(self, input2d, weight2d):
    # inputs and weights are in integer format

    rows_per_input = input2d.shape[0] // self.batch_size
    input_bin = dec2bin(input2d[0:rows_per_input], self._cim_args.input_precision)
    cell_value = dec2val(weight2d, self._cim_args.weight_precision, self._cim_args.bitcell)

    input_file_name =  './layer_record_' + self._cim_args.model + '/input' + str(self.name) + '.csv'
    weight_file_name =  './layer_record_' + self._cim_args.model + '/weight' + str(self.name) + '.csv'
    f = open('./layer_record_' + self._cim_args.model + '/trace_command.sh', "a")
    f.write(weight_file_name+' '+input_file_name+' ')

    np.savetxt(input_file_name, input_bin, delimiter=",",fmt='%s')
    np.savetxt(weight_file_name, cell_value, delimiter=",",fmt='%s')


def make_records(args):
    # create layer record directory
    if not os.path.exists('./layer_record_'+str(args.model)):
        os.makedirs('./layer_record_'+str(args.model))

    if os.path.exists('./layer_record_'+str(args.model)+'/trace_command.sh'):
        os.remove('./layer_record_'+str(args.model)+'/trace_command.sh')

    f = open('./layer_record_'+str(args.model)+'/trace_command.sh', "w")
    f.write('./NeuroSIM/main ./NeuroSIM/NetWork_'+str(args.model)+'.csv '+str(args.weight_precision)+' '+str(args.input_precision)+' '+str(args.sub_array[0])+' '+str(args.parallel_read)+' ')
    f.close()

    # remove existing NetWork.csv
    if os.path.exists('./NeuroSIM/NetWork_'+str(args.model)+'.csv'):
        os.remove('./NeuroSIM/NetWork_'+str(args.model)+'.csv')  

# def Neural_Sim(self, input, output): 
#     global model_n

#     print("quantize layer ", self.name)
#     input_file_name =  './layer_record_' + str(model_n) + '/input' + str(self.name) + '.csv'
#     weight_file_name =  './layer_record_' + str(model_n) + '/weight' + str(self.name) + '.csv'
#     f = open('./layer_record_' + str(model_n) + '/trace_command.sh', "a")
#     f.write(weight_file_name+' '+input_file_name+' ')
#     if FP:
#         weight_q = float_quantizer.float_range_quantize(self.weight,self.wl_weight)
#     else:
#         weight_q = wage_quantizer.Q(self.weight,self.wl_weight)
#     write_matrix_weight( weight_q.cpu().data.numpy(),weight_file_name)
#     if len(self.weight.shape) > 2:
#         k=self.weight.shape[-1]
#         padding = self.padding
#         stride = self.stride  
#         write_matrix_activation_conv(stretch_input(input[0].cpu().data.numpy(),k,padding,stride),None,self.wl_input,input_file_name)
#     else:
#         write_matrix_activation_fc(input[0].cpu().data.numpy(),None ,self.wl_input, input_file_name)

# def write_matrix_weight(input_matrix,filename):
#     cout = input_matrix.shape[0]
#     weight_matrix = input_matrix.reshape(cout,-1).transpose()
#     np.savetxt(filename, weight_matrix, delimiter=",",fmt='%10.5f')


# def write_matrix_activation_conv(input_matrix,fill_dimension,length,filename):
#     filled_matrix_b = np.zeros([input_matrix.shape[2],input_matrix.shape[1]*length],dtype=str)
#     filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
#     for i,b in enumerate(filled_matrix_bin):
#         filled_matrix_b[:,i::length] =  b.transpose()
#     np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


# def write_matrix_activation_fc(input_matrix,fill_dimension,length,filename):

#     filled_matrix_b = np.zeros([input_matrix.shape[1],length],dtype=str)
#     filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
#     for i,b in enumerate(filled_matrix_bin):
#         filled_matrix_b[:,i] =  b
#     np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


# def stretch_input(input_matrix,window_size = 5,padding=(0,0),stride=(1,1)):
#     input_shape = input_matrix.shape
#     output_shape_row = int((input_shape[2] + 2*padding[0] -window_size) / stride[0] + 1)
#     output_shape_col = int((input_shape[3] + 2*padding[1] -window_size) / stride[1] + 1)
#     item_num = int(output_shape_row * output_shape_col)
#     output_matrix = np.zeros((input_shape[0],item_num,input_shape[1]*window_size*window_size))
#     iter = 0
#     if (padding[0] != 0):
#         input_tmp = np.zeros((input_shape[0], input_shape[1], input_shape[2] + padding[0]*2, input_shape[3] + padding[1] *2))
#         input_tmp[:, :, padding[0]: -padding[0], padding[1]: -padding[1]] = input_matrix
#         input_matrix = input_tmp
#     for i in range(output_shape_row):
#         for j in range(output_shape_col):
#             for b in range(input_shape[0]):
#                 output_matrix[b,iter,:] = input_matrix[b, :, i*stride[0]:i*stride[0]+window_size,j*stride[1]:j*stride[1]+window_size].reshape(input_shape[1]*window_size*window_size)
#             iter += 1

#     return output_matrix


# def dec2bin(x,n):
#     y = x.copy()
#     out = []
#     scale_list = []
#     delta = 1.0/(2**(n-1))
#     x_int = x/delta

#     base = 2**(n-1)

#     y[x_int>=0] = 0
#     y[x_int< 0] = 1
#     rest = x_int + base*y
#     out.append(y.copy())
#     scale_list.append(-base*delta)
#     for i in range(n-1):
#         base = base/2
#         y[rest>=base] = 1
#         y[rest<base]  = 0
#         rest = rest - base * y
#         out.append(y.copy())
#         scale_list.append(base * delta)

#     return out,scale_list

# def bin2dec(x,n):
#     bit = x.pop(0)
#     base = 2**(n-1)
#     delta = 1.0/(2**(n-1))
#     y = -bit*base
#     base = base/2
#     for bit in x:
#         y = y+base*bit
#         base= base/2
#     out = y*delta
#     return out

# def remove_hook_list(hook_handle_list):
#     for handle in hook_handle_list:
#         handle.remove()

# def hardware_evaluation(model, args): 
#     global model_n
#     model_n = args.model
    
#     hook_handle_list = []
#     if not os.path.exists('./layer_record_'+str(args.model)):
#         os.makedirs('./layer_record_'+str(args.model))
#     if os.path.exists('./layer_record_'+str(args.model)+'/trace_command.sh'):
#         os.remove('./layer_record_'+str(args.model)+'/trace_command.sh')
#     f = open('./layer_record_'+str(args.model)+'/trace_command.sh', "w")
#     f.write('./NeuroSIM/main ./NeuroSIM/NetWork_'+str(args.model)+'.csv '+str(args.weight_precision)+' '+str(args.input_precision)+' '+str(args.sub_array)+' '+str(args.parallel_read)+' ')
    
#     for i, layer in enumerate(model.modules()):
#         if isinstance(layer, cim_conv.CIMConv2d, cim_linear.CIMLinear):
#             hook_handle_list.append(layer.register_forward_hook(Neural_Sim))
#     return hook_handle_list
