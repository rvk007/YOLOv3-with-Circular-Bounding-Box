"""
References
1.
"""

import torch
import argparse
import numpy as np
import torch.nn as nn
import cv2
import math

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
class Detect(nn.Module):
    def __init__(self, anchors):
        super(Detect, self).__init__()
        self.anchors = anchors

def get_config(config_file):
    with open(config_file) as f:
        data = f.read().split('\n')
    
    cfg_list, cfg, curr_dict = [], [], {}
    for d in data:
        if len(d)>0 and d[0]!='#':
            cfg_list.append(d.lstrip().rstrip())
    for c in cfg_list:
        if c[0] == '[':
            if len(curr_dict)>0: 
                cfg.append(curr_dict) 
                curr_dict={}
            curr_dict['type'] = ''.join(list(c)[1:-1])
        else:
            c = c.split('=')
            curr_dict[c[0].rstrip()] = c[1].lstrip()

    cfg.append(curr_dict)
    return cfg

def create_model(cfg):
    input_channel=3
    output_channels = []
    l_modules = nn.ModuleList()

    for idx, b in enumerate(cfg[1:]):

        m = nn.Sequential()
        if b['type'] == 'convolutional':
            k_size = int(b['size'])
            out_channels = int(b['filters'])
            bias = False if 'batch_normalize' in b else True
            conv = nn.Conv2d(
                in_channels=input_channel,
                out_channels=out_channels,
                kernel_size=k_size,
                stride=int(b['stride']),
                padding=(k_size-1)//2,
                bias = bias
            )
            m.add_module(f'conv{idx}',conv)

            if 'batch_normalize' in b:
                batchnorm = nn.BatchNorm2d(out_channels)
                m.add_module(f'bn{idx}',batchnorm)
            
            if b['activation'] == 'leaky':
                activation = nn.LeakyReLU(0.1, inplace=True)
                m.add_module(f'leaky{idx}',activation)
        
        elif b['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
            m.add_module(f'upsample{idx}',upsample)

        elif b['type'] == 'route':
            b['layers'] = b['layers'].split(',')
            b['layers'][0] = int(b['layers'][0])
            layer1 = b['layers'][0]
            if len(b['layers'])==1:
                b['layers'][0] = int(idx+layer1)
                out_channels = output_channels[b['layers'][0]]
            elif len(b['layers'])>1:
                b['layers'][0] = int(idx+layer1)
                b['layers'][1] = int(b['layers'][1])
                out_channels = output_channels[b['layers'][0]] + output_channels[b['layers'][1]]

            m.add_module(f'route{idx}',Identity())

        elif b['type'] == 'shortcut':
            # _from = int(b['from'])
            m.add_module(f'shortcut{idx}',Identity())
        
        elif b['type'] == 'yolo':
            mask = b['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = b['anchors'].split(',')
            anchors = [(int(anchors[i]), int(anchors[i+1])) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            b['anchors'] = anchors
            detect = Detect(anchors)
            m.add_module(f'yolo{idx}',detect)

        l_modules.append(m)
        output_channels.append(out_channels)
        input_channel = out_channels
    
    return cfg[0], l_modules

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.cfg = get_config(cfgfile)
        self.net, self.l_modules = create_model(self.cfg)
        
    def forward(self, x):
        modules = self.cfg[1:]
        # cache the outputs for the route layer
        outputs = {}   
        done = 0     
        for idx, module in enumerate(modules):        
            m_type = (module["type"])
            if m_type == "convolutional" or m_type == "upsample":
                x = self.l_modules[idx](x)
                outputs[idx] = x
                
            elif m_type == "route":
                layers = module["layers"]
                layers = [int(x) for x in layers]

                if len(layers) == 1:
                    x = outputs[layers[0]]
                elif len(layers) > 1:
                    map1 = outputs[layers[0]]
                    map2 = outputs[layers[1]]
                    x = torch.cat((map1,map2),1)
                outputs[idx] = x
                
            elif  m_type == "shortcut":
                _from = int(module["from"])
                x = outputs[idx-1] + outputs[idx+_from]  
                outputs[idx] = x
                
            elif m_type == 'yolo':
                anchors = self.l_modules[idx][0].anchors
                inp_dim = int(self.net["height"])
                n_classes = int(module["classes"])
            
                #ransform 
                x = x.data
                x = self._detection(x,inp_dim,anchors,n_classes)
                
                if not done:
                    detections = x
                    done = 1
                else:       
                    detections = torch.cat((detections, x), 1)

                outputs[idx] = outputs[idx-1]
                
        try:
            return detections
        except:
            return 0

    def _detection(self, x,inp_dim,anchors,n_classes):
        batch_size = x.size(0)
        grid_size = x.size(2)
        stride =  inp_dim // grid_size 
        bbox_attrs = 4 + n_classes 
        num_anchors = len(anchors) 
    
        prediction = x.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
        prediction = prediction.transpose(1,2).contiguous()
        prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

        # the dimension of anchors is wrt original image.We will make it corresponding to feature map
        anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

        #Sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
        prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
        prediction[:,:,3] = torch.sigmoid(prediction[:,:,3])
    
        #Add the center offsets
        grid = np.arange(grid_size)
        a,b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1,1) #(1,gridsize*gridsize,1)
        y_offset = torch.FloatTensor(b).view(-1,1)
        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
        prediction[:,:,:2] += x_y_offset

        #log space transform height and the width
        anchors = torch.FloatTensor(anchors)
        anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
        prediction[:,:,2] = torch.exp(prediction[:,:,2])*anchors #width and height
        prediction[:,:,4: 4 + n_classes] = torch.sigmoid((prediction[:,:, 4 : 4 + n_classes]))    
        prediction[:,:,:3] *= stride 

        return prediction
    
    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        weights = np.fromfile(fp, dtype = np.float32)
        
        p = 0
        for idx in range(len(self.l_modules)):
            m_type = self.cfg[idx + 1]["type"]
            
            if m_type == "convolutional":
                model = self.l_modules[idx]
                try:
                    batch_normalize = int(self.cfg[idx+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                if (batch_normalize):
                    bn = model[1]
                    
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[p:p + num_bn_biases])
                    p += num_bn_biases
                    bn_weights = torch.from_numpy(weights[p: p + num_bn_biases])
                    p  += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[p: p + num_bn_biases])
                    p  += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[p: p + num_bn_biases])
                    p  += num_bn_biases
                    
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[p: p + num_biases])
                    p = p + num_biases
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                    
                # load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[p:p+num_weights])
                p = p + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

def intersected_area(circle1, circle2):
    c1_x, c1_y, c1_r = circle1[:,0], circle1[:,1], circle1[:,2]
    c2_x, c2_y, c2_r = circle2[:,0], circle2[:,1], circle2[:,2]

    hpyt = math.hypot(c2_x - c1_x, c2_y - c1_y)
    if hpyt < c1_r + c2_r:
        a = c1_r*c1_r
        b = c2_r*c2_r
        x = (a - b + hpyt * hpyt) / (2 * hpyt)
        z = x * x
        y = math.sqrt(a - z)
        if hpyt <= abs(c2_r - c1_r) :
            return math.PI * min(a, b)
        return a * math.asin(y / c1_r) + b * math.asin(y / c2_r) - y * (x + math.sqrt(z + b - a))
    return 0

def bounding_box_iou(circle1, circle2):
    # Intersection area
    intersection_area = intersected_area(circle1, circle2)

    # Union Area
    c1_r = circle1[:,2]
    c2_r = circle2[:,2]

    c1_area = math.PI* c1_r * c1_r
    c2_area = math.PI* c2_r * c2_r
    iou = intersection_area / (c1_area + c2_area - intersection_area)
    return iou

def bbox_detection(prediction, conf_threshold, n_classes, nms_conf = 0.4):
    mask = (prediction[:,:,4] > conf_threshold).float().unsqueeze(2) # objectness should be greater than object confidence threshold
    prediction = prediction*mask
    
    # we don't need to transform as center_x, center_y and radius are enough to make a circle bbox
    # # box attributes: (center_x, center_y, height, width)
    # # convert to top_left_x, top_left_y and bottom_right_x, bottom_right_y
    # box_corner = torch.zeros(prediction.shape)
    # box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    # box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    # box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    # box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    # prediction[:,:,:3] = box_corner[:,:,:3]
    
    batch_size = prediction.size(0)
    done = False
    
    # apply Non-Max Suppression
    for ind in range(batch_size):  
        # get all of the possible classes in an image
        img_pred = prediction[ind]
        max_conf, max_conf_score = torch.max(img_pred[:,4:4+ n_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)

        # concat probabilities with box attributes
        combined = (img_pred[:,:4], max_conf, max_conf_score)
        img_pred = torch.cat(combined, 1) 

        # class index will come from those who have objectness > object confidence threshold
        potential_idx =  (torch.nonzero(img_pred[:,3])) 
        img_pred_ = img_pred[potential_idx.squeeze(),:].view(-1,6) # 4(box_attribute + objectness)+2(prob)

        try:
            img_classes = torch.unique(img_pred_[:,-1]) # last index has the predicted class index
        except:
             continue
       
        # NMS
        for _class in img_classes:
            _class_mask = img_pred_*(img_pred_[:,-1] == _class).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(_class_mask[:,-2]).squeeze()
            img_pred_class = img_pred_[class_mask_ind].view(-1,6)
            
            # sort them by highest probability 
            conf_sort_index = torch.sort(img_pred_class[:,4], descending = True )[1]
            img_pred_class = img_pred_class[conf_sort_index]
            idx = img_pred_class.size(0)
            
            for i in range(idx):
                try:
                    iou = bounding_box_iou(img_pred_class[i].unsqueeze(0), img_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break
                
                # if iou < NMS threshold
                iou_mask = (iou < nms_conf).float().unsqueeze(1)
                img_pred_class[i+1:] *= iou_mask
                non_zero_ind = torch.nonzero(img_pred_class[:,3]).squeeze()
                img_pred_class = img_pred_class[non_zero_ind].view(-1,6)
            
            #creating a row with index of images
            batch_idx = img_pred_class.new(img_pred_class.size(0), 1).fill_(ind)
            seq = batch_idx, img_pred_class
            if not done:
                output = torch.cat(seq,1)
                done = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output

def preprocess_image(img, inp_dim):
    # convert input image to tensor
    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0] # width, height
    img = resize_img(orig_im, (inp_dim, inp_dim))
    img = img[:,:,::-1]
    # transpose to get channel first
    img = img.transpose((2,0,1)).copy() # copy() to avoid negative stride in the array  
    img = torch.from_numpy(img)/255
    img = img.unsqueeze(0)
    return img, orig_im, dim

def resize_img(img, conf_inp_h):
    # resize image without altering aspect ratio
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = conf_inp_h # 608, 608

    # to maintain the aspect ratio
    ratio = min(w/img_w, h/img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    # fill the extra pixels with 128
    n_img = np.full((conf_inp_h[1], conf_inp_h[0], 3), 128)
    n_img[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,:] = resized_image
    return n_img

def draw_boxes(x, img, classes):
    c1 = tuple(x[1:3])
    c2 = tuple(x[3:5])
    class_name = int(x[-1])
    label = "{0}".format(classes[class_name])
    color = (0,0,255)
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])),color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])) ,color, -1)
    cv2.putText(img, label, (int(c1[0]), int(c1[1]) + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

def load_classes(class_file):
    fp = open(class_file, "r")
    names = fp.read().split("\n")
    return names

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', type=str, default='yolov3.cfg')
    parse.add_argument('--img', type=str)
    parse.add_argument('--nmsthresh', type=float, default=0.5)
    parse.add_argument('--confthresh', type=float, default=0.5)

    weightsfile = 'yolov3.weights'
    classfile = 'classname.txt'
    opt = parse.parse_args()

    # get the model
    model = Darknet(opt.config)
    model.load_weights(weightsfile)

    # preprocess input image
    conf_inp_h = int(model.net["height"])
    processed_image, original_image, original_img_dim = preprocess_image(opt.img,conf_inp_h)
    im_dim = original_img_dim[0], original_img_dim[1]
    im_dim = torch.FloatTensor(im_dim).repeat(1,2)

    # make a prediction on the input image
    model.eval()
    with torch.no_grad():
        pred = model(processed_image)
    classes = load_classes(classfile)
    n_classes = len(classes)
    output = bbox_detection(pred, conf_threshold=opt.confthresh, n_classes=n_classes, nms_conf = opt.nmsthresh)

    # need real data to visualize circular BBOX
  