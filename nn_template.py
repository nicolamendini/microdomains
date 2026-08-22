import torch
import torch.nn as nn

## general network class, where the network is simply defined by a list of layers
class Network(nn.Module):
    
    def __init__(self, h, device, bias=False):
        
        super(Network, self).__init__()
        
        # initialising the layers dictionary and storing architecture
        self.layers = {}
        self.h = h
        
        # filling in the layers dictionary
        for i in range(len(h)):
            
            layer_type = h[i][0]
                
            if layer_type=='dense':
                self.layers[i] = nn.Linear(h[i][2], h[i][1], bias=bias, device=device)
                
            elif layer_type=='conv2d':
                self.layers[i] = nn.Conv2d(
                    h[i-1][1], 
                    h[i][1], 
                    h[i][2], 
                    padding=h[i][2]//2, 
                    bias=bias,
                    device=device
                )
                
            elif layer_type=='pool2d':
                self.layers[i] = nn.MaxPool2d(h[i][2])
                
            elif layer_type=='upsample':
                self.layers[i] = nn.Upsample(scale_factor=h[i][2])
                
            elif layer_type=='flatten':
                self.layers[i] = nn.Flatten()
                
            elif layer_type=='unflatten':
                self.layers[i]= nn.Unflatten(1, (h[i][1], h[i][2], h[i][2]))
                
            elif layer_type=='relu':
                self.layers[i] = nn.ReLU()

            elif layer_type=='tanh':
                self.layers[i] = nn.Tanh()
                
        #print(self.layers.keys())    
                        
    def forward(self, x_b, debug=False):

        act = self.layers[0](x_b)
        
        # applying layer after layer
        for i in range(1, len(self.h)):
            
            if debug:
                print(i, act.shape)
                
            act = self.layers[i](act)
            
        return act

