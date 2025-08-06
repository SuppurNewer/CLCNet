import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

class CA_attention(nn.Module):
    def __init__(self, channel, h, w, reduction=4):
        super().__init__()
        self.h = h
        self.w = w
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)
    
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(1, 0, 2, 3) 
        out = self.global_pooling(x).view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).view(x.size(0), -1, 1, 1)
        out = out * x
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride,dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation)
        self.BN = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.BN(x)
        x = self.pool(x)
        return x

class MultiScaleBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size=[1,3,5,7],
                 dilation=[1,2,3,3]):
        super().__init__()
        
        self.conv_block_1 = ConvBlock(in_channels, out_channels, kernel_size=kernel_size[0], padding=1, stride=2,dilation=dilation[0])
        self.conv_block_2 = ConvBlock(in_channels, out_channels, kernel_size=kernel_size[1], padding=1, stride=2,dilation=dilation[1])
        self.conv_block_3 = ConvBlock(in_channels, out_channels, kernel_size=kernel_size[2], padding=1, stride=2,dilation=dilation[2])
        self.conv_block_4 = ConvBlock(in_channels, out_channels, kernel_size=kernel_size[3], padding=1, stride=2,dilation=dilation[3])

        self.concat = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=11, padding=1, stride=1,
                               dilation=1)
        
        self.conv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels*4,
                               kernel_size=3, padding=1, stride=2,
                               dilation=1)
   
    def forward(self, x):
        out1 = F.relu(self.conv_block_1(x))
        out2 = F.relu(self.conv_block_2(x))
        out3 = F.relu(self.conv_block_3(x))
        out4 = F.relu(self.conv_block_4(x))

        out5 = F.relu(self.concat(x))
        # print(out5.shape)
        
        concat_out = torch.cat((out1, out2, out3, out4), dim=2) + out5 
        concat_out = self.conv(concat_out)
        return concat_out
        # return concat_out
        

class AttentionBlock(nn.Module):
    def __init__(self,
                 d_model=16, 
                 nhead=4, 
                 num_layers=2,
                 dim_feedforward=64, 
                 dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, num_channels]
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)  # [batch_size, num_channels, seq_len]
        return x
    
class GenomicCNN(nn.Module):
    def __init__(self, 
                 num_chroms = 10,
                 in_channels = 3, 
                 out_channels = 32,
                 embedding_size = 128,
                 dropout_1 = 0.2,
                 dropout_2 = 0.2,
                 num_layers = 1,
                 num_fc = 128,
                 kernel_size_res=[1,3,5,7],
                 ):
        super().__init__()
        self.num_chroms = num_chroms
        self.res_blocks_global = MultiScaleBlock(in_channels, out_channels, kernel_size_res)
        self.res_blocks_local = nn.ModuleList([MultiScaleBlock(in_channels, out_channels, kernel_size_res) for _ in range(num_chroms)])
        self.conv = ConvBlock(in_channels=out_channels*4, out_channels=out_channels*8,
                  kernel_size=3, padding=1, stride=1,dilation=1)
        self.attention_blocks =AttentionBlock(d_model = out_channels*4,nhead=4,num_layers=num_layers,dim_feedforward = out_channels*8, dropout=dropout_2) 
        self.fc1_embedding = nn.Linear(5603*64, embedding_size*2)
        self.fc2_embedding = nn.Linear(embedding_size*2, embedding_size)

        self.fc1_primary = nn.Linear(5603*64, num_fc*2)
        self.dropout = nn.Dropout(dropout_1)
        self.fc2_primary = nn.Linear(num_fc*2, 1)

    def forward(self, x):
        res_global = self.res_blocks_global(x[0])
        res_local = [conv_block(x[i+1]) for i, conv_block in enumerate(self.res_blocks_local)]
        res_outs_local= torch.stack(res_local, dim=1)
        
        res_outs_flat = res_outs_local.view(res_outs_local.shape[0], res_outs_local.shape[2], -1)
        
        combined = torch.cat((res_global, res_outs_flat), dim=2)
        combined = self.attention_blocks(combined)
        combined = self.conv(combined)
        # print(combined.shape)
        combined = combined.view(combined.size(0), -1)

        embedding = self.fc1_embedding(combined)
        embedding = self.fc2_embedding(embedding)
        
        primary_output = self.dropout(F.relu(self.fc1_primary(combined)))
        primary_output = self.fc2_primary(primary_output)
        
        return primary_output, embedding

class GenomicCNN_no_local(nn.Module):
    def __init__(self, 
                 num_chroms = 10,
                 in_channels = 3, 
                 out_channels = 32,
                 embedding_size = 128,
                 dropout_1 = 0.2,
                 dropout_2 = 0.2,
                 num_layers = 1,
                 num_fc = 128,
                 kernel_size_res=[1,3,5,7],
                 ):
        super().__init__()
        # self.num_chroms = num_chroms
        self.res_blocks_global = MultiScaleBlock(in_channels, out_channels, kernel_size_res)
        # self.res_blocks_local = nn.ModuleList([MultiScaleBlock(in_channels, out_channels, kernel_size_res) for _ in range(num_chroms)])
        self.attention_blocks =AttentionBlock(d_model = out_channels*4,nhead=4,num_layers=num_layers,dim_feedforward = out_channels*8, dropout=dropout_2) 
        self.conv = ConvBlock(in_channels=out_channels*4, out_channels=out_channels*8,
                  kernel_size=3, padding=1, stride=1,dilation=1)
        # self.global_fc = nn.Linear(num_fc*2, 1)
        # self.local_fc = nn.Linear(num_fc*2, 1)
        
        self.fc1_embedding = nn.Linear(623*64, embedding_size*2)
        self.fc2_embedding = nn.Linear(embedding_size*2, embedding_size)

        self.fc1_primary = nn.Linear(623*64, num_fc*2)
        self.dropout = nn.Dropout(dropout_1)
        self.fc2_primary = nn.Linear(num_fc*2, 1)

    def forward(self, x):
        res_global = self.res_blocks_global(x[0])
        # res_local = [conv_block(x[i+1]) for i, conv_block in enumerate(self.res_blocks_local)]
        # res_outs_local= torch.stack(res_local, dim=1)
        
        # res_outs_flat = res_outs_local.view(res_outs_local.shape[0], res_outs_local.shape[2], -1)
        
        # combined = torch.cat((res_global, res_outs_flat), dim=2)
        combined = self.attention_blocks(res_global)
        combined = self.conv(combined)
        # print(combined.shape)
        combined = combined.view(combined.size(0), -1)

        embedding = F.relu(self.fc1_embedding(combined))
        embedding = self.fc2_embedding(embedding)
        
        primary_output = self.dropout(F.relu(self.fc1_primary(combined)))
        primary_output = self.fc2_primary(primary_output)
        
        return primary_output, embedding
    
class GenomicCNN_no_contrastive(nn.Module):
    def __init__(self, 
                 num_chroms = 10,
                 in_channels = 3, 
                 out_channels = 32,
                 embedding_size = 128,
                 dropout_1 = 0.2,
                 dropout_2 = 0.2,
                 num_layers = 1,
                 num_fc = 128,
                 kernel_size_res=[1,3,5,7],
                 ):
        super().__init__()
        self.num_chroms = num_chroms
        self.res_blocks_global = MultiScaleBlock(in_channels, out_channels, kernel_size_res)
        self.res_blocks_local = nn.ModuleList([MultiScaleBlock(in_channels, out_channels, kernel_size_res) for _ in range(num_chroms)])
        self.attention_blocks =AttentionBlock(d_model = out_channels*4,nhead=4,num_layers=num_layers,dim_feedforward = out_channels*8, dropout=dropout_2) 
        self.conv = ConvBlock(in_channels=out_channels*4, out_channels=out_channels*8,
                  kernel_size=3, padding=1, stride=1,dilation=1)

        self.fc1_primary = nn.Linear(5603*64, num_fc*2)
        self.dropout = nn.Dropout(dropout_1)
        self.fc2_primary = nn.Linear(num_fc*2, 1)

    def forward(self, x):
        res_global = self.res_blocks_global(x[0])
        res_local = [conv_block(x[i+1]) for i, conv_block in enumerate(self.res_blocks_local)]
        res_outs_local= torch.stack(res_local, dim=1)
        
        res_outs_flat = res_outs_local.view(res_outs_local.shape[0], res_outs_local.shape[2], -1)
        
        combined = torch.cat((res_global, res_outs_flat), dim=2)
        combined = self.attention_blocks(combined)
        combined = self.conv(combined)
        # print(combined.shape)
        combined = combined.view(combined.size(0), -1)

        primary_output = self.dropout(F.relu(self.fc1_primary(combined)))
        primary_output = self.fc2_primary(primary_output)
        
        return primary_output

class GenomicCNN_no_con_no_local(nn.Module):
    def __init__(self, 
                 num_chroms = 10,
                 in_channels = 3, 
                 out_channels = 32,
                 embedding_size = 128,
                 dropout_1 = 0.2,
                 dropout_2 = 0.2,
                 num_layers = 1,
                 num_fc = 128,
                 kernel_size_res=[1,3,5,7],
                 ):
        super().__init__()
        # self.num_chroms = num_chroms
        self.res_blocks_global = MultiScaleBlock(in_channels, out_channels, kernel_size_res)
        # self.res_blocks_local = nn.ModuleList([MultiScaleBlock(in_channels, out_channels, kernel_size_res) for _ in range(num_chroms)])
        self.attention_blocks =AttentionBlock(d_model = out_channels*4,nhead=4,num_layers=num_layers,dim_feedforward = out_channels*8, dropout=dropout_2) 
        self.conv = ConvBlock(in_channels=out_channels*4, out_channels=out_channels*8,
                  kernel_size=3, padding=1, stride=1,dilation=1)

        self.fc1_primary = nn.Linear(623*64, num_fc*2)
        self.dropout = nn.Dropout(dropout_1)
        self.fc2_primary = nn.Linear(num_fc*2, 1)

    def forward(self, x):
        res_global = self.res_blocks_global(x[0])
        # res_local = [conv_block(x[i+1]) for i, conv_block in enumerate(self.res_blocks_local)]
        # res_outs_local= torch.stack(res_local, dim=1)
        
        # res_outs_flat = res_outs_local.view(res_outs_local.shape[0], res_outs_local.shape[2], -1)
        
        # combined = torch.cat((res_global, res_outs_flat), dim=2)
        combined = self.attention_blocks(res_global)
        combined = self.conv(combined)
        # print(combined.shape)
        combined = combined.view(combined.size(0), -1)

        primary_output = self.dropout(F.relu(self.fc1_primary(combined)))
        primary_output = self.fc2_primary(primary_output)
        
        return primary_output
    
class CLCNet_origin(nn.Module):
    
    def __init__(self, shuffle=True, seed=42, input_dim=67500, shared_dim=[4096,2048,1024]):
        super().__init__()
        self.shuffle = shuffle  # 是否打乱，0 不打乱，1 打乱
        self.seed = seed        # 随机种子
        # 共享层
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, shared_dim[0]),
            nn.ReLU(),
            nn.Linear(shared_dim[0], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[2])
        )
        
        self.shared_layer_1 = nn.Sequential(
            nn.Linear(input_dim, shared_dim[2]),
        )
        # 主任务的输出层
        self.main_task_layer = nn.Sequential(
            nn.Linear(shared_dim[2], shared_dim[1]),
            nn.ReLU(),
            
            nn.Linear(shared_dim[1], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], 1)
        )
        # 辅助任务的输出层
        self.aux_task_layer = nn.Sequential(
            nn.Linear(shared_dim[2], shared_dim[1]),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(shared_dim[1], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[0])
        )

    
    def forward(self, x):
        # 展平并连接输入
        # flat_tensors = [x[i].view(x[i].size(0), -1) for i in range(11)]
        if self.shuffle:
            # 设置随机种子
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            # 打乱顺序
            random.shuffle(x)
        # x_cat = torch.cat(flat_tensors, dim=1)
        
        # 共享前向传播
        shared_features = self.shared_layer(x)
        shared_features = shared_features + self.shared_layer_1(x)
        
        # 主任务输出
        main_output = self.main_task_layer(shared_features)
        
        # 辅助任务输出
        aux_output = self.aux_task_layer(shared_features)
        aux_output = F.normalize(aux_output, p=2, dim=1)
        
        return main_output, aux_output

class CLCNet(nn.Module):
    def __init__(self, shuffle=True, seed=42, input_dim=67500, shared_dim=[4096, 2048, 1024]):
        super().__init__()
        self.shuffle = shuffle
        self.seed = seed

        # 共享层：增加Dropout和BatchNorm
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, shared_dim[0]),
            nn.ReLU(),
            nn.BatchNorm1d(shared_dim[0]),
            nn.Dropout(0.3),
            nn.Linear(shared_dim[0], shared_dim[1]),
            nn.ReLU(),
            nn.BatchNorm1d(shared_dim[1]),
            nn.Dropout(0.3),
            nn.Linear(shared_dim[1], shared_dim[2])
        )
        
        # 简化的辅助共享层
        self.shared_layer_1 = nn.Sequential(
            nn.Linear(input_dim, shared_dim[2]),
        )

        # 主任务输出层
        self.main_task_layer = nn.Sequential(
            nn.Linear(shared_dim[2], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], 1)
        )

        # 辅助任务输出层
        self.aux_task_layer = nn.Sequential(
            nn.Linear(shared_dim[2], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[0])
        )

    def forward(self, x):
        if self.shuffle:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.shuffle(x)

        # 共享前向传播
        shared_features = self.shared_layer(x)
        shared_features = shared_features + self.shared_layer_1(x)

        # 主任务输出
        main_output = self.main_task_layer(shared_features)

        # 辅助任务输出
        aux_output = self.aux_task_layer(shared_features)
        aux_output = F.normalize(aux_output, p=2, dim=1)

        return main_output, aux_output
    
class LinearModel_no_con(nn.Module):
    def __init__(self, input_dim=67500, shared_dim=[4096,2048,1024]):
        super().__init__()

        # 共享层
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, shared_dim[0]),
            nn.ReLU(),
            nn.Linear(shared_dim[0], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[2])
        )
        
        self.shared_layer_1 = nn.Sequential(
            nn.Linear(input_dim, shared_dim[2]),
        )
        # 主任务的输出层
        self.main_task_layer = nn.Sequential(
            nn.Linear(shared_dim[2], shared_dim[1]),
            nn.ReLU(),
            
            nn.Linear(shared_dim[1], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], 1)
        )

    
    def forward(self, x_cat):

        # 共享前向传播
        shared_features = self.shared_layer(x_cat)
        shared_features = shared_features + self.shared_layer_1(x_cat)
        
        # 主任务输出
        main_output = self.main_task_layer(shared_features)
        
        return main_output

class LinearModel_no_local(nn.Module):
    def __init__(self, input_dim=67500, shared_dim=[4096,2048,1024]):
        super().__init__()

        # 共享层
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, shared_dim[0]),
            nn.ReLU(),
            nn.Linear(shared_dim[0], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[2])
        )
        
        self.shared_layer_1 = nn.Sequential(
            nn.Linear(input_dim, shared_dim[2]),
        )
        # 主任务的输出层
        self.main_task_layer = nn.Sequential(
            nn.Linear(shared_dim[2], shared_dim[1]),
            nn.ReLU(),
            
            nn.Linear(shared_dim[1], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], 1)
        )
        # 辅助任务的输出层
        self.aux_task_layer = nn.Sequential(
            nn.Linear(shared_dim[2], shared_dim[1]),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(shared_dim[1], shared_dim[1]),
            nn.ReLU(),
            nn.Linear(shared_dim[1], shared_dim[0])
        )

    
    def forward(self, x):
        # 展平并连接输入
        flat_tensors = [x[i].view(x[i].size(0), -1) for i in range(1)]
        x_cat = torch.cat(flat_tensors, dim=1)
        
        # 共享前向传播
        shared_features = self.shared_layer(x_cat)
        shared_features = shared_features + self.shared_layer_1(x_cat)
        
        # 主任务输出
        main_output = self.main_task_layer(shared_features)
        
        # 辅助任务输出
        aux_output = self.aux_task_layer(shared_features)
        aux_output = F.normalize(aux_output, p=2, dim=1)
        
        return main_output, aux_output   

class DeepGS(nn.Module):
    def __init__(self,dim=67500):
        super().__init__()
        
        # Convolutional layer: 1x18 kernel, 8 filters, stride 1x1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=18, stride=1)
        
        # Max-pooling layer: 1x4 pooling, stride 1x4
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.fc1 = nn.Linear(dim, 32)
        self.fc2 = nn.Linear(32, 1)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.05)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        return x

class DeepGSpre(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=18, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        # print(x.shape)
        return x
# model = DeepGS()
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=0.00001)
# epochs = 6000

class DNNGP(nn.Module):
    def __init__(self, input_dim):
        super(DNNGP, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4)

        # 批归一化层
        self.bn = nn.BatchNorm1d(num_features=64)
        
        # Dropout
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        # Flatten + Dense + Output
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)  # 输出为回归值
        return x

class DNNGPpre(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4)
        self.bn = nn.BatchNorm1d(num_features=64)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # flatten
        return x
  
if __name__ == "__main__":
    num_chroms = 10
    in_channels = 3
    out_channels = 8
    num_features_global = 2500
    num_features_local = 2000
    batch_size = 2
    embedding_size = 256
    dropout_1 = 0.5
    dropout_2 = 0.5
    num_layers = 1
    num_fc = 256

    model = GenomicCNN(num_chroms=num_chroms, 
                       in_channels=in_channels, 
                       out_channels=out_channels, 
                       embedding_size=embedding_size, 
                       dropout_1=dropout_1,
                       dropout_2=dropout_2,
                       num_layers=num_layers,
                       num_fc=num_fc,
                       kernel_size_res=[1,3,5,7])
    

    example_input = [torch.randn(batch_size, in_channels, num_features_global)] + \
        [torch.randn(batch_size, in_channels, num_features_local) for _ in range(num_chroms)]

    output,aux_output= model(example_input)
    print(output,aux_output)
    print(output.shape, aux_output.shape)
