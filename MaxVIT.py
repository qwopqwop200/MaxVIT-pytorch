import torch
import torch.nn as nn
import torch.nn.functional as F

class Channel_Layernorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features,1)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class SE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(in_dim, hidden_dim,1, bias=False),
                                  nn.GELU(),
                                  nn.Conv2d(hidden_dim, in_dim,1, bias=False),
                                  nn.Sigmoid())
    def forward(self, x):
        return x * self.gate(x.mean((2,3), keepdim=True))  

class MBConv(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size=3,stride_size=1,expand_rate = 4,se_rate = 0.25,dropout = 0.):
        super().__init__()
        hidden_dim = int(expand_rate * out_dim)
        self.bn = nn.BatchNorm2d(in_dim)
        self.expand_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
                                         nn.BatchNorm2d(hidden_dim),
                                         nn.GELU())
        self.dw_conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride_size, kernel_size//2, groups=hidden_dim, bias=False),
                                     nn.BatchNorm2d(hidden_dim),
                                     nn.GELU())
        self.se = SE(hidden_dim,max(1,int(out_dim*se_rate)))
        self.out_conv = nn.Sequential(nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
                                      nn.BatchNorm2d(out_dim))
        if stride_size > 1:
            self.proj = nn.Sequential(nn.MaxPool2d(kernel_size, stride_size, kernel_size//2),
                                      nn.Conv2d(in_dim, out_dim, 1, bias=False)) 
        else: 
            self.proj = nn.Identity()
    
    def forward(self, x):
        out = self.bn(x)
        out = self.expand_conv(out)
        out = self.dw_conv(out)
        out = self.se(out)
        out = self.out_conv(out)
        return out + self.proj(x)
    
class Rel_Attention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.win_h,self.win_w = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_bias_table = nn.Parameter(torch.zeros((2*self.win_h-1)*(2*self.win_w-1), num_heads))

        coords = torch.meshgrid((torch.arange(self.win_h), torch.arange(self.win_w)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.win_h - 1
        relative_coords[1] += self.win_w - 1
        relative_coords[0] *= 2 * self.win_w - 1
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.qkv = nn.Conv2d(dim,dim*3,1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B,H,W,C,_,_ = x.shape
        N = self.win_h * self.win_w
        x = x.reshape(-1,C,self.win_h,self.win_w)

        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.num_heads))
        relative_bias = relative_bias.reshape(N,N,-1).permute(2,0,1).contiguous()
        
        q,k,v = self.qkv(x).chunk(3, dim=1)
        q = q.reshape([-1,C,N]).reshape(-1, self.num_heads, C // self.num_heads, N).permute(0,1,3,2).contiguous()
        k = k.reshape([-1,C,N]).reshape(-1, self.num_heads, C // self.num_heads, N).permute(0,1,3,2).contiguous()
        v = v.reshape([-1,C,N]).reshape(-1, self.num_heads, C // self.num_heads, N).permute(0,1,3,2).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + relative_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).reshape([-1,C,self.win_h,self.win_w])
        x = self.proj(x)
        x = self.proj_drop(x).reshape([B,H,W,C,self.win_h,self.win_w])
        return x
    
def block(x,window_size):
    B,C,H,W = x.shape
    x = x.reshape(B,C,H//window_size[0],window_size[0],W//window_size[1],window_size[1])
    x = x.permute(0,2,4,1,3,5).contiguous()
    return x
                      
def unblock(x):
    B,H,W,C,win_H,win_W = x.shape
    x = x.permute(0,3,1,4,2,5).contiguous().reshape(B,C,H*win_W,H*win_W)
    return x
    
class Window_Block(nn.Module):
    def __init__(self, dim, block_size=(7,7), num_heads=8, mlp_ratio=4., qkv_bias=False,qk_scale=None, drop=0., 
                 attn_drop=0.,drop_path=0., act_layer=nn.GELU ,norm_layer=Channel_Layernorm):
        super().__init__()
        self.block_size = block_size
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Rel_Attention(dim, block_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self,x):
        assert x.shape[2]%self.block_size[0] == 0 & x.shape[3]%self.block_size[1] == 0, 'image size should be divisible by block_size'
        
        out = block(self.norm1(x),self.block_size)
        out = self.attn(out)
        x = x + self.drop_path(unblock(self.attn(out)))
        out = self.mlp(self.norm2(x))
        x = x + self.drop_path(out)
        return x
    
class Grid_Block(nn.Module):
    def __init__(self, dim, grid_size=(7,7), num_heads=8, mlp_ratio=4., qkv_bias=False,qk_scale=None, drop=0., 
                 attn_drop=0.,drop_path=0., act_layer=nn.GELU ,norm_layer=Channel_Layernorm):
        super().__init__()
        self.grid_size = grid_size
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Rel_Attention(dim, grid_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self,x):
        assert x.shape[2]%self.grid_size[0] == 0 & x.shape[3]%self.grid_size[1] == 0, 'image size should be divisible by grid_size'
        grid_size = (x.shape[2]//self.grid_size[0], x.shape[3]//self.grid_size[1])
        
        out = block(self.norm1(x),grid_size)
        out = out.permute(0,4,5,3,1,2).contiguous()
        out = self.attn(out).permute(0,4,5,3,1,2).contiguous()
        x = x + self.drop_path(unblock(out))
        out = self.mlp(self.norm2(x))
        x = x + self.drop_path(out)
        return x
    
class Max_Block(nn.Module):
    def __init__(self, in_dim, out_dim , num_heads=8.,block_size=(7,7), grid_size=(7,7),
                 mbconv_ksize = 3,pooling_size = 1,mbconv_expand_rate=4,se_reduce_rate=0.25,
                 mlp_ratio=4,qkv_bias=False,qk_scale=None, drop=0., attn_drop=0.,drop_path=0., 
                 act_layer=nn.GELU ,norm_layer=Channel_Layernorm):
        super().__init__()
        self.mbconv = MBConv(in_dim,out_dim,mbconv_ksize,pooling_size,mbconv_expand_rate,se_reduce_rate,drop)
        self.block_attn = Window_Block(out_dim, block_size, num_heads, mlp_ratio, qkv_bias,qk_scale, drop, 
                                       attn_drop,drop_path, act_layer ,norm_layer)
        self.grid_attn = Grid_Block(out_dim, grid_size, num_heads, mlp_ratio, qkv_bias,qk_scale, drop, 
                                    attn_drop,drop_path, act_layer ,norm_layer)
        
    def forward(self, x):
        x = self.mbconv(x)
        x = self.block_attn(x)
        x = self.grid_attn(x)
        return x
                      
class MaxViT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv_stem = nn.Sequential(nn.Conv2d(args['input_dim'], args['stem_dim'], 3,2,3//2),
                                       nn.BatchNorm2d(args['stem_dim']),
                                       nn.GELU(),
                                       nn.Conv2d(args['stem_dim'], args['stem_dim'], 3,1,3//2),
                                       nn.BatchNorm2d(args['stem_dim']),
                                       nn.GELU())
        in_dim = args['stem_dim']
        self.max_blocks = nn.ModuleList([])
        for i,num_block in enumerate(args['stage_num_block']):
            layers = nn.ModuleList([])
            out_dim = args['stage_dim']*(2**i)
            num_head = args['num_heads']*(2**i)
            for i in range(num_block):
                pooling_size = args['pooling_size']if i == 0 else 1
                layers.append(Max_Block(in_dim,out_dim,num_head,args['block_size'], 
                                        args['grid_size'],args['mbconv_ksize'],pooling_size,
                                        args['mbconv_expand_rate'],args['se_rate'],args['mlp_ratio'],
                                        args['qkv_bias'],args['qk_scale'], args['drop'], args['attn_drop'],
                                        args['drop_path'],args['act_layer'] ,args['norm_layer']))
                in_dim = out_dim
            self.max_blocks.append(layers)
        self.last_conv = nn.Sequential(nn.Conv2d(in_dim,in_dim,1, bias=False),
                                       nn.BatchNorm2d(in_dim),
                                       nn.GELU())
        self.proj = nn.Linear(in_dim,args['num_classes'])
        self.softmax = nn.Softmax(1)
        
    def forward(self, x):
        x = self.conv_stem(x)
        for blocks in self.max_blocks:
            for block in blocks:
                x = block(x)
        x = self.last_conv(x)
        x = self.softmax(self.proj(x.mean([2, 3])))
        return x
    
tiny_args = {
'stage_num_block':[2,2,5,2],
'input_dim':3,
'stem_dim':64,
'stage_dim':64,
'num_heads': 2,
'mbconv_ksize': 3,
'pooling_size': 2,
'num_classes':1000,
'block_size':(7,7),
'grid_size':(7,7),
'mbconv_expand_rate':4,
'se_rate':0.25,
'mlp_ratio':4,
'qkv_bias':True,
'qk_scale':None, 
'drop':0., 
'attn_drop':0.,
'drop_path':0.,
'act_layer':nn.GELU,
'norm_layer':Channel_Layernorm}

small_args = {
'stage_num_block':[2,2,5,2],
'input_dim':3,
'stem_dim':64,
'stage_dim':96,
'num_heads': 3,
'mbconv_ksize': 3,
'pooling_size': 2,
'num_classes':1000,
'block_size':(7,7),
'grid_size':(7,7),
'mbconv_expand_rate':4,
'se_rate':0.25,
'mlp_ratio':4,
'qkv_bias':True,
'qk_scale':None, 
'drop':0., 
'attn_drop':0.,
'drop_path':0.,
'act_layer':nn.GELU,
'norm_layer':Channel_Layernorm}

base_args = {
'stage_num_block':[2,6,14,2],
'input_dim':3,
'stem_dim':64,
'stage_dim':96,
'num_heads': 3,
'mbconv_ksize': 3,
'pooling_size': 2,
'num_classes':1000,
'block_size':(7,7),
'grid_size':(7,7),
'mbconv_expand_rate':4,
'se_rate':0.25,
'mlp_ratio':4,
'qkv_bias':True,
'qk_scale':None, 
'drop':0., 
'attn_drop':0.,
'drop_path':0.,
'act_layer':nn.GELU,
'norm_layer':Channel_Layernorm}

large_args = {
'stage_num_block':[2,6,14,2],
'input_dim':3,
'stem_dim':128,
'stage_dim':128,
'num_heads': 4,
'mbconv_ksize': 3,
'pooling_size': 2,
'num_classes':1000,
'block_size':(7,7),
'grid_size':(7,7),
'mbconv_expand_rate':4,
'se_rate':0.25,
'mlp_ratio':4,
'qkv_bias':True,
'qk_scale':None, 
'drop':0., 
'attn_drop':0.,
'drop_path':0.,
'act_layer':nn.GELU,
'norm_layer':Channel_Layernorm}

xlarge_args = {
'stage_num_block':[2,6,14,2],
'input_dim':3,
'stem_dim':192,
'stage_dim':192,
'num_heads': 6,
'mbconv_ksize': 3,
'pooling_size': 2,
'num_classes':1000,
'block_size':(7,7),
'grid_size':(7,7),
'mbconv_expand_rate':4,
'se_rate':0.25,
'mlp_ratio':4,
'qkv_bias':True,
'qk_scale':None, 
'drop':0., 
'attn_drop':0.,
'drop_path':0.,
'act_layer':nn.GELU,
'norm_layer':Channel_Layernorm}

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def test():
    print(f'MaxViT-T:{get_n_params(MaxViT(tiny_args))}')
    print(f'MaxViT-S:{get_n_params(MaxViT(small_args))}')
    print(f'MaxViT-B:{get_n_params(MaxViT(base_args))}')
    print(f'MaxViT-L:{get_n_params(MaxViT(large_args))}')
    print(f'MaxViT-XL:{get_n_params(MaxViT(xlarge_args))}')
    print(MaxViT(tiny_args)(torch.zeros(2,3,224,224)).shape)
    
if __name__ == "__main__":
    test()
