import timm
import torch


def main():
    net = timm.SegNest(img_size=512, in_chans=1, 
                  patch_size=4, num_levels=4, 
                  embed_dims=(768, 768, 768, 768), num_heads=(768, 768, 768, 768), 
                  depths=(3, 3, 3, 3), num_classes=14, 
                  mlp_ratio=4.).float()
    x = torch.rand(1, 1, 512, 512).float()
    y = net(x)
    print("x", x.shape)
    print("len", len(y))
    for i in range(len(y)):
        print("y",i, y[i].shape)

if __name__ == '__main__':
    main()