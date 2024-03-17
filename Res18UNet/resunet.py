import torch
from torch import nn
from torchvision.models.resnet import resnet18


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=resnet18(weights='DEFAULT')
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.e5 = nn.Conv2d(512, 1024, kernel_size=3, padding=0)
        
        
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dconv11= nn.Conv2d(512,512,kernel_size=3,padding=0,stride=1)
        self.dconv12= nn.Conv2d(512,512,kernel_size=3,padding=0,stride=1)
        self.dconv13 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dconv14 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        
        self.dconv21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dconv22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)        
        
        self.dconv31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dconv32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.dconv41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dconv42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        self.dconv51 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dconv52 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv6 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.last_layer = nn.Conv2d(32, 2, kernel_size=3,stride=1,padding=1)
        
        
    def forward(self,x):

        # Mentioned Size Format: [BATCH_SIZE, z, x, y]
        
        self.op1 = self.model.conv1(x)  # torch.Size([32, 64, 128, 128])
        self.op1 = self.model.bn1(self.op1)
        self.op1 = self.model.relu(self.op1)
        
        self.op2 = self.model.maxpool(self.op1) # torch.Size([32, 64, 64, 64])        
    
        self.op3 = self.model.layer1(self.op2) # torch.Size([32, 64, 64, 64])
        self.op4 = self.model.layer2(self.op3) # torch.Size([32, 128, 32, 32])
        self.op5 = self.model.layer3(self.op4) # torch.Size([32, 256, 16, 16])
        self.op6 = self.model.layer4(self.op5) # torch.Size([32, 512, 8, 8])

        
        self.op7=(self.e5(self.op6)) # torch.Size([32, 1024, 6, 6])
        self.op8=(self.upconv1(self.op7)) # torch.Size([32, 512, 12, 12])

        self.op9=self.dconv12(((self.dconv11(self.op8)))) # torch.Size([32, 512, 8, 8])
        self.op9_cat=torch.concat((self.op9,self.op6),dim=1) # torch.Size([32, 1024, 8, 8])

        self.op10=self.dconv13((self.op9_cat))
        self.op11=self.dconv14((self.op10))

        self.op12=self.upconv2((self.op11)) # # torch.Size([32, 256, 16, 16])
        self.op12_cat=torch.concat((self.op12,self.op5),dim=1) # torch.Size([32, 512, 16, 16])

        self.op13=self.dconv21((self.op12_cat))
        self.op14=self.dconv22((self.op13))

        self.op15=self.upconv3((self.op14)) # torch.Size([32, 128, 32, 32])
        self.op15_cat=torch.concat((self.op15,self.op4),dim=1) # torch.Size([32, 256, 32, 32])
        
        self.op16=self.dconv31((self.op15_cat))
        self.op17=self.dconv32((self.op16))

        self.op18=self.upconv4((self.op17)) # torch.Size([32, 64, 64, 64])
        self.op18_cat=torch.concat((self.op18,self.op3),dim=1) # torch.Size([32, 128, 64, 64])

        self.op19=self.dconv41((self.op18_cat))
        self.op20=self.dconv42((self.op19))

        self.op21=self.upconv5((self.op20)) # torch.Size([32, 64, 128, 128])
        self.op21_cat=torch.concat((self.op21,self.op1),dim=1) # torch.Size([32, 128, 128, 128])

        self.op22=self.dconv51((self.op21_cat))
        self.op23=self.dconv52((self.op22))

        self.op24=(self.upconv6(self.op23)) # torch.Size([32, 32, 256, 256])
        self.op25=(self.last_layer(self.op24)) # torch.Size([32, 2, 256, 256])
        
