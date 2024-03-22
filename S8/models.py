
dropout_value = 0.1
class Net_BN(nn.Module):
    def __init__(self):
        super(Net_BN, self).__init__()
        # C1, input size: 32, in_channels = 3, jin = 1
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # out = 32 , jout = 1, rf = 3

        # C2, in = 32, jin = 1
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # out = 32 , jout = 1, rf = 5

        # c3, in = 32, jin = 1
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # out = 32, jout = 1, rf = 5

        # TRANSITION BLOCK, MaxPool P1
        self.P1 = nn.MaxPool2d(2, 2) # output_size = 16, jout = 2, rf = 6

        # C4, in = 16, jin = 2
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # out = 16, jout = 2, rf = 10

        # C5, in = 16, jin = 2 
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # out = 16, jout = 2, rf = 14 

        # C6, in = 16, jin = 2 
        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # out = 16, jout = 2, rf = 18

        # c7, in = 16, jin = 2 
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # out = 16, jout = 2, rf = 18
  
        # TRANSITION BLOCK, MaxPool P2
        self.P2 = nn.MaxPool2d(2, 2) # output_size = 8, rf = 20

        # C8, in = 8, jin = 4
        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # out = 8, jout = 4, rf = 28 

        # C9, in = 8, jin = 4 
        self.C9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # out = 8, jout = 4, rf = 36

        # C10, in = 8, jin = 4 
        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=28, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(28),
            nn.Dropout(dropout_value)
        ) # out = 8, jout = 4, rf = 44

        self.GAP = nn.Sequential(
            nn.AvgPool2d(kernel_size=8) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # out = 1 

        # c11, input size: 
        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.ReLU(),
            # nn.BatchNorm2d(10),
            # nn.Dropout(dropout_value)
        ) # out = , rf = 

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.c3(x)
        x = self.P1(x)
        x = self.C4(x)
        x = self.C5(x)
        x = self.C6(x)
        x = self.c7(x)
        x = self.P2(x)
        x = self.C8(x)
        x = self.C9(x)
        x = self.C10(x)
        x = self.GAP(x)
        x = self.c11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

#### Modelf or groupNorm (with N=4 groups)
num_groups = 4  ### group norm with 4 groups
class Net_GN(nn.Module):
    def __init__(self):
        super(Net_GN, self).__init__()
        # C1, input size: 32, in_channels = 3, jin = 1
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 16),
            nn.Dropout(dropout_value)
        ) # out = 32 , jout = 1, rf = 3

        # C2, in = 32, jin = 1
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 32),
            nn.Dropout(dropout_value)
        ) # out = 32 , jout = 1, rf = 5

        # c3, in = 32, jin = 1
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 64),
            nn.Dropout(dropout_value)
        ) # out = 32, jout = 1, rf = 5

        # TRANSITION BLOCK, MaxPool P1
        self.P1 = nn.MaxPool2d(2, 2) # output_size = 16, jout = 2, rf = 6

        # C4, in = 16, jin = 2
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 16),
            nn.Dropout(dropout_value)
        ) # out = 16, jout = 2, rf = 10

        # C5, in = 16, jin = 2
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 32),
            nn.Dropout(dropout_value)
        ) # out = 16, jout = 2, rf = 14

        # C6, in = 16, jin = 2
        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 32),
            nn.Dropout(dropout_value)
        ) # out = 16, jout = 2, rf = 18

        # c7, in = 16, jin = 2
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 32),
            nn.Dropout(dropout_value)
        ) # out = 16, jout = 2, rf = 18

        # TRANSITION BLOCK, MaxPool P2
        self.P2 = nn.MaxPool2d(2, 2) # output_size = 8, rf = 20

        # C8, in = 8, jin = 4
        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 16),
            nn.Dropout(dropout_value)
        ) # out = 8, jout = 4, rf = 28

        # C9, in = 8, jin = 4
        self.C9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 32),
            nn.Dropout(dropout_value)
        ) # out = 8, jout = 4, rf = 36

        # C10, in = 8, jin = 4
        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=28, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 28),
            nn.Dropout(dropout_value)
        ) # out = 8, jout = 4, rf = 44

        self.GAP = nn.Sequential(
            nn.AvgPool2d(kernel_size=8) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # out = 1

        # c11, input size:
        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.ReLU(),
            # nn.BatchNorm2d(10),
            # nn.Dropout(dropout_value)
        ) # out = , rf =

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.c3(x)
        x = self.P1(x)
        x = self.C4(x)
        x = self.C5(x)
        x = self.C6(x)
        x = self.c7(x)
        x = self.P2(x)
        x = self.C8(x)
        x = self.C9(x)
        x = self.C10(x)
        x = self.GAP(x)
        x = self.c11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



#### Modelf or LayerNorm (with N=1 in groupNorm)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # C1, input size: 32, in_channels = 3, jin = 1
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 16),
            nn.Dropout(dropout_value)
        ) # out = 32 , jout = 1, rf = 3

        # C2, in = 32, jin = 1
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 32),
            nn.Dropout(dropout_value)
        ) # out = 32 , jout = 1, rf = 5

        # c3, in = 32, jin = 1
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 64),
            nn.Dropout(dropout_value)
        ) # out = 32, jout = 1, rf = 5

        # TRANSITION BLOCK, MaxPool P1
        self.P1 = nn.MaxPool2d(2, 2) # output_size = 16, jout = 2, rf = 6

        # C4, in = 16, jin = 2
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 16),
            nn.Dropout(dropout_value)
        ) # out = 16, jout = 2, rf = 10

        # C5, in = 16, jin = 2
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 32),
            nn.Dropout(dropout_value)
        ) # out = 16, jout = 2, rf = 14

        # C6, in = 16, jin = 2
        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 32),
            nn.Dropout(dropout_value)
        ) # out = 16, jout = 2, rf = 18

        # c7, in = 16, jin = 2
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 32),
            nn.Dropout(dropout_value)
        ) # out = 16, jout = 2, rf = 18

        # TRANSITION BLOCK, MaxPool P2
        self.P2 = nn.MaxPool2d(2, 2) # output_size = 8, rf = 20

        # C8, in = 8, jin = 4
        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 16),
            nn.Dropout(dropout_value)
        ) # out = 8, jout = 4, rf = 28

        # C9, in = 8, jin = 4
        self.C9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 32),
            nn.Dropout(dropout_value)
        ) # out = 8, jout = 4, rf = 36

        # C10, in = 8, jin = 4
        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=28, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 28),
            nn.Dropout(dropout_value)
        ) # out = 8, jout = 4, rf = 44

        self.GAP = nn.Sequential(
            nn.AvgPool2d(kernel_size=8) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # out = 1

        # c11, input size:
        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.ReLU(),
            # nn.BatchNorm2d(10),
            # nn.Dropout(dropout_value)
        ) # out = , rf =

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.c3(x)
        x = self.P1(x)
        x = self.C4(x)
        x = self.C5(x)
        x = self.C6(x)
        x = self.c7(x)
        x = self.P2(x)
        x = self.C8(x)
        x = self.C9(x)
        x = self.C10(x)
        x = self.GAP(x)
        x = self.c11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
