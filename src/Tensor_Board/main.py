import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

writer = SummaryWriter('runs/exp_demo')

class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN feature
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # LSTM section
        self.lstm = nn.LSTM(
            input_size=64*4*4, 
            hidden_size=128, 
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # attention section
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # input shape: (batch_size, seq_len, C, H, W)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        cnn_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            features = self.cnn(frame)  # (bs, 64, 4, 4)
            features = features.view(batch_size, -1)  # (bs, 64*4*4)
            cnn_features.append(features)
        
        cnn_features = torch.stack(cnn_features, dim=1)  # (bs, seq_len, 64*4*4)
        lstm_out, _ = self.lstm(cnn_features)  # (bs, seq_len, 256)
        attention_weights = self.attention(lstm_out)  # (bs, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (bs, 256)
        return self.classifier(context_vector)

# 3. 创建模拟数据 (视频分类任务)
def generate_fake_data(batch_size=16, seq_len=10):
    # 生成随机视频数据 (batch, seq_len, 3, 64, 64)
    videos = torch.randn(batch_size, seq_len, 3, 64, 64)
    labels = torch.randint(0, 10, (batch_size,))
    return videos, labels

train_data, train_labels = generate_fake_data(100)
val_data, val_labels = generate_fake_data(20)
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)

model = ComplexModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def train(model, dataloader, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        if batch_idx % 5 == 0:  
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(
                        f"Gradients/{name}", 
                        param.grad, 
                        epoch * len(dataloader) + batch_idx
                    )
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(
                    f"Parameters/{name}", 
                    param, 
                    epoch
                )
    
    return total_loss / len(dataloader)

def validate(model, dataloader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
    
    accuracy = correct / len(dataloader.dataset)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    return accuracy

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

for epoch in range(100):
    train_loss = train(model, train_loader, epoch)
    val_acc = validate(model, val_loader, epoch)
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    
    scheduler.step()
    
    if epoch == 0:
        dummy_input = torch.randn(1, 10, 3, 64, 64)  
        writer.add_graph(model, dummy_input)

writer.flush()
writer.close()