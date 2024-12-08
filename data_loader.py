from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, batch_size=32, train_split=0.9):
    # 数据预处理
    data_transforms = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载整个数据集
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    
    # 获取类别名称
    class_names = full_dataset.classes
    
    # 计算数据集的大小
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    # 随机划分数据集
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, class_names
