import torch
from data_loader import get_data_loaders
from model import get_pretrained_model
import os

def train_model(model, train_loader, val_loader, num_epochs=20, model_save_dir='models', print_freq=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.layer4.parameters()) + list(model.fc.parameters()), 
        lr=0.001
    )

    best_val_accuracy = 0.0

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # 更频繁地打印损失
            if (batch_idx + 1) % print_freq == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.4f}')

        # 在验证集上评估
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct.double() / len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

        epoch_model_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        print(f'Model for epoch {epoch+1} saved at {epoch_model_path}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model updated with accuracy: {best_val_accuracy:.4f}')

    print("训练完成")

if __name__ == "__main__":
    data_dir = 'data/data'  # 确保路径指向正确的目录
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size=64)
    num_classes = len(class_names)
    model = get_pretrained_model(num_classes=num_classes)
    train_model(model, train_loader, val_loader, num_epochs=20)
