import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_pretrained_model
import pandas as pd
import os

def load_test_data(data_dir, csv_file, batch_size=32):
    test_df = pd.read_csv(csv_file)

    data_transforms = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, root_dir, transform=None):
            self.dataframe = dataframe
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 1])
            image = datasets.folder.default_loader(img_name)
            if self.transform:
                image = self.transform(image)
            return image, self.dataframe.iloc[idx, 0]

    test_dataset = TestDataset(dataframe=test_df, root_dir=data_dir, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, test_df

def predict_and_generate_submission(model, test_loader, test_df, class_names, submission_file='submission.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    predicted_breeds = [class_names[p] for p in predictions]

    submission_df = pd.DataFrame({'id': test_df['id'], 'breed': predicted_breeds})
    submission_df.to_csv(submission_file, index=False)
    print(f'Submission file saved to {submission_file}')

if __name__ == "__main__":
    train_data_dir = 'data/data'
    train_dataset = datasets.ImageFolder(root=train_data_dir)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    model = get_pretrained_model(num_classes=num_classes)
    model.load_state_dict(torch.load('models/best_model.pth'))

    test_data_dir = 'data/test_data/test_data'
    test_csv_file = 'data/test_image_index.csv'
    test_loader, test_df = load_test_data(test_data_dir, test_csv_file)

    predict_and_generate_submission(model, test_loader, test_df, class_names)
