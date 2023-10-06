from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from CustomDataset import CustomDataset

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


import torchsummary
from torch.utils.tensorboard import SummaryWriter
import torch.profiler


writer = SummaryWriter()

# prof = torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('runs/profiler/'),
#         record_shapes=True,
#         with_stack=True,
#         use_cuda=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = CustomDataset(root_dir='Datasets/imagenet_images', transform=transform)

print('len(dataset))',len(dataset))

class_to_idx = dataset.class_to_idx

conf_matrix_labels = class_to_idx.keys()

batch_size = 64
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# train validate test split


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


val_size = train_size - int(0.8 *train_size )
train_size = train_size - val_size

train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[train_size,val_size])

print('len(train_dataset))',len(train_dataset))

print('len(val_dataset))',len(val_dataset))

print('len(test_dataset))',len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



# model

class CustomCNN(nn.Module):
    def __init__(self,num_classes = 4):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = x.view(-1, 256 * 28 * 28)  # Flatten for fully connected layers
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        return x



model = CustomCNN()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
model.to(device)

torchsummary.summary(model, input_size=(3, 224, 224))
PATH = 'best_model.pth'


# loss function and optimizer

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# training loop

print('start training')

epochs = 30
best_val_loss = 10.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()


    epoch_train_loss = running_loss / len(train_dataloader)
    epoch_train_acc = correct_train / total_train

    # Log Training accuracy and loss to TensorBoard
    writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
    writer.add_scalar('Loss/train', epoch_train_loss, epoch)
    
    # validation 
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_val_loss = val_loss / len(val_dataloader)
    epoch_val_acc = correct / total

    print(f'Epoch {epoch+1}/{epochs}, '
          f'Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_acc:.4f}, '
          f'Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.4f}')
    
    # Log validation accuracy and loss to TensorBoard
    writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
    writer.add_scalar('Loss/val', epoch_val_loss, epoch)

    # Save the model if it has the best validation accuracy
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), PATH)

    # prof.step()

print('Finished Training')

# prof.stop()


# loading the model

model = CustomCNN()
model.to(device)
model.load_state_dict(torch.load(PATH))

# testing loop
model.eval()
correct = 0
total = 0
test_loss = 0.0
all_preds = []
all_labels = []
num_classes = 4




# test_dataloader = train_dataloader

with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = correct / total
test_loss /= len(test_dataloader)

# Log Testing accuracy and loss to TensorBoard

# writer.add_scalar('Accuracy/test', test_acc, epoch)
# writer.add_scalar('Loss/test', test_loss, epoch)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# class_to_idx = {'ocean': 0, 'person': 1, 'plant': 2, 'sky': 3}

conf_matrix = confusion_matrix(all_labels, all_preds)

# conf_matrix_top_5 = confusion_matrix(all_labels, top_5_all_preds)

cmd_top5 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=conf_matrix_labels)
cmd_top5.plot(cmap='Blues',colorbar=False)
cmd_top5.ax_.set(xlabel='Predicted', ylabel='True')
cmd_top5.figure_.savefig('confusion_matrix.png')

# plt.figure(figsize=(num_classes, num_classes))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.xticks(range(num_classes), class_to_idx.keys())
# plt.yticks(range(num_classes), class_to_idx.keys())
# plt.title('Confusion Matrix')
# # plt.show()
# plt.savefig('confusion_matrix.png')

#close writer

writer.close()


################



# Step 1: Load the Model
# class_index = {'cat': 0, 'dog': 1, 'person': 2, 'plant': 3}
# def display_results(image_path, confidence, predicted_class):
#     image = Image.open(image_path)
#     plt.imshow(image)
#     plt.title(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}')
#     plt.axis('off')
#     # plt.show()
#     plt.savefig('output.png')
# # Step 2: Preprocess the Image
# def preprocess_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Adjust size as needed
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#     image = Image.open(image_path).convert('RGB')
#     return transform(image).unsqueeze(0)

# # Step 3: Inference
# def predict(image_path, model):
#     image_tensor = preprocess_image(image_path)
#     image_tensor = image_tensor.to(device)
#     with torch.no_grad():
#         output = model(image_tensor)

#     probabilities = torch.nn.functional.softmax(output[0], dim=0)
#     confidence, predicted_class = torch.max(probabilities, 0)

#     return confidence.item(), predicted_class.item()

# # Step 4: Display the Image and Results


# # Replace 'path/to/your/image.jpg' with the path to your image
# image_path = 'Datasets/imagenet_images/person/3905109798_5c03a89bf8.jpg'
# confidence, predicted_class = predict(image_path, model)
# predicted_class = list(class_index.keys())[list(class_index.values()).index(predicted_class)]
# display_results(image_path, confidence, predicted_class)
