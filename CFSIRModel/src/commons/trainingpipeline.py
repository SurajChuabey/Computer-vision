from src.datapreprocessing.dataloader import CFSIRDataLoader
from src.commons.model import CFSIRModel
from src.constant.constant import Constants
from src.utility.tensorboard import TensorBoard
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch



class TrainingPipeline:

    def __init__(self,epochs,learning_rate,batch_size,device):
        self.epochs = epochs
        self.device = device
        self.lr = learning_rate
        self.batch_size = batch_size
        self.model = CFSIRModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer, mode='min', factor=0.1, patience=5
        )
        self.train_data,self.test_data = CFSIRDataLoader().load_data(Constants.DATA_DIR_PATH,batch_size=self.batch_size)
        self.writer = TensorBoard.get_tensorboard_writer()
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        self.writer.add_graph(self.model, dummy_input)

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            running_loss = Constants.ZERO
            correct = 0
            total = 0
            progress_bar = tqdm(self.train_data, desc=f"Epoch [{epoch+1}/{self.epochs}]", leave=False)

            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                model_pred = self.model(images)
                loss = self.criterion(model_pred, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(model_pred.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                progress_bar.set_postfix(loss=loss.item())

            train_loss = running_loss / len(self.train_data)
            train_acc = 100 * correct / total

            # Log training metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Validation phase
            self.model.eval()
            val_loss = Constants.ZERO
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.test_data:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(self.test_data)
            print(f"Epoch [{epoch+1}/{self.epochs}] Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

            # Log validation metrics
            val_acc = 100 * correct / total
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/validation', val_acc, epoch)

            print(f'Epoch [{epoch+1}/{self.epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} '
                  f'Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%')

            # Step the scheduler based on validation loss
            self.scheduler.step(val_loss)
            
    def eval(self):
        correct,total = Constants.ZERO , Constants.ZERO
        with torch.no_grad():
            for images, labels in self.test_data:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, Constants.ONE)
                total += labels.size(Constants.ZERO)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set: {100 * correct / total:.2f}%")

    def predict(self, image_path, class_names):

        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img_tensor = transform(image).unsqueeze(Constants.ZERO).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, Constants.ONE)
            pred_class = class_names[predicted.item()]
            confidence = torch.softmax(outputs, dim=1)[0][predicted.item()]
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)

        draw.text((10, 10), f"Predicted: {pred_class}  || Confidence: {confidence.item():.2f}", fill=(255, 0, 0))

        plt.imshow(draw_img)
        plt.axis("off")
        plt.show()

        return pred_class
    
    def save_model(self, path=Constants.MODEL_PATH):
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load_model(self, path=Constants.MODEL_PATH):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model weights loaded from {path}")
        return self.model
    
    def __del__(self):
        self.writer.close()


