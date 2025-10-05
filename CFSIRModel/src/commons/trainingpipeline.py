from src.datapreprocessing.dataloader import CFSIRDataLoader
from src.commons.model import CFSIRModel
from src.constant.constant import Constants
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_data,self.test_data = CFSIRDataLoader().load_data(Constants.DATA_DIR_PATH,batch_size=self.batch_size)

    def train(self):
        # outer loop for epcohs 
        for epoch in range(self.epochs):
            running_loss = Constants.ZERO
            progress_bar = tqdm(self.train_data, desc=f"Epoch [{epoch+1}/{self.epochs}]", leave=False)

            for images,labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad() 
                model_pred = self.model(images)
                loss = self.criterion(model_pred,labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                
            print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {running_loss/len(self.train_data):.4f}")

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

        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)

        draw.text((10, 10), f"Predicted: {pred_class}", fill=(255, 0, 0))

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


