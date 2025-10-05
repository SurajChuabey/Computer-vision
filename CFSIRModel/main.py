from src.utility.utils import Utility
from src.commons.trainingpipeline import TrainingPipeline
from src.constant.constant import Constants
import torch 

def main():
    Utility.SeeDataset()
    print(f"[MODEL] Starting Training .........")
    device = torch.device(Constants.CUDA if torch.cuda.is_available() else Constants.CPU)

    # training params
    epochs = int(Utility.config(Constants.MODEL_PARAMETERS,Constants.EPOCHS))
    learning_rate = float(Utility.config(Constants.MODEL_PARAMETERS,Constants.LEARNING_RATE))
    batch_size = int(Utility.config(Constants.DATASET,Constants.BATCH_SIZE))
    save_model_path = Utility.config(Constants.MODEL_PARAMETERS,Constants.SAVED_MODEL_PATH)

    model = TrainingPipeline(epochs=epochs,learning_rate=learning_rate,batch_size=batch_size,device=device)
    model.train()
    model.save_model(save_model_path)

    # model evaluation
    print(f"[MODEL] Starting Model Evaulation .........")
    model.eval()

    # model testing
    print(f"[MODEL] Running inference on single frame .........")
    model.predict(image_path='results/birds1.jpeg',class_names=Constants.CLASSES)
    
if __name__ == "__main__":
    main()
