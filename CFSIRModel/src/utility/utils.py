from src.datapreprocessing.preprocessing import CFSIRDataset
from src.constant.constant import Constants
import configparser

class Utility:

    @staticmethod
    def SeeDataset():
        dataset = CFSIRDataset(Constants.DATA_DIR_PATH,train=False)
        print(f"Dataset length: {len(dataset)}")
        sample_img, sample_label = dataset[0]
        print(f"Sample image shape: {sample_img.shape}, Sample label: {sample_label}")

    @staticmethod
    def read_ini_file(file_path):

        config = configparser.ConfigParser()
        config.read(file_path)

        config_dict = {}
        for section in config.sections():
            config_dict[section] = {}
            for key, value in config.items(section):
                config_dict[section][key] = value

        return config_dict
    
    @staticmethod
    def config(env=None,param=None):

        config_dict = Utility.read_ini_file(Constants.CONFIG_FILE_PATH)
        return config_dict[env][param.lower()]
