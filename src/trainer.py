import sys
import os
# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 直接导入，避免相对导入问题
from src.model import Model
from utils.utils import DataPreprocessing


class Trainer(Model):
    def __init__(self):
        super().__init__()
        self.model = Model()
    
    def train(self):
        pass

    def visulize(self):
        pass

    def save_model(self):
        pass

if __name__ == "__main__":
    # 测试代码
    trainer = Trainer()
    print('Trainer 初始化成功!')