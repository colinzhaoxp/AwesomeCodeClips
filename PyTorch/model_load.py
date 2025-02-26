import torch
import torch.nn as nn

# 定义模型结构
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = MyModel()

# 加载权重，需要注意这里的模型权重还在cpu上
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device("cpu")))

# 设置为评估模式
model.eval()

# 准备输入数据（假设输入是 10 维的）
input_data = torch.randn(1, 10)

# 进行推理
with torch.no_grad():
    output = model(input_data)

print("Output:", output)