from accelerate import Accelerator, DeepSpeedPlugin
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

class SimpleNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    input_dim = 10
    hidden_dim = 256
    output_dim = 2

    batch_size = 64
    data_size = 10000

    input_data = torch.randn(data_size, input_dim)
    target_data = torch.randn(data_size, output_dim)

    dataset = torch.utils.data.TensorDataset(input_data, target_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNet(input_dim, hidden_dim, output_dim)

    deepspeed = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
    )
    accelerator = Accelerator(deepspeed_plugin=deepspeed)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(1000):
        model.train()
        for batch in dataloader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
        print(f"Epoch {epoch} loss: {loss.item()}")
    print("Training complete")

    # Save Model
    accelerator.save_model(model.state_dict(), "model.pt")

    # 启动方式
    # python -m accelerate.commands.launch --main_process_port 29501 demo.py

    