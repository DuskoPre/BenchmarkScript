import torch
import time
from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return torch.randn(1000), torch.tensor(0)

def benchmark_training(model, dataset_size=1000, batch_size=32, num_epochs=1):
    dataset = DummyDataset(dataset_size)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    start_time = time.time()
    for epoch in range(num_epochs):
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"Total time for {num_epochs} epoch(s): {total_time} seconds")
    return total_time

# Example usage
class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = torch.nn.Linear(1000, 10)
    def forward(self, x):
        return self.fc(x)

model = DummyModel()
benchmark_training(model)
