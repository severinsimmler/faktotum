import torch
from torch.autograd import Variable
from sklearn import metrics


class Model(torch.nn.Module):
    input_size = 3
    output_size = 1

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = X.view(-1, 16 * 5 * 5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X


class Classification:
    def fit(self, X_train, y_train, epochs=100, lr: float = 0.01):
        self._model = Model()
        if torch.cuda.is_available():
            self._model.cuda()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)

        for epoch in range(epochs):
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(X_train).cuda()).float()
                labels = Variable(
                    torch.from_numpy(y_train.reshape(-1, 1)).cuda()
                ).float()
            else:
                inputs = Variable(torch.from_numpy(X_train)).float()
                labels = Variable(torch.from_numpy(y_train.reshape(-1, 1))).float()

            optimizer.zero_grad()

            outputs = self._model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            print(f"Epoch {epoch}, loss {loss.item()}")

    def evaluate(self, X_test, y_test):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(X_test).cuda()).float()
            else:
                inputs = Variable(torch.from_numpy(X_test)).float()
            outputs = self._model(inputs).cpu().data.numpy().reshape(1, -1)[0]
            return metrics.mean_squared_error(y_test, outputs)
