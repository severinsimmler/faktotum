import torch
from torch.autograd import Variable
from sklearn import metrics


class Model(torch.nn.Module):
    input_size = 3
    output_size = 1

    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, X):
        return self.linear(X)


class Regression:
    def fit(self, X_train, y_train, epochs=100, lr: float = 0.01):
        self._model = Model()
        if torch.cuda.is_available():
            self._model.cuda()
        criterion = torch.nn.MSELoss() 
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)

        for epoch in range(epochs):
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(X_train).cuda()).float()
                labels = Variable(torch.from_numpy(y_train.reshape(-1, 1)).cuda()).float()
            else:
                inputs = Variable(torch.from_numpy(X_train)).float()
                labels = Variable(torch.from_numpy(y_train.reshape(-1, 1))).float()

            optimizer.zero_grad()

            outputs = self._model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            print(f'Epoch {epoch}, loss {loss.item()}')

    def evaluate(self, X_test, y_test):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(X_test).cuda()).float()
            else:
                inputs = Variable(torch.from_numpy(X_test)).float()
            outputs = self._model(inputs).cpu().data.numpy().reshape(1, -1)[0]
            return metrics.mean_squared_error(y_test, outputs)
