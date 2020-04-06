import torch
from sklearn import metrics, preprocessing
from torch.autograd import Variable

from faktotum.research.utils import EarlyStopping


class Model(torch.nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(input_size, 2500),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2500),
            torch.nn.Linear(2500, 1000),
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 1),
        )

    def forward(self, x):
        return self.features(x)


class Regression:
    def fit(
        self, X_train, y_train, epochs=1000, lr: float = 1e-3, batch_size: int = 256
    ):
        self._model = Model(X_train.shape[1])
        if torch.cuda.is_available():
            self._model.cuda()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=10, verbose=True)

        for epoch in range(epochs):
            inputs = Variable(torch.from_numpy(X_train)).float()
            labels = Variable(torch.from_numpy(y_train.reshape(-1, 1))).float()

            permutation = torch.randperm(inputs.size()[0])

            for i in range(0, inputs.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i : i + batch_size]
                batch_x, batch_y = inputs[indices], labels[indices]

                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()

                optimizer.zero_grad()

                outputs = self._model(batch_x)

                loss = criterion(outputs, batch_y)
                loss.backward()

                optimizer.step()

            print(f"Epoch {epoch}, loss {loss.item()}")

            early_stopping(loss, self._model)

            if early_stopping.early_stop:
                print("Early stopping")
                self._model.load_state_dict(torch.load("checkpoint.pt"))
                return

        torch.save(self._model.state_dict(), "final-model.pt")

    def evaluate(self, X_test, y_test):
        X_test = preprocessing.normalize(X_test)
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(X_test).cuda()).float()
            else:
                inputs = Variable(torch.from_numpy(X_test)).float()
            outputs = self._model(inputs).cpu().data.numpy().reshape(1, -1)[0]
            return (
                metrics.mean_squared_error(y_test, outputs),
                metrics.mean_absolute_error(y_test, outputs),
            )

    def predict(self, X):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(X).cuda()).float()
            else:
                inputs = Variable(torch.from_numpy(X)).float()
            self._model.eval()
            return self._model(inputs).cpu().data.numpy().reshape(1, -1)[0]
