import torch
from sklearn import metrics
from torch.autograd import Variable
import torch.nn.functional as F

from faktotum.research.utils import EarlyStopping


"""
    (word_reprojection_map): Linear(in_features=4396, out_features=4396, bias=True)
    (rnn): GRU(4396, 512)
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (decoder): Linear(in_features=512, out_features=2, bias=True)
"""


class Model(torch.nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 1),
        )

    def forward(self, x):
        return self.features(x)


class Classifier:
    # todo: normalization
    # kleinere learning rate 1e-3
    #  If your target is missing the feature dimension ([batch_size] instead of [batch_size, 1]), an unwanted broadcast might be applied.
    def fit(
        self, X_train, y_train, epochs=1000, lr: float = 1e-3, batch_size: int = 256
    ):
        self._model = Model(X_train.shape[1])
        if torch.cuda.is_available():
            self._model.cuda()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=5, verbose=True)

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

            pred = F.softmax(self.forward(inputs))
            ans = []
            # Pick the class with maximum weight
            for t in pred:
                if t[0] > t[1]:
                    ans.append(0)
                else:
                    ans.append(1)
            return torch.tensor(ans)
            return self._model(inputs).cpu().data.numpy().reshape(1, -1)[0]
