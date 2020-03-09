class Classifier:
    @staticmethod
    def _build_model(self, input_shape):
        model = models.Sequential()
        model.add(layers.Dense(64, activation="relu", input_shape=(input_shape,)))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(1))

        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        return model

    def fit(self, X_train, y_train, epochs=100):
        model = self._build_model(self.X_train.shape[1])
        return model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=1)
