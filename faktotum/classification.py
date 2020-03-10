class Classifier:
    @staticmethod
    def _build_model(self, input_shape, conv1d=False):
        model = models.Sequential()
        if conv1d:
            sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)
            x = Conv1D(128, 5, activation='relu')(embedded_sequences)
            x = MaxPooling1D(5)(x)
            x = Conv1D(128, 5, activation='relu')(x)
            x = MaxPooling1D(5)(x)
            x = Conv1D(128, 5, activation='relu')(x)
            x = GlobalMaxPooling1D()(x)
            x = Dense(128, activation='relu')(x)
            preds = Dense(len(labels_index), activation='softmax')(x)

        else:
            model.add(layers.Dense(64, activation="relu", input_shape=(input_shape,)))
            model.add(layers.Dense(64, activation="relu"))
            model.add(layers.Dense(1))

        model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
        return model

    def fit(self, X_train, y_train, epochs=100, conv1d=False):
        model = self._build_model(self.X_train.shape[1], conv1d)
        return model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=1)
