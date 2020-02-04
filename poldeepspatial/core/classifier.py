import numpy
import keras
from labels import LABEL_SPATIAL, LABEL_OTHER
from keras import layers
from pyfasttext import FastText
from callbacks import ClassAccuracy, Accuracy


class FeatureGenerator:

    def __init__(self, fastext_path):
        self.fasttext = FastText(fastext_path)

    def generate_record(self, tuple):
        tr = self.fasttext.get_numpy_vector(tuple[0])
        si = self.fasttext.get_numpy_vector(tuple[1])
        lm = self.fasttext.get_numpy_vector(tuple[2])
        #return numpy.concatenate((tr, lm))
        #return numpy.concatenate((tr, si, lm))
        return numpy.concatenate((tr, si, lm, lm - tr))
        #return numpy.concatenate((si, lm - tr, tr - lm))

    def generate(self, values):
        return numpy.array([self.generate_record(value) for value in values])


class SpatialClassifier:

    def __init__(self, feature_generator):
        self.gen = feature_generator
        self.model = self.create_model()
        self.score = []

    def create_model(self):
        input_size = len(self.gen.generate_record(("a", "b", "c")))
        inputs = keras.Input(shape=(input_size,))
        x = inputs
        x = layers.Dense(input_size, activation='relu')(x)
        x = layers.Dropout(0.8)(x)
        x = layers.Dense(900, activation='relu')(x)
        x = layers.Dropout(0.8)(x)
        x = layers.Dense(600, activation='relu')(x)
        x = layers.Dropout(0.8)(x)
        x = layers.Dense(300, activation='relu')(x)
        x = layers.Dropout(0.8)(x)
        #x = layers.Dense(100, activation=tf.nn.relu)(x)
        #x = layers.Dropout(0.8)(x)
        #outputs = layers.Dense(2, activation='sigmoid')(x)
        outputs = layers.Dense(2, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='spatial_tr_si_lm_classifier')
        model.summary()

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=['accuracy'])
        return model

    def fit(self, train_x, train_y, test_x, test_y):
        callback_train = ClassAccuracy(train_x, train_y, 1, "Class accuracy on train")
        callback_train_acc = Accuracy(train_x, train_y, "Total accuracy on train")
        callback_test_1 = ClassAccuracy(test_x, test_y, 1, "Class accuracy on test")
        callback_test_0 = ClassAccuracy(test_x, test_y, 0, "Class accuracy on test")
        callback_test_acc = Accuracy(test_x, test_y, "Total accuracy on test")
        callbacks = [callback_train, callback_train_acc, callback_test_1, callback_test_0, callback_test_acc]

        print(train_x.shape)
        self.model.fit(train_x, train_y, epochs=40, validation_split=0.2, verbose=1, callbacks=callbacks)
        self.score = callback_test_acc.avg_score(5) * 100

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)

    def predict(self, trajector, spatial_indicator, landmark):
        x = [trajector, spatial_indicator, landmark]
        x = self.gen.generate([x])
        y = self.model.predict_on_batch(x)[0]
        return LABEL_SPATIAL if y[1] > y[0] else LABEL_OTHER

    def get_score(self):
        return self.score

    @staticmethod
    def load_model(feature_generator, model_path):
        classifier = SpatialClassifier(feature_generator)
        classifier.load(model_path)
        return classifier
