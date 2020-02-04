from keras.callbacks import Callback


class ClassAccuracy(Callback):

    def __init__(self, data_x, data_y, class_label, label="class accuracy"):
        super(ClassAccuracy, self).__init__()
        self.label = "%s for %s" % (label, class_label)
        self.class_label = class_label
        self.data_x = data_x
        self.data_y = data_y
        self.scores = []

    def on_epoch_end(self, epoch, logs={}):
        pred_y = self.model.predict_on_batch(self.data_x)
        pred_y_labels = []
        for row in pred_y:
            pred_y_labels.append( 1 if row[1] > row[0] else 0)
        (acc, tp, fp) = self.score_class(self.data_y, pred_y_labels)
        logs[self.label] = acc
        print(' - {:04.2f} (tp: {:3.0f}, fp: {:3.0f} -- {:15s})'.format(acc * 100, tp, fp, self.label))
        self.scores.append(acc)

    def score_class(self, data_y, pred_y):
        tp = 0.0
        fp = 0.0

        for i in range(0, len(data_y)):
            label = data_y[i]
            pred = pred_y[i]
            if pred == self.class_label:
                if label == pred:
                    tp += 1.0
                else:
                    fp += 1.0
        acc = 0.0 if tp+fp == 0.0 else tp/(tp+fp)
        return acc, tp, fp

    def avg_score(self, last_n_epochs):
        last = self.scores[:-last_n_epochs]
        return sum(last)/len(last)


class Accuracy(Callback):

    def __init__(self, data_x, data_y, label="accuracy"):
        super(Accuracy, self).__init__()
        self.label = label
        self.data_x = data_x
        self.data_y = data_y
        self.scores = []

    def on_epoch_end(self, epoch, logs={}):
        pred_y = self.model.predict_on_batch(self.data_x)
        pred_y_labels = []
        for row in pred_y:
            pred_y_labels.append( 1 if row[1] > row[0] else 0)
        (acc, tp, fp) = self.score_class(self.data_y, pred_y_labels)
        logs[self.label] = acc
        print(' - {:04.2f} (tp: {:3.0f}, fp: {:3.0f} -- {:15s})'.format(acc * 100, tp, fp, self.label))
        self.scores.append(acc)

    def score_class(self, data_y, pred_y):
        tp = 0.0
        fp = 0.0

        for i in range(0, len(data_y)):
            label = data_y[i]
            pred = pred_y[i]
            if label == pred:
                tp += 1.0
            else:
                fp += 1.0
        acc = 0.0 if tp+fp == 0.0 else tp/(tp+fp)
        return acc, tp, fp

    def avg_score(self, last_n_epochs):
        last = self.scores[-last_n_epochs:]
        return sum(last)/len(last)
