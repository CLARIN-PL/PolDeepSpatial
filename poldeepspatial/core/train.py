import os

from labels import LABEL_SPATIAL, LABEL_OTHER
import data
import tsv
from classifier import FeatureGenerator, SpatialClassifier


def transform_labels(labels):
    labelToId = {}
    labelToId[LABEL_SPATIAL] = 1
    labelToId[LABEL_OTHER] = 0
    return [labelToId[label] for label in labels]


def train_eval(path_train, path_eval, repeats=5, save=False):
    (train_x, train_y) = tsv.load(path_train)
    (test_x, test_y) = tsv.load(path_eval)

    print(data.format_stats(train_x, train_y))
    print(data.format_stats(test_x, test_y))

    #fastext_path = "/bigdata/embeddings/facebookresearch/cc.pl.300.bin"
    fastext_path = os.path.join("resources", "kgr10.plain.skipgram.dim300.neg10.bin")
    feature_generator = FeatureGenerator(fastext_path)

    errors = []
    scores = []
    for n in range(0, repeats):
        classifier = SpatialClassifier(feature_generator)

        tr_train_x = classifier.gen.generate(train_x)
        tr_train_y = transform_labels(train_y)
        tr_test_x = classifier.gen.generate(test_x)
        tr_test_y = transform_labels(test_y)

        classifier.fit(tr_train_x, tr_train_y, tr_test_x, tr_test_y)
        scores.append(classifier.get_score())

        for (t, label) in zip (train_x, train_y):
            decision = classifier.predict(t[0], t[1], t[2])
            if decision != label:
                errors.append(decision + "_" + "_".join(t))

        if n + 1 == repeats and save:
            model_path = os.path.join("models", "spatial_tr_si_lm_classifier.h5")
            classifier.save(model_path)

    scores_str = ", ".join(["%5.2f" % score for score in sorted(scores)])
    print("[Summary] Avg score=%5.2f;  Scores: %s" % (sum(scores)/len(scores), scores_str))

    errors_count = {}
    for error in errors:
        c = errors_count.setdefault(error, 0)
        errors_count[error] = c+1

    for error in sorted(errors_count.items(), key=lambda x: x[1]):
        print("%2d %s" % (error[1], error[0]))


train_eval("data/pst20_14si_train.tsv", "data/pst20_14si_test.tsv", 1, True)
