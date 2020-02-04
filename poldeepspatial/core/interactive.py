import os

from classifier import FeatureGenerator, SpatialClassifier

model_path = os.path.join("models", "spatial_tr_si_lm_classifier.h5")
fastext_path = os.path.join("resources", "kgr10.plain.skipgram.dim300.neg10.bin")
feature_generator = FeatureGenerator(fastext_path)
classifier = SpatialClassifier.load_model(feature_generator, model_path)

while True:
    print()
    x = input("Enter (trajector indicator landmark): ")
    if x == '':
        exit(0)

    item = [word.strip() for word in x.split(" ")]
    if len(item) == 3:
        label = classifier.predict(item[0], item[1], item[2])
        print("%s: %s" % (label, str(item)))
        pass
    else:
        print("Error: invalid input")
