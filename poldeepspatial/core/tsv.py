import codecs

def load(filename):
    values = []
    labels = []
    for line in codecs.open(filename, "r", "utf-8"):
        line = line.strip()
        if len(line) == 0:
            continue
        cols = line.split("\t")
        if len(cols) == 5:
            values.append([cols[0]] + cols[2:4])
            labels.append(cols[4])
        else:
            print("[WARN] %s" % line)
    return (values, labels)
