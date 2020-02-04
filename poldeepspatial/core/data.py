import random


def stratify_labels(data_xs, data_ys):
    groups = group_by_label(data_xs, data_ys)

    min_count = min([len(group) for group in groups.values()])

    xs_out = []
    ys_out = []

    for (label, items) in groups.items():
        xs_out += random.choices(items, k=min_count)
        ys_out += [label] * min_count

    return xs_out, ys_out


def group_by_label(data_xs, data_ys):
    groups = {}
    for (x, y) in zip(data_xs, data_ys):
        if y not in groups:
            groups[y] = []
        groups[y].append(x)
    return groups


def format_stats(data_xs, data_ys):
    groups = group_by_label(data_xs, data_ys)
    texts = []
    texts.append("%4d %-10s" % (len(data_xs), "total"))
    for (label, items) in groups.items():
        texts.append("%4d %-10s" % (len(items), label))
    return "; ".join(texts)
