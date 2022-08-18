import json
import random

random.seed(42)
fname = "datasets/benchmarks/paws/dev_and_test.json"

with open(fname, "r") as f:
    data = [json.loads(e) for e in f.readlines()]
random.shuffle(data)

labels = {0: [], 1: []}
for e in data:
    labels[e["label"]].append(e)

dev, test = [], []

for label, label_item in labels.items():
    dev.extend(label_item[: len(label_item) // 2])
    test.extend(label_item[len(label_item) // 2 :])

print(len(dev) + len(test))

print(sum([e["label"] for e in dev]) / len(dev))
print(sum([e["label"] for e in test]) / len(test))

with open(
    "datasets/benchmarks/paws/validation.json",
    "w",
) as f:
    for e in dev:
        f.write(json.dumps(e) + "\n")
with open(
    "datasets/benchmarks/paws/test.json",
    "w",
) as f:
    for e in test:
        f.write(json.dumps(e) + "\n")
