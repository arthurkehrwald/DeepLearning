import typing
import random
from collections import namedtuple

LEARN_RATE = 0.0001
EPOCHS = 100
TRAINING_DATA_SIZE = 100

TrainingExample = namedtuple("TrainingExample", ["a", "b", "sum"])

def infer(input: TrainingExample, weights: typing.List[float]) -> float:
    return input.a * weights[0] + input.b * weights[1]


def calc_gradient(input: TrainingExample, result: float) -> typing.List[float]:
    g1 = (result - input.sum) * input.a
    g2 = (result - input.sum) * input.b
    return [g1, g2]


def adjust_weights(
    weights: typing.List[float], gradient: typing.List[float]
) -> typing.List[float]:
    return [w - LEARN_RATE * g for w, g in zip(weights, gradient)]


def gen_starting_weights() -> typing.List[float]:
    return [random.uniform(-1, 1), random.uniform(-1, 1)]


def gen_training_data() -> typing.List[TrainingExample]:
    ret = []
    for _ in range(TRAINING_DATA_SIZE):
        a = random.uniform(-100, 100)
        b = random.uniform(-100, 100)
        ret.append(TrainingExample(a, b, a + b))
    return ret


def main():
    training_data = gen_training_data()
    weights = gen_starting_weights()
    for _ in range(EPOCHS):
        random.shuffle(training_data)
        for ex in training_data:
            result = infer(ex, weights)
            gradient = calc_gradient(ex, result)
            weights = adjust_weights(weights, gradient)
    print(weights)


if __name__ == "__main__":
    main()
