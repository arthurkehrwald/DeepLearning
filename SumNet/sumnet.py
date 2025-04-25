import typing
import random
from collections import namedtuple
import matplotlib.pyplot as plt
import math

RUNS = 8
LEARN_RATE = 0.00015
EPOCHS = 1
TRAINING_DATA_SIZE = 50

TrainingExample = namedtuple("TrainingExample", ["a", "b", "sum"])
TrainingStats = namedtuple("TrainingStats", ["weight_history", "error_history"])


def infer(input: TrainingExample, weights: typing.List[float]) -> float:
    return input.a * weights[0] + input.b * weights[1]


def calc_error(result: float, expected_result: float) -> float:
    return 0.5 * (result - expected_result) ** 2


def calc_gradient(input: TrainingExample, result: float) -> typing.List[float]:
    g1 = (result - input.sum) * input.a
    g2 = (result - input.sum) * input.b
    return [g1, g2]


def adjust_weights(
    weights: typing.List[float], gradient: typing.List[float], learn_rate: float
) -> typing.List[float]:
    return [w - learn_rate * g for w, g in zip(weights, gradient)]


def gen_starting_weights() -> typing.List[float]:
    # Begin on a random point on a circle with radius 10 with center (1, 1)
    angle = random.random() * math.tau
    return [math.cos(angle) * 10 + 1, math.sin(angle) * 10 + 1]


def gen_training_data(size) -> typing.List[TrainingExample]:
    ret = []
    for _ in range(size):
        a = random.uniform(-100, 100)
        b = random.uniform(-100, 100)
        ret.append(TrainingExample(a, b, a + b))
    return ret


def plot_stats(stats: typing.List[TrainingStats]):
    fig, axs = plt.subplots(ncols=2)

    axs[0].set_title("Gewichte")
    axs[0].set_xlabel("w1")
    axs[0].set_ylabel("w2")

    axs[1].set_title("Fehler")
    axs[1].set_xlabel("Epoche")
    axs[1].set_ylabel("Fehlerfunktion")
    axs[1].set_yscale("log")

    for single_run_stats in stats:
        w1_history = [weights[0] for weights in single_run_stats.weight_history]
        w2_history = [weights[1] for weights in single_run_stats.weight_history]
        axs[0].plot(w1_history, w2_history)
        axs[1].plot(single_run_stats.error_history)

    plt.show()


def train(training_data_size: int, epochs: int, learn_rate: float) -> TrainingStats:
    training_data = gen_training_data(training_data_size)
    weights = gen_starting_weights()
    weight_history: typing.List[typing.List[float]] = [weights]
    error_history: typing.List[float] = []
    for _ in range(epochs):
        random.shuffle(training_data)
        for ex in training_data:
            result = infer(ex, weights)
            expected_result = ex.sum
            error = calc_error(result, expected_result)
            error_history.append(error)
            gradient = calc_gradient(ex, result)
            weights = adjust_weights(weights, gradient, learn_rate)
            weight_history.append(weights)
    return TrainingStats(weight_history, error_history)


def main():
    training_stats = [
        train(TRAINING_DATA_SIZE, EPOCHS, LEARN_RATE) for _ in range(RUNS)
    ]
    plot_stats(training_stats)


if __name__ == "__main__":
    main()
