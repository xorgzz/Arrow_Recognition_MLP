from gen_lib import gen_data_set
from mlp_utils import load_model, predict


def main():
    print("--- TESTOWANIE MODELU ---")
    net = load_model("model_strzalek.pth")
    if not net:
        print("Brak modelu!")
        return

    patterns, labels = gen_data_set()
    sums = 0
    alls = len(patterns)

    for i in range(alls):
        result = predict(net, patterns[i])
        if result == labels[i]:
            sums += 1

    print(f"Wynik testu: {sums}/{alls}")
    print(f"Skuteczność: {((sums / alls) * 100):.2f}%")


if __name__ == '__main__':
    main()