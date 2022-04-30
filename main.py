from classifier.classifier import LinearBinaryClassifier
from classifier.utils import draw_statistics, generate_point_set, print_statistics

def main():
    train_X, train_y = generate_point_set(50)

    classifier = LinearBinaryClassifier(train_X, train_y, mi=0.1, w=[1, 1], b=0)
    classifier.train()
    print_statistics(classifier)
    draw_statistics(train_X, classifier)

if __name__ == "__main__":
    main()