from sklearn import svm

def main():
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC()
    clf.fit(X, y)

    print(clf.predict([[2., 2.]]))
