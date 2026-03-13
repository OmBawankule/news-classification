
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def train_model(X,y):

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    models={
        "Logistic Regression":LogisticRegression(max_iter=1000),
        "Naive Bayes":MultinomialNB(),
        "Linear SVM":LinearSVC()
    }

    best_model=None
    best_acc=0

    for name,model in models.items():

        model.fit(X_train,y_train)

        pred=model.predict(X_test)

        acc=accuracy_score(y_test,pred)

        print(name,"Accuracy:",acc)

        if acc>best_acc:
            best_acc=acc
            best_model=model

    return best_model,X_test,y_test
