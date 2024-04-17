# # https://www.kaggle.com/code/llabhishekll/text-preprocessing-and-sms-spam-detection/notebook
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from nltk.corpus import stopwords
# from keras.layers import SimpleRNN
# from nltk.stem import WordNetLemmatizer
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# import pickle, joblib

# # Import the models from sklearn
# from sklearn.naive_bayes import MultinomialNB, BernoulliNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier

# from sklearn.metrics import f1_score
# from sklearn.model_selection import learning_curve,validation_curve
# from sklearn.model_selection import KFold

# # importing voting classifier
# from sklearn.ensemble import VotingClassifier
# from scikeras.wrappers import KerasClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score
# from keras.models import Sequential

# from keras.layers import Conv1D, Flatten, Dense, Input, Embedding, MaxPooling1D
# from sklearn.linear_model import SGDClassifier 

# lemmatizer = WordNetLemmatizer()
# stopset = set(stopwords.words("english"))


# class CombinedClassifier:
#     def __init__(self):
#         self.vectorizer = TfidfVectorizer()

#         self.A = MultinomialNB(alpha=1.0,fit_prior=True)
#         self.B = DecisionTreeClassifier()
#         self.C = RandomForestClassifier(n_estimators=50)
#         self.D = MLPClassifier(early_stopping=True, batch_size=128, random_state=42, verbose=False)
#         # self.E = SVC(probability=True)
#         self.E = SGDClassifier(loss='log_loss', penalty='l2', max_iter=10000, tol=1e-5)
#         self.F = LogisticRegression(penalty='l1',solver='liblinear',C=1.0,random_state=50)

#         self.G = BernoulliNB(alpha=1.0, fit_prior=True)
#         self.classifiers = [self.A,self.B,self.D,self.G]
#         self.classifiers = [self.B,self.D,self.G]



#     def load_dataset(self, file):
#         # pd.read_csv("../Datasets/sms.csv", encoding="latin-1")
#         self.data = pd.read_csv(file, encoding="latin-1")

#     # function to train classifier
#     def train_classifier(self, clf, X_train, y_train):    
#         clf.fit(X_train, y_train)
        

#     # function to predict features 
#     def predict_labels(self, clf, features):
#         return(clf.predict(features))

#         # Preprocess the new data
#     def preprocess_new_data(self, new_data):
#         preprocessed_data = new_data.apply(self.standardise_text)
#         return preprocessed_data

#     # Transform the preprocessed data into the same format as training data
#     def transform_new_data(self, preprocessed_data):
#         X_new = self.vectorizer.transform(preprocessed_data)
#         return X_new

#     # Predict labels for the new data
#     def predict_new_data(self, clf, X_new):
#         y_pred_new = clf.predict(X_new)
#         return y_pred_new
        
#     #My code
#     def standardise_text(self,data):
#         punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
#         data = str(data).lower()

#         #Remove puncutation
#         # for i in punctuation:
#         #     if i in data:
#         #         data = data.replace(i, "")
        
#         #Remove stop words
#         for word in stopwords.words("english"):
#             if word in data:
#                 data.replace(word, "")
        
#         filtered = []
#         #Lemmonise words
#         for word in data.split(" "):
#             filtered.append(lemmatizer.lemmatize(word))

#         #Return filtered array as string
#         return ' '.join(filtered).strip()
    
#     def load_model(self):
#         self.votingClassifier = joblib.load('voting_classifier.joblib')
#         self.vectorizer = joblib.load('vectorizer.joblib')
#         print(self.votingClassifier)
    
#     #My code
#     def save_model(self):
#         c = input("Save model? Y/N")

#         if c == "Y":
#             joblib.dump(self.votingClassifier, 'voting_classifier.joblib')
#             joblib.dump(self.vectorizer, 'vectorizer.joblib')
#             print("Model saved")

#     def train(self):
#         data = self.data
#         #Convert label to numerical variable
#         data["Class"] = data["Label"].map({'ham':0, 'smish': 1})

#         # # Count the number of words in each Text
#         # data['Count']=0
#         # for i in np.arange(0,len(data.Text)):
#         #     data.loc[i,'Count'] = len(data.loc[i,'Text'])

#         # Extract feature column 'Text'
#         X = self.vectorizer.fit_transform(data.Text.apply(self.standardise_text))
#         # Extract target column 'Class'
#         y = data.Class

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80)



#         cv_score = []
#         for c in self.classifiers:
#             scores = cross_val_score(c, X_train, y_train, cv=5, scoring='accuracy')
#             cv_score.append(scores.mean())
#             print(scores)

#         total_score = sum(cv_score)
#         weights = [score / total_score for score in cv_score]
#         print(weights)

#         self.A.fit(X_train, y_train)
#         self.G.fit(X_train, y_train)

#         self.votingClassifier = VotingClassifier(estimators=[
#         # ('NB', self.A),
#         ('DTC', self.B),
#         ('NN', self.D),
#         ('BNB', self.G)
#         ], 
#         voting='hard', 
#         weights=weights,
#         # weights=[1,2]
#         verbose=True  
#         )


#         pred_val = [0,0,0,0]
#         pred_val = [0,0,0]


#         for a in range(0,(len(self.classifiers))):
#             self.train_classifier(self.classifiers[a], X_train, y_train)
#             y_pred = self.predict_labels(self.classifiers[a],X_test)
#             pred_val[a] = f1_score(y_test, y_pred) 
#             print(pred_val[a])
        
#         self.votingClassifier.fit(X_train, y_train)
#         print(self.votingClassifier)
#         self.A.fit(X_train, y_train)
#         self.G.fit(X_train, y_train)
#         self.save_model()


    # def predict(self, sample_messages):
    #     # Transform the sample dataset
    #     sample_features = self.vectorizer.transform([self.standardise_text(message) for message in sample_messages])
    #     # Perform classification on the sample dataset
    #     sample_predictions = self.votingClassifier.predict(sample_features)
    #     # print(self.votingClassifier.predict_proba(sample_features))


    #     for c in self.classifiers:
    #         print(c.predict(sample_features))

    #     return sample_predictions

# # https://www.kaggle.com/code/llabhishekll/text-preprocessing-and-sms-spam-detection/notebook
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from nltk.corpus import stopwords
# from keras.layers import SimpleRNN
# from nltk.stem import WordNetLemmatizer
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# import pickle, joblib

# # Import the models from sklearn
# from sklearn.naive_bayes import MultinomialNB, BernoulliNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier

# from sklearn.metrics import f1_score
# from sklearn.model_selection import learning_curve,validation_curve
# from sklearn.model_selection import KFold

# # importing voting classifier
# from sklearn.ensemble import VotingClassifier
# from scikeras.wrappers import KerasClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score
# from keras.models import Sequential

# from keras.layers import Conv1D, Flatten, Dense, Input, Embedding, MaxPooling1D
# from sklearn.linear_model import SGDClassifier 

# lemmatizer = WordNetLemmatizer()
# stopset = set(stopwords.words("english"))


# class CombinedClassifier:
#     def __init__(self):
#         self.vectorizer = TfidfVectorizer()

#         self.A = MultinomialNB(alpha=1.0,fit_prior=True)
#         self.B = DecisionTreeClassifier()
#         self.C = RandomForestClassifier(n_estimators=50)
#         self.D = MLPClassifier(early_stopping=True, batch_size=128, random_state=42, verbose=False)
#         # self.E = SVC(probability=True)
#         self.E = SGDClassifier(loss='log_loss', penalty='l2', max_iter=10000, tol=1e-5)
#         self.F = LogisticRegression(penalty='l1',solver='liblinear',C=1.0,random_state=50)

#         self.G = BernoulliNB(alpha=1.0, fit_prior=True)
#         self.classifiers = [self.A,self.B,self.D,self.G]
#         self.classifiers = [self.B,self.D,self.G]



#     def load_dataset(self, file):
#         # pd.read_csv("../Datasets/sms.csv", encoding="latin-1")
#         self.data = pd.read_csv(file, encoding="latin-1")

#     # function to train classifier
#     def train_classifier(self, clf, X_train, y_train):    
#         clf.fit(X_train, y_train)
        

#     # function to predict features 
#     def predict_labels(self, clf, features):
#         return(clf.predict(features))

#         # Preprocess the new data
#     def preprocess_new_data(self, new_data):
#         preprocessed_data = new_data.apply(self.standardise_text)
#         return preprocessed_data

#     # Transform the preprocessed data into the same format as training data
#     def transform_new_data(self, preprocessed_data):
#         X_new = self.vectorizer.transform(preprocessed_data)
#         return X_new

#     # Predict labels for the new data
#     def predict_new_data(self, clf, X_new):
#         y_pred_new = clf.predict(X_new)
#         return y_pred_new
        
#     #My code
#     def standardise_text(self,data):
#         punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
#         data = str(data).lower()

#         #Remove puncutation
#         # for i in punctuation:
#         #     if i in data:
#         #         data = data.replace(i, "")
        
#         #Remove stop words
#         for word in stopwords.words("english"):
#             if word in data:
#                 data.replace(word, "")
        
#         filtered = []
#         #Lemmonise words
#         for word in data.split(" "):
#             filtered.append(lemmatizer.lemmatize(word))

#         #Return filtered array as string
#         return ' '.join(filtered).strip()
    
#     def load_model(self):
#         self.votingClassifier = joblib.load('voting_classifier.joblib')
#         self.vectorizer = joblib.load('vectorizer.joblib')
#         print(self.votingClassifier)
    
#     #My code
#     def save_model(self):
#         c = input("Save model? Y/N")

#         if c == "Y":
#             joblib.dump(self.votingClassifier, 'voting_classifier.joblib')
#             joblib.dump(self.vectorizer, 'vectorizer.joblib')
#             print("Model saved")

#     def train(self):
#         data = self.data
#         #Convert label to numerical variable
#         data["Class"] = data["Label"].map({'ham':0, 'smish': 1})

#         # # Count the number of words in each Text
#         # data['Count']=0
#         # for i in np.arange(0,len(data.Text)):
#         #     data.loc[i,'Count'] = len(data.loc[i,'Text'])

#         # Extract feature column 'Text'
#         X = self.vectorizer.fit_transform(data.Text.apply(self.standardise_text))
#         # Extract target column 'Class'
#         y = data.Class

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80)



#         cv_score = []
#         for c in self.classifiers:
#             scores = cross_val_score(c, X_train, y_train, cv=5, scoring='accuracy')
#             cv_score.append(scores.mean())
#             print(scores)

#         total_score = sum(cv_score)
#         weights = [score / total_score for score in cv_score]
#         print(weights)

#         self.A.fit(X_train, y_train)
#         self.G.fit(X_train, y_train)

#         self.votingClassifier = VotingClassifier(estimators=[
#         # ('NB', self.A),
#         ('DTC', self.B),
#         ('NN', self.D),
#         ('BNB', self.G)
#         ], 
#         voting='hard', 
#         weights=weights,
#         # weights=[1,2]
#         verbose=True  
#         )


#         pred_val = [0,0,0,0]
#         pred_val = [0,0,0]


#         for a in range(0,(len(self.classifiers))):
#             self.train_classifier(self.classifiers[a], X_train, y_train)
#             y_pred = self.predict_labels(self.classifiers[a],X_test)
#             pred_val[a] = f1_score(y_test, y_pred) 
#             print(pred_val[a])
        
#         self.votingClassifier.fit(X_train, y_train)
#         print(self.votingClassifier)
#         self.A.fit(X_train, y_train)
#         self.G.fit(X_train, y_train)
#         self.save_model()


#     def predict(self, sample_messages):
#         # Transform the sample dataset
#         sample_features = self.vectorizer.transform([self.standardise_text(message) for message in sample_messages])
#         # Perform classification on the sample dataset
#         sample_predictions = self.votingClassifier.predict(sample_features)
#         # print(self.votingClassifier.predict_proba(sample_features))


#         for c in self.classifiers:
#             print(c.predict(sample_features))

#         return sample_predictions

# https://www.kaggle.com/code/llabhishekll/text-preprocessing-and-sms-spam-detection/notebook
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from keras.layers import SimpleRNN
from nltk.stem import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle, joblib

# Import the models from sklearn
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve,validation_curve
from sklearn.model_selection import KFold

# importing voting classifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from keras.models import Sequential

from keras.layers import Conv1D, Flatten, Dense, Input, Embedding, MaxPooling1D
import re

lemmatizer = WordNetLemmatizer()
stopset = set(stopwords.words("english"))


class CombinedClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

        self.A = MultinomialNB(alpha=1.0,fit_prior=False)
        self.B = AdaBoostClassifier(n_estimators=100)
        self.C = RandomForestClassifier(n_estimators=100)
        self.D = MLPClassifier(early_stopping=True, batch_size=128, verbose=False)
        self.E = BernoulliNB(alpha=1.0, fit_prior=False)
        self.F = DecisionTreeClassifier()


        self.classifiers = [self.A,self.B,self.C,self.D,self.E, self.F]
        # self.classifiers = [self.A,self.B,self.D, self.E, self.F]
        # self.classifiers = [self.A,self.B,self.D, self.E]
        # self.classifiers = [self.A,self.B,self.D]
        # self.classifiers = [self.B,self.C,self.D,self.E]


    def load_dataset(self, file):
        # pd.read_csv("../Datasets/sms.csv", encoding="latin-1")
        self.data = pd.read_csv(file, encoding="latin-1")

    # function to train classifier
    def train_classifier(self, clf, X_train, y_train):    
        clf.fit(X_train, y_train)
        

    # function to predict features 
    def predict_labels(self, clf, features):
        return(clf.predict(features))

        # Preprocess the new data
    def preprocess_new_data(self, new_data):
        preprocessed_data = new_data.apply(self.standardise_text)
        return preprocessed_data

    # Transform the preprocessed data into the same format as training data
    # def transform_new_data(self, preprocessed_data):
    #     X_new = self.vectorizer.transform(preprocessed_data)
    #     return X_new

    def transform_new_data(self,preprocessed_data):
        X_new = self.vectorizer.fit_on_texts(preprocessed_data)
        self.sequences = self.tokeniser.texts_to_sequences(X_new)
        self.sequences_matrix = sequence.pad_sequences(self.sequences,maxlen=200)


    # Predict labels for the new data
    def predict_new_data(self, clf, X_new):
        y_pred_new = clf.predict(X_new)
        return y_pred_new
        
    #My code
    def standardise_text(self,data):
        punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        data = str(data).lower()
        data = re.sub('[a-zA-Z]', data)

        # #Remove puncutation
        # for i in punctuation:
        #     if i in data:
        #         data = data.replace(i, "")
        
        #Remove stop words
        for word in stopwords.words("english"):
            if word in data:
                data.replace(word, "")
        
        filtered = []
        #Lemmonise words
        for word in data.split(" "):
            filtered.append(lemmatizer.lemmatize(word))

        #Return filtered array as string
        return ' '.join(filtered).strip()
    
    def load_model(self, model, vectorizer):
        self.votingClassifier = joblib.load(model)
        self.vectorizer = joblib.load(vectorizer)
        print(self.votingClassifier)
    
    #My code
    def save_model(self):
        c = input("Save model? Y/N")

        if c == "Y":
            joblib.dump(self.votingClassifier, 'voting_classifier.joblib')
            joblib.dump(self.vectorizer, 'vectorizer.joblib')
            print("Model saved")

    def train(self):
        data = self.data
        #Convert label to numerical variable
        data["Class"] = data["Label"].map({'ham':0, 'smish': 1})

        # Count the number of words in each Text
        data['Count']=0
        for i in np.arange(0,len(data.Text)):
            data.loc[i,'Count'] = len(data.loc[i,'Text'])


        corpus = []

        for i in range(0, len(data.Text)):
            message = re.sub('[^a-zA-Z]', ' ', data.Text[i])
            message = message.lower()
            message = message.split()
            message = [lemmatizer.lemmatize(word) for word in message if not word in stopwords.words('english')]
            message = ' '.join(message)
            corpus.append(message)

        # Extract feature column 'Text'
        X = self.vectorizer.fit_transform(corpus).toarray()
        # Extract target column 'Class'
        y = data.Class

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, train_size=0.90)



        cv_score = []
        for c in self.classifiers:
            scores = cross_val_score(c, X_train, y_train, cv=5, scoring='accuracy')
            cv_score.append(scores.mean())
            print(scores)

        total_score = sum(cv_score)
        weights = [score / total_score for score in cv_score]
        print(weights)

        self.votingClassifier = VotingClassifier(estimators=[
        ('NB', self.A),
        ('ABC', self.B),
        ('RF', self.C),
        ('NN', self.D),
        ('BNB', self.E),
        ('KNN', self.F)
        ], 
        voting='soft', 
        weights=weights,
        verbose=True  
        )


        pred_val = [0,0,0,0,0,0]


        for a in range(0,(len(self.classifiers))):
            self.train_classifier(self.classifiers[a], X_train, y_train)
            y_pred = self.predict_labels(self.classifiers[a],X_test)
            pred_val[a] = f1_score(y_test, y_pred) 
            print(pred_val[a])
        
        self.votingClassifier.fit(X_train, y_train)
        print(self.votingClassifier)
        self.save_model()



    def predict(self, sample_messages):
        # Transform the sample dataset
        corpus = []

        for i in range(0, len(sample_messages)):
            message = re.sub('[^a-zA-Z]', ' ', sample_messages[i])
            message = message.lower()
            message = message.split()
            message = [lemmatizer.lemmatize(word) for word in message if not word in stopwords.words('english')]
            message = ' '.join(message)
            corpus.append(message)

        # Extract feature column 'Text'
        sample_features = self.vectorizer.transform(corpus).toarray()

        # Perform classification on the sample dataset
        predictions = self.votingClassifier.predict_proba(sample_features)
        print(self.votingClassifier.predict(sample_features))

        class_predictions = [("smish", p[1]) if p[1] > 0.5 else ("ham", p[1]) for p in predictions]
        return class_predictions
