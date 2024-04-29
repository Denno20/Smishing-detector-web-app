#Import libraries
import pandas as pd
from combinedClassifier import CombinedClassifier
import warnings
warnings.filterwarnings('ignore')

#Create classifier
model = CombinedClassifier()

decision = input("Train the model? Y/N").upper()

if decision == "Y":
    model.load_dataset("./Datasets/combined_dataset.csv")
    #The option to save the model appears in this function
    #WARNING: Saving the model will OVERWRITE the current model joblib files saved in memory
    model.train()
else:
    model.load_model("voting_classifier.joblib", "vectorizer.joblib")

#Example data
unseen = ["You have won $5000. The prize needs to be claimed ASAP. Please reply with your bank information so we can deposit the money into your account.", "Hey, see you soon!"]
predicted = model.predict(pd.Series(unseen))
print(predicted)