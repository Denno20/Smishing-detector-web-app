#views.py
#Required imports
from .scripts.CombinedModel.combinedClassifier import CombinedClassifier
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import json

#Define path for model
path = 'predictor/scripts/CombinedModel/'
model = None

#Typically when the server is started
#But this prevents the model from constantly reloading
if model is None:
    model = CombinedClassifier()
    model.load_model(path + "voting_classifier.joblib", path + "vectorizer.joblib")

#Return main page
def index(request):
    return render(request, "predictor/index.html")


def predict(request):
    #Get prediction from model and return it to web page
    predictions = model.predict([request.POST.get("sms")])
    return JsonResponse({'prediction':str(predictions[0][0]), 'value': str(predictions[0][1])})





