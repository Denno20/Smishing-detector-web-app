from django.urls import path

from . import views

#Define url path for index page and hidden predict page
urlpatterns = [
    path("", views.index, name="index"),
    path("predict", views.predict, name="predict")
]