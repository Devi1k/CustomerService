from django.urls import path
from . import views

urlpatterns = [
    path('intent', views.intent_cls),
]
