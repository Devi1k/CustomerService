from django.urls import path
from . import views

urlpatterns = [
    path('matterContent', views.identify),
]
