from django.urls import path
from .views import InputView, RetrieveView, ClearView

urlpatterns = [
    path('input/', InputView.as_view(), name='input'),
    path('retrieve/', RetrieveView.as_view(), name='retrieve'),
    path('clear/', ClearView.as_view(), name='clear'),
] 