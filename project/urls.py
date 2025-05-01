from django.urls import path, include

urlpatterns = [
    path('api/', include('trust_api.urls')),
] 