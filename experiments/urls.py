from django.urls import path
from .views import RunExperimentAPIView

urlpatterns = [
    path('run/', RunExperimentAPIView.as_view(), name='run-experiment'),
]
