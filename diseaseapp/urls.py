from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('', views.home, name="index"),
    path('predict_form', views.predict_form, name="predict_form"),
    path('predict', views.predict, name="predict"),
    path('login', views.login_view, name="login"),
    path('register', views.signup, name="signup"),
    path('logout', views.logout_view, name="logout"),
    path('appointment', views.appointment, name="appointment"),
    path('appointment_result', views.appointment_result, name="appointment_result"),
    path('doctors', views.get_data, name="doctors"),
    path('about', views.about, name="about"),

]
