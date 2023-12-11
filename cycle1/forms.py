# knn_app/forms.py
from django import forms

class CycleForm(forms.Form):
    age = forms.IntegerField(min_value=1, max_value=100)
    gender = forms.ChoiceField(choices=[('male', 'Male'), ('female', 'Female'), ('others', 'Others')])
    targetDistance = forms.IntegerField()
    history = forms.IntegerField(min_value=1, max_value=100) 
    art = forms.IntegerField(min_value=1, max_value=100) 
    nature = forms.IntegerField(min_value=1, max_value=100) 
    sights = forms.IntegerField(min_value=1, max_value=100) 
    museums = forms.IntegerField(min_value=1, max_value=100) 
    churches = forms.IntegerField(min_value=1, max_value=100) 
    funActivities = forms.IntegerField(min_value=1, max_value=100) 
    origin = forms.CharField(max_length=255)
    destination = forms.CharField(max_length=255)
