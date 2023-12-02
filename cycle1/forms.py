# knn_app/forms.py
from django import forms

class CycleForm(forms.Form):
    age = forms.IntegerField(min_value=1, max_value=100)
    gender = forms.ChoiceField(choices=[('male', 'Male'), ('female', 'Female'), ('child', 'Child')])
    interests = forms.MultipleChoiceField(choices=[('history', 'History'), ('art', 'Art'), ('nature', 'Nature')], widget=forms.CheckboxSelectMultiple())
    poi = forms.MultipleChoiceField(choices=[('res', 'Restaurant'), ('park', 'Park'), ('mall', 'Mall'), ('hotel', 'Hotel'), ('stadium', 'Stadium')], widget=forms.CheckboxSelectMultiple())
    maxTime = forms.IntegerField()
    minTime = forms.IntegerField()
