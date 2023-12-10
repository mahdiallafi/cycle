from django.shortcuts import render
from .knn_script import run_knn
from .forms import CycleForm
import pandas as pd

# Create your views here.
def home_screen_view(request):
   print(request.headers)
   return render(request,"test.html",{})

def knn_view(request):
    result = run_knn()
    return render(request, 'knn_results.html', {'result': result})   

def about_us(request):
    return render(request, 'aboutus.html')

def services(request):
    return render(request, 'services.html')

def fill_form(request):
    cities = pd.read_csv('/home/mahdi/cycleing/python/cycleapp/cycle1/timStuff/places.csv', delimiter=';', low_memory=False)
    city_list = cities.iloc[:, 1].tolist()
    return render(request, 'fill_form.html',{'city_list': city_list})

def show_in_map(request, latitude, longitude):
    latitude = float(latitude)
    longitude = float(longitude)
    # Your view logic here
    return render(request, 'show_in_map.html', {'latitude': latitude, 'longitude': longitude})


def submit_form(request):
    if request.method == 'POST':
        form = CycleForm(request.POST)
        
        if form.is_valid():
            # Extract form data
            age_input = form.cleaned_data['age']

            # Map age to range
            mapped_age = map_age_to_range(age_input)  # Map age to the desired range (as a string)
            history = form.cleaned_data['history']
            art = form.cleaned_data['art']
            print(f"Age received in view: {mapped_age}")
            
            gender = form.cleaned_data['gender']
            nature = form.cleaned_data['nature']
            sights = form.cleaned_data['sights']
            funActivities = form.cleaned_data['funActivities']
            maxDestination = form.cleaned_data['maxDestination']
            minDestination = form.cleaned_data['minDestination']
            origin = form.cleaned_data['origin']  # Corrected from 'orgin' to 'origin'
            destination = form.cleaned_data['destination']

            # Process the data using your Python script
            similar_items = run_knn(mapped_age, gender, maxDestination, minDestination, history, art, nature, sights, funActivities, origin, destination)

            result_dict_list = similar_items
            # Pass similar_items to the template
            return render(request, 'knn_results.html', {'result': result_dict_list})
    else:
        form = CycleForm()

    return render(request, 'test.html', {'form': form})



def map_age_to_range(age):
    if age <= 6:
        return '6-15'
    elif age <= 16:
        return '16-25'
    elif age <= 26:
        return '26-35'
    elif age <= 36:
        return '36-45'
    elif age <= 46:
        return '46-55'
    elif age <= 56:
        return '56-65'
    elif age <= 66:
        return '66-75'
    elif age <= 75:
        return '75+'
    else:
        return 'Invalid Age'
