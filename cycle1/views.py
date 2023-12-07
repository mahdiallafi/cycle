from django.shortcuts import render
from .knn_script import run_knn
from .forms import CycleForm

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
    return render(request, 'fill_form.html')

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
            funActivities= form.cleaned_data['funActivities']
            max_time = form.cleaned_data['maxTime']
            min_time = form.cleaned_data['minTime']

            # Process the data using your Python script
            similar_items = run_knn(mapped_age, gender, max_time, min_time,history,art, nature,sights,funActivities)

            # Pass similar_items to the template
            return render(request, 'knn_results.html', {'similar_items': similar_items})
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
