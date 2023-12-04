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
            age = form.cleaned_data['age']
            gender = form.cleaned_data['gender']
            interests = form.cleaned_data['interests']
            poi = form.cleaned_data['poi']
            max_time = form.cleaned_data['maxTime']
            min_time = form.cleaned_data['minTime']

            # Process the data using your Python script
            similar_items = run_knn(age, gender, interests, poi, max_time, min_time)

            # Pass similar_items to the template
            return render(request, 'knn_results.html', {'similar_items': similar_items})
    else:
        form = CycleForm()

    return render(request, 'test.html', {'form': form})
