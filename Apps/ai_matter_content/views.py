from django.http import JsonResponse
from django.shortcuts import render


# Create your views here.

def identify(request):
    print(request.method)
    return JsonResponse({'message': 'unknown methods',
                         'code': 50012})
