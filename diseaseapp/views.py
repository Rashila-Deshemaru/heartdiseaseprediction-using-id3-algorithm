import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from django.shortcuts import redirect, render
from django.core.mail import send_mail
from .decorators import unauthenticated_user, allowed_users
from django.contrib.auth.decorators import login_required
from django.conf import settings


# Section to define the Node class, which represents a node in the decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    # to check if the node is leaf node
    def is_leaf(self):
        return self.value is not None

#creating decision tree module
class DecisionTree:
    #declaring variable using initiailizer
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    #stopping criteria for construction of the decision tree
    def _is_finished(self, depth):
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        return False
    
    # calculating entropy
    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    #creating a split in the data based on a given threshold
    def _create_split(self, X, threshold):
        left_idx = np.argwhere(X <= threshold).flatten()
        right_idx = np.argwhere(X > threshold).flatten()
        return left_idx, right_idx

    # calculating information gain
    def _information_gain(self, X, y, threshold):
        parent_loss = self._entropy(y)
        left_idx, right_idx = self._create_split(X, threshold)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)
        
        child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        return parent_loss - child_loss

    # finding the best feature and threshold to split the data
    def _best_split(self, X, y):
        split = {'score':- 1, 'feat': None, 'thresh': None}

        n_features = X.shape[1]
        for feat in range(n_features):
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']

    # building decision tree
    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)
    
    #traverse tree
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    # fit the decision tree model with the given training data
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    # function for prediction
    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self._traverse_tree(x, self.root))
        return np.array(y_pred)






def home(request):
    return render(request, "index.html")

@login_required(login_url='login')
def predict_form(request):
    return render(request, 'predict_form.html')

@login_required(login_url='login')
def predict(request):
    if request.method == 'POST':
        temp = {}
        temp['ages'] = request.POST.get('ages')
        temp['sex'] = request.POST.get('sex')
        temp['cp'] = request.POST.get('cp')
        temp['trestbps'] = request.POST.get('trestbps')
        temp['chol'] = request.POST.get('chol')
        temp['fbs'] = request.POST.get('fbs')
        temp['restecg'] = request.POST.get('restecg')
        temp['thalach'] = request.POST.get('thalach')
        temp['exang'] = request.POST.get('exang')
        temp['oldpeak'] = request.POST.get('oldpeak')
        temp['slope'] = request.POST.get('slope')
        temp['ca'] = request.POST.get('ca')
        temp['thal'] = request.POST.get('thal')



        # Load your dataset
        path = './model/heart.csv'
        data = pd.read_csv(path)

        # Segmenting the target and feature variable
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values


        # generating train data
        X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a decision tree classifier using decision module
        dtc = DecisionTree(max_depth=500, min_samples_split=2)
        #training the data
        dtc.fit(X_train, y_train)
               

        #take user input
        user_input = np.array([int(temp['ages']), int(temp['sex']), int(temp['cp']), int(temp['trestbps']), int(temp['chol']),
            int(temp['fbs']),int(temp['restecg']),int(temp['thalach']),
            int(temp['exang']),float(temp['oldpeak']), int(temp['slope']),
            int(temp['ca']),int(temp['thal'])])  

        #reshaping 2D user input array into 1D array
        user_input = user_input.reshape(1, -1)

        # predicting using user input
        prediction = dtc.predict(user_input)

        if prediction == 0:
            conclusion = "Heart Disease not Found"
        elif prediction == 1:
            conclusion = "Heart Disease Found"
        else: 
            conclusion = "Wrong"

    return render(request, 'result.html', {'Disease': conclusion})

# Registering Users
from .forms import NewUserForm
from django.shortcuts import  render, redirect ,HttpResponse
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

# For creating the account
def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()

            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)

            login(request, user)
            return redirect('index')

            # return redirect('login')
        else:
            return render(request, 'authentication/register.html', {'form': form})
    
    else:
        form = UserCreationForm()
        return render(request, "authentication/register.html", {'form': form})


# For log in
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')
        else:
            return render(request, 'authentication/login.html', {'form': form})
    
    else:
        form = AuthenticationForm()
        return render(request, 'authentication/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('index')


@login_required(login_url='login')
def appointment(request):
    return render(request, 'appointment.html')

@login_required(login_url='login')
def appointment_result(request):
    your_name = request.POST['your_name']
    your_phone = request.POST['your_phone']
    your_address = request.POST['your_address']
    your_email = request.POST['your_email']
    appointment_day = request.POST['appointment_day']
    appointment_time = request.POST['appointment_time']
    your_doctor = request.POST['your_doctor']
    your_message = request.POST['your_message']
 
    email_message = "Patient name: " + your_name + " Patient Phone number: " + your_phone + " Appointment day: " + appointment_day + " Appointment Time: " + appointment_time + " Patient's Remarks: " + your_message
    
    send_mail(
        your_name,
        email_message,
        your_email,
        ['rashirashila2000@gmail.com']
    )
    return render(request, 'appointment_result.html', {
        'your_name': your_name,
        'your_phone': your_phone, 
        'your_address': your_address,
        'your_email' : your_email,
        'appointment_day' : appointment_day,
        'appointment_time' : appointment_time,
        'your_doctor' : your_doctor,
        'your_message' : your_message
        })

from . models import Doctor
def get_data(request):
    data = Doctor.objects.all()
    return render(request, 'doctor.html', {'data': data})

def about(request):
    return render(request, 'about.html')