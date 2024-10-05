README 2

Team member names and email addresses
    Andrew King - kingand20@students.ecu.edu
    Samese Heyward - heywards20@students.ecu.edu
Quick Start

First, we need to install a Python 3 virtual environment with:

sudo apt-get install python3-venv
Create a virtual environment:

python3 -m venv python_venv
You need to activate the virtual environment when you want to use it:

source python_venv/bin/activate
To fufil all the requirements for the python server, you need to run:

pip3 install -r requirements.txt

To use KElbowVisualizer to decide K for K-means
Install:
sudo apt-get update

sudo apt-get install python3-tk

(optional if you have an error, you need to run the previous command again after this one) sudo apt –fix-broken install

pip3 install yellowbrick

Because we are now inside a virtual environment. We do not need sudo.

Then you can start the server with:

jupyter-notebook

Which K works the best
    Accuracy for best K: 3

The best K accuracy
    49.67

Insert a confusion matrix for the best K
![alt text](image-1.png)
    
