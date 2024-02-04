## readme for the application

extract the zip file 

the following commands can be run to install the 
libraries that are needed for running the application:

1. pip install U transformers
2. pip install datasets


It takes around 2030 minutes to run 
inferences on the flask application on my machine. 
To have a shorter inference time ~ 56 minutes, 
upload the notebook (ncri.ipynb) to gdrive
and run it using google colab with the 
the datasets and pretrained model uploaded 
to drive.
On flask the time difference is about 56 minutes.
I believe I could've made the program a bit faster 
if I had a bit more time.

The flask application can be run by the following commands:
(after cloning the repo and running the commands 
above for installing the libraries)

Installation guide for flask if needed https://flask.palletsprojects.com/en/2.3.x/installation/
1. python3 m venv .venv
2. source ./venv/bin/activate
3. Install the libraries
3. flask app app run

The time output for the executions are displayed 
on the console. 

There is a textbox on the web application
where you can add in any sentence and check what the model 
predicts. 

For any questions, do not hesitate to contact me. 
