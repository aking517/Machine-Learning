Andrew King - kingand20@students.ecu.edu

Samese Heyward - heywards20@students.ecu.edu

ReadMe:

This is going to be a quickstart guide for HW4 of CSCI 4120 Machine Learning

The first step is installing a python IDE: I recommend Pycharm community. It is free, and it is what was used for this assignment. PyCharm community is available to download for a Windows machine at the following url: https://www.jetbrains.com/pycharm/download/?section=windows

Once pycharm community is set up, the .py file submitted to Canvas will be able to be opened.

In order to run the file correctly there are going to be certain libraries that must be installed for the code to run correctly, this can be done by clicking on the 'Python Packages' icon to the left of the terminal within PyCharm. The following packages will be installed:

-numpy

-pandas

-datetime

-scikit-learn

Lastly, you need to download two csv files on your computer to read data from. Both of these files can be found here:
https://github.com/ruiwu1990/CSCI_4120/tree/master/Linear_regression/data

Once the files are downloaded there are going to be two lines of code that need to be changed to match the filepath on your own device, they are lines 12 and 17 and what needs to be changed is the following:
'C:\\Users\\YOURNAMEHERE\\Downloads\\FremontBridge.csv',
'C:\\Users\\YOURNAMEHERE\\Downloads\\BicycleWeather.csv'
This is assuming the file will be downloaded in your downloads folder, make sure to \\ instead of just \ in the code.

Once everthing above is done simply run the code and view the results!

In my run of the code it was determined that the best model for this to run on was the ridge model.
The output is as follows:

The linear regression best cross-validation score is: 0.6780936339925822
The lasso best cross-validation score is:  0.6781371551415102
The best lasso alpha is:  0.6280291441834259
The ridge best cross-validation score is:  0.6781807933169146
The best ridge alpha is:  2.310129700083163
The best model is:  Ridge




