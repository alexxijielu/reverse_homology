# Tutorial 3.1: Visualizing Impacts of Mutations using Local Web-app

## Overview

This tutorial will instruct users on how to visualize the mutations in 
the N-terminal IDR of p27, known as the kinase inhibitory domain (we 
refer to this IDR as p27-KID), that impact the features that most
distinguish this IDR from other IDRs in our model. In other words, this
tutorial instructs on how to produce what we call "mutational scanning 
maps" in our paper. Note that this tutorial focuses on p27-KID 
specifically, but the next tutorial (3.2 - Predicting Mutations for 
your own IDRs) guides you on how to produce the files necessary to 
visualize your own IDRs using the same process. 

## Step 0: Install Web-app Requirements Locally

To visualize mutations, we provide a companion web-app. This web-app
is intended to be run locally on a device with a monitor (i.e. if you
are using a remote GPU server and a local desktop/laptop, we recommend
you install it on your local device instead.) 

Install the requirements for the web-app at 
`/mutational_scanning_webapp/webapp_requirements.txt` on your 
local machine.

## Step 1: Boot the web-app

Next `/mutational_scanning_webapp/create_website.py` contains all of 
the code you need to run the web-app. 

Modify line 15 to point to the files for the IDR you want to visualize:
> idrpath = './p27kid/'

(This points to the files for p27-kid provided in the Github repository
by default, so no changes are needed for this tutorial. You will need 
to unzip the file in the Github repository.)

Next, simply run the code:
> python create_website.py

If the code successfully runs, you will see the following lines (or
something similar) in your console/stdout:
>Dash is running on http://127.0.0.1:8050/
> * Serving Flask app 'create_website' (lazy loading)
> * Environment: production
>   WARNING: This is a development server. Do not use it in a production deployment.
>   Use a production WSGI server instead.
> * Debug mode: on

Open the link in your web browser to see the web-app. Note that this
web-app is only visible to you locally, and cannot be accessed or shared
without other users. 

## Step 2: Explore your IDR in the web-app

The webapp consists of three components:
* **At the top is a summary of the mutational scanning maps** for the IDR.
The drop-down menu lets you select specific features for follow-up in
greater detail.
* Upon selecting a feature, the **letter map for the selected feature 
will be shown in the middle**, and the **heat map will be shown on the
bottom**. The letter map shows what positions are important to keep
(above the axis), or will increase the feature if mutated (below the axis).
The heat map shows the impacts of specific mutations as a grid.
See the paper for more specific definitions.

The animation below probably explains it better...

![Webapp](webapp_animation.gif)
