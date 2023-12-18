# 🚀Embedding a Machine Learning API using FastAPI 🚀


[![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![fastapi](https://img.shields.io/badge/FastAPI-009485?style=for-the-badge&logo=fastapi&logoColor=white)](https://img.shields.io/badge/FastAPI-3776AB?style=for-the-badge&logo=fastapi&logoColor=white)
![Issues](https://img.shields.io/github/issues/eaedk/streamlit-iris-app?style=for-the-badge&logo=appveyor)
![PR](https://img.shields.io/github/issues-pr/eaedk/streamlit-iris-app?style=for-the-badge&logo=appveyor)
[![Open Source Love png1](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)


## Project Description 
This project builds an API(Application Programming Interface) integrating a ML model using FastAPI to develop a powerful and scalable application for predictive analytics and real-time data processing. 

## Table of Contents
1. [Overview Of the Project](#overview)

  - [Description of dataset](#dataset)

2. [Application / Deployed Links](#application)

3. [Technology Stack](#technology)

4. [Deliverables](#deliverables)

5. [Execution](#execution)

6. [API Endpoints](#api-endpoints)

7. [App Usage](#usage)

8. [Resources](#resources)

9. [Contact Information](#ontact)


## 1. Overview Of the Project <a name="overview"></a>

- The sepsis prediction project ia about building a machine learning model designed to accurately predict sepsis illness in intensive care unit (ICU) patients. The model has undergone rigorous training and evaluation to ensure its effectiveness in identifying patients at risk of developing sepsis.

- The project provides a comprehensive solution, including a well-documented FastAPI hosted on a platform like the Hugging Face Model Hub. This API allows seamless integration of the sepsis prediction model into existing healthcare systems, providing healthcare professionals with valuable insights to improve patient care.

- To simplify deployment and usage, the project includes a Dockerfile that streamlines the setup process and ensures the necessary dependencies are installed. This enables easy deployment of the sepsis prediction model in various environments, both local and cloud-based.


### Description of dataset <a name="dataset"></a>

<table>
  <tr>
    <th>Column Name</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>ID</td>
    <td>Unique number to represent patient ID</td>
  </tr>
  <tr>
    <td>PRG</td>
    <td>Plasma glucose</td>
  </tr>
  <tr>
    <td>PL</td>
    <td>Blood Work Result-1 (mu U/ml)</td>
  </tr>
  <tr>
    <td>PR</td>
    <td>Blood Pressure (mm Hg)</td>
  </tr>
  <tr>
    <td>SK</td>
    <td>Blood Work Result-2 (mm)</td>
  </tr>
  <tr>
    <td>TS</td>
    <td>Blood Work Result-3 (mu U/ml)</td>
  </tr>
  <tr>
    <td>M11</td>
    <td>Body mass index (weight in kg/(height in m)^2)</td>
  </tr>
  <tr>
    <td>BD2</td>
    <td>Blood Work Result-4 (mu U/ml)</td>
  </tr>
  <tr>
    <td>Age</td>
    <td>Patients age (years)</td>
  </tr>
  <tr>
    <td>Insurance</td>
    <td>If a patient holds a valid insurance card</td>
  </tr>
  <tr>
    <td>Sepssis</td>
    <td>Positive: if a patient in ICU will develop sepsis, and Negative: otherwise</td>
  </tr>
</table>

## 2. Application / Deployed Links <a name="application"></a>
<table>
  <tr>
    <th>API</th>
    <th>Deployed links</th>
  </tr>
  <tr>
    <td>FastApi</td>
    <td><a href="https://bright1-sepsis-prediction-api.hf.space/docs">Sepsis Prediction API-huggingface</a></td>
  </tr>
  <tr>
  
  </tr>

</table>


## 3. Technology Stack <a name="technology"></a>
 
<table>
  <tr>
    <th>Technology</th>
    <th>Version</th>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.11.5</td>
  </tr>
  <tr>
    <td>FastAPI</td>
    <td>0.105.0</td>
  </tr>
  <tr>
    <td>Uvicorn</td>
    <td>0.24.0</td>
  </tr>
    <tr>
    <td>Scikit-learn</td>
    <td>1.3.2</td>
  </tr>
  </tr>
    <tr>
    <td>Pandas</td>
    <td>2.1.4</td>
  </tr>
  </tr>
    <tr>
    <td>Jinja2</td>
    <td>3.1.2</td>
  </tr>
  
</table>

## 4. Deliverables <a name="deliverables"></a>
1. A jupyter notebook for training a classification model
2. A classification Model
3. An API App built with FastApi
4. A main.py that make calls to the build and hosted API
5. A Dockerfile for easy deployment 

## 8. App Usage <a name="usage"></a>
To test the various endpoints of the API using the provided documentation, follow these steps:

1. Start by accessing the API documentation, which provides detailed information about the available endpoints and their functionalities.

2. Locate the section that describes the input fields and parameters required for each endpoint. It will specify the expected data format, such as JSON or form data, and the necessary input fields.

4. Enter the required input data into the corresponding input fields or parameters as specified in the documentation.

5. Send the request by clicking the "Execute" button or using the appropriate method in your chosen tool. The API will process the request and generate the output based on the provided inputs.

6. Retrieve the response from the API, which will contain the generated output. This output may include predictions, probability scores, or any other relevant information related to sepsis prediction.

7. Repeat the process by varying the input data to explore the capabilities of the API. Make sure to follow the documentation's guidelines for each endpoint to ensure accurate results.

## 9. Resources <a name="resources"></a>
Here are some ressources you would read to have a good understanding of FastAPI :
- [Tutorial - User Guide](https://fastapi.tiangolo.com/tutorial/)
- [Video - Building a Machine Learning API in 15 Minutes ](https://youtu.be/C82lT9cWQiA)
- [FastAPI for Machine Learning: Live coding an ML web application](https://www.youtube.com/watch?v=_BZGtifh_gw)
- [Video - Deploy ML models with FastAPI, Docker, and Heroku ](https://www.youtube.com/watch?v=h5wLuVDr0oc)
- [FastAPI Tutorial Series](https://www.youtube.com/watch?v=tKL6wEqbyNs&list=PLShTCj6cbon9gK9AbDSxZbas1F6b6C_Mx)
- [Http status codes](https://www.linkedin.com/feed/update/urn:li:activity:7017027658400063488?utm_source=share&utm_medium=member_desktop)


## 10. Contact Information <a name="contact"></a>

<table>
  <tr>
    <th>Name</th>
    <th>LinkedIn</th>
    <th>GitHub</th>
    
  </tr>
  <tr>
    <td>Benjamin Kaitany</td>
    <td><a href="https://www.linkedin.com/in/benjamin-kaitany-5017681b2/">@kaitany</a></td>
    <td><a href="https://github.com/kaitany">@kaitany</a></td>
  </tr>
</table>