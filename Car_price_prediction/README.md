# Car Resale Price Prediction

## Overview

This project aims to develop predictive models to estimate the resale price of a car based on various features that influence its market value. The models utilize linear regression and lasso regression techniques to provide accurate valuation.

## Features

- **Car Brand**: The manufacturer or brand of the car.
- **Year**: The manufacturing year of the car.
- **Sold Price**: The historical selling price of the car.
- **Present Price**: The current price of the car in the market.
- **KMS Driven**: The total kilometers driven by the car.
- **Fuel Type**: The type of fuel used by the car (e.g., petrol, diesel).
- **Seller Type**: The type of seller (e.g., individual, dealer).
- **Transmission Type**: The type of transmission (e.g., manual, automatic).
- **Owners**: The number of previous owners of the car.

## Dataset

The dataset used for model training can be downloaded from Kaggle:

- [Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?resource=download)

## Getting Started

### Prerequisites

- Python 3.8 or higher. I have used 3.11.5 .
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```

2. Navigate to the project directory:
    ```bash
    cd <project-directory>
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Project Structure

- `datasets/`: Contains all the datasets used in model building.
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis and model training.
- `models/`: Folder where the trained models are saved (`linear_regression_model.pkl` and `lasso_regression_model.pkl`).
- `main.py`: FastAPI application file with endpoints for predictions.
- `requirements.txt`: File listing the Python dependencies for the project.

### Running the Application

To start the FastAPI application, run the following command:
```bash
python main.py
```

The application will be available at `http://127.0.0.1:8000`.

### API Endpoints

1. **Linear Regression Prediction**
   - **Endpoint**: `/car_price_prediction/linear`
   - **Method**: POST
   - **Payload**: 
     ```json
     {
       "Year": 2010,
       "Present_Price": 20.450,
       "Kms_Driven": 50024,
       "Fuel_Type": 1,
       "Seller_Type": 0,
       "Transmission": 0,
       "Owner": 0
     }
     ```

2. **Lasso Regression Prediction**
   - **Endpoint**: `/car_price_prediction/lasso`
   - **Method**: POST
   - **Payload**: 
     ```json
     {
       "Year": 2010,
       "Present_Price": 20.450,
       "Kms_Driven": 50024,
       "Fuel_Type": 1,
       "Seller_Type": 0,
       "Transmission": 0,
       "Owner": 0
     }
     ```

### Testing

You can test the API endpoints using tools like [Postman](https://www.postman.com/) or `curl`.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [scikit-learn](https://scikit-learn.org/) for machine learning models
