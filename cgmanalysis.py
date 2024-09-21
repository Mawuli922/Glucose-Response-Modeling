import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# read file into pandas dataframe
df = pd.read_json("raw_data.json")

# inspect contents of dataframe
print(df.head())
print(df.info())

# evaluate the pearson's correlation statistic
corr_mat = df[["current_nA", "substrate_reference_mM", "bias_potential_mV", "full_cell_potential_mV"]].corr()

# visualize the correlation matrix using heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_mat, annot=True)
plt.title("Pearson's Correlation Matrix of Variables in Dataset")
plt.show()

# plot the line graph of the current response variable
df["current_nA"].plot()
plt.xlabel("Time (s)")
plt.ylabel("Current (nA)")
plt.title("Trend of Current Response Values of Time")
plt.show()
# plot a scatter plot of the current response with full cell potential column
plt.scatter(df["current_nA"], df["substrate_reference_mM"])
plt.title("response current vs glucose substrate levels")
plt.xlabel("Current (nA)")
plt.ylabel("Glucose substrate (mm)")
plt.show()

# Plot a distribution of the frequency of glucose concentration levels in the dataset
plt.figure(figsize=(10, 6))
df["substrate_reference_mM"].value_counts().plot(kind="bar")
plt.title("Count Frequency of Glucose Levels in Experimental Dataset")
plt.xticks(rotation=15)
plt.xlabel("substrate reference (mm)")
plt.ylabel("count")
plt.show()

class GlucoseCurrentRegressionModel:
    def __init__(self, dataframe, time_col, glucose_col, current_col):
        """
        Initializes the class with the dataframe , time, glucose and current columns
        :param dataframe: A pandas object
        :param time_col: time in seconds
        :param glucose_col: positive float
        :param current_col: positive float
        """
        self.df = dataframe
        self.time_col = time_col
        self.glucose_col = glucose_col
        self.current_col = current_col
        self.X = None
        self.y = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.stable_df = None

    def derivative_computation(self):
        """
        Calculates the derivatives of glucose and current with respect to time to identify stable
        portions where the changes are minimal
        :return: dataframe with added derivatives columns for both glucose and current
        """
        self.df['glucose_derivative'] = np.gradient(self.df[self.glucose_col], self.df[self.time_col])
        self.df['current_derivative'] = np.gradient(self.df[self.current_col], self.df[self.time_col])


    def identify_stable_regions(self, glucose_threshold= 0.01, current_threshold= 0.01, settling_time= 5):
        """
        Identifies stable portions in the data by filtering out sections where the derivative of glucose
        and current is higher than the threshold and excludes the settling time
        :param glucose_threshold: Maximum allowed rate of change of glucose to be considered stable
        :param current_threshold: Maximum allowed rate of change of current to be considered stable
        :param settling_time: Time to ignore after initial glucose addition
        :return: Filtered dataframe with stable glucose and current conditions
        """

        # Ignore the first n seconds after glucose is added initially
        self.df = self.df[self.df[self.time_col] > settling_time]
        #  Filter for rows with both stable glucose and current gradient values
        stable_df = self.df[(self.df['glucose_derivative'].abs()< glucose_threshold)&
                            (self.df['current_derivative'].abs()< current_threshold)]

        self.stable_df = stable_df

    def extract_current_and_glucose_pairs(self):
        """
        Extracts the current-glucose pairs from stable regions of the dataframe.
        Assumes glucose stability hence each glucose value corresponds to a stable current value
        :return: a dataframe grouped by glucose value and their average current values
        """

        # Group by glucose concentration and select the mean of the stable current values
        current_glucose_pairs = self.stable_df.groupby(self.glucose_col)[self.current_col].mean().reset_index()
        return current_glucose_pairs

    def preprocess_data(self, test_size=0.2, scale=True, glucose_threshold=0.01, current_threshold=0.01, settling_time= 5):
        """
        preprocess the data by extracting stable glucose-current pairs, and splitting into train and test
        sets with the option to scale features using StandardScaler
        :param test_size: percentage of data to allocate to testing, default value of 0.2
        :param scale: Boolean value to implement scaling, default value of True
        :return: Training and Testing sets for both Glucose and Current variables
        """
        # Perform glucose-current extraction as part of preprocessing
        self.derivative_computation()
        self.identify_stable_regions(glucose_threshold, current_threshold, settling_time)
        stable_pairs = self.extract_current_and_glucose_pairs()

        # Define X and y

        self.X = stable_pairs[[self.glucose_col]]
        self.y = stable_pairs[self.current_col]

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def train_model(self):
        """
        Trains a linear regression model on the preprocessed data
        :return: Coefficients of regression
        """
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    def evaluate_model(self):
        """
        Evaluates performance of the model using Mean Squared Error, R-Squared and Mean Absolute Error
        :return: Mean Squared Error, R-Squared, Mean Absolute Error
        """
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R-Squared score: {r2}")
        print(f"Mean Absolute Error: {mae}")

    def plot_residuals(self):
        """
        Plots the residuals (differences between actual values and model estimates)
        :return: A seaborn residual plot
        """

        residuals = self.y_test - self.y_pred
        length = [k for k in range(len(residuals))]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=length, y=residuals, color="blue")
        plt.title("Residuals plot")
        plt.xlabel("Index position")
        plt.ylabel("Residuals")
        plt.show()

    def plot_best_fit_line(self):
        """
        Plots the line of best fit using the model's coefficients, overlaid on scatter plot of the actual data
        :return: A matplotlib combination graph of line and scatter graph
        """

        # Extract the model's coefficients and intercepts

        slope = self.model.coef_[0]
        intercept = self.model.intercept_

        # Calculate the predicted y values
        line_of_best_fit = slope*self.X_train + intercept

        # Plot the actual data scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_train, self.y_train, color='blue', label='Actual')

        # Plot the line of best fit
        plt.plot(self.X_train, line_of_best_fit, color='red', label='Best fit')
        plt.title("Scatter Plot with Best Fit Line")
        plt.xlabel("X (Standardized Glucose Concentration)")
        plt.ylabel("y (Current)")
        plt.legend()
        plt.show()
        print(f"Model slope:{slope}")
        print(f"Model intercept: {intercept}")

    def run(self):
        """
        Complete workflow from preprocessing to results evaluation
        :return:
        """
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()
        self.plot_residuals()
        self.plot_best_fit_line()

model = GlucoseCurrentRegressionModel(
    dataframe=df,
    time_col='time_s',
    glucose_col='substrate_reference_mM',
    current_col='current_nA'

)

model.run()






