"""                     Import libraries.                       """
from sklearn.model_selection import train_test_split

# Function for splitting the data.
def split_data(X, y, split_test=0.2, split_val=0.1, random_state=4):
    # Split to get the test data.
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=split_test, random_state=random_state, shuffle=True)

    # Split to get train and validation data.
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=split_val, random_state=random_state, shuffle=True)

    return X_train, X_val, X_test, y_train, y_val, y_test