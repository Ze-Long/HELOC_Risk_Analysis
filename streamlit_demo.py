import streamlit as st
import pickle
import numpy as np
from sklearn import metrics

# Load the pipeline and data
# pipe = pickle.load(open('pipe_logistic.sav', 'rb'))
X_test = pickle.load(open('X_test.sav', 'rb'))
y_test = pickle.load(open('y_test.sav', 'rb'))

dic = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}


# Function to test certain index of dataset
def test_demo(index):
    values = X_test[index]  # Input the value from dataset

    # Create four sliders in the sidebar
    a = st.sidebar.slider('Sepal Length', 0.0, 10.0, values[0], 0.1)
    b = st.sidebar.slider('Sepal Width', 0.0, 10.0, values[1], 0.1)
    c = st.sidebar.slider('Petal Length', 0.0, 10.0, values[2], 0.1)
    d = st.sidebar.slider('Petal Width', 0.0, 10.0, values[3], 0.1)

    # Print the prediction result
    alg = ['Decision Tree', 'Support Vector Machine', 'Logistic Regression']
    classifier = st.selectbox('Which algorithm?', alg)
    if classifier == 'Decision Tree':
        # different trained models should be saved in pipe with the help pickle
        '''
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        acc = dtc.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_dtc = dtc.predict(X_test)
        cm_dtc = confusion_matrix(y_test, pred_dtc)
        st.write('Confusion matrix: ', cm_dtc)
        '''
        st.text('Decision Tree Chosen')

    elif classifier == 'Support Vector Machine':
        '''
        svm = SVC()
        svm.fit(X_train, y_train)
        acc = svm.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_svm = svm.predict(X_test)
        cm = confusion_matrix(y_test, pred_svm)
        st.write('Confusion matrix: ', cm)
        '''
        st.text('SVM Chosen')

    else:
        pipe = pickle.load(open('pipe_logistic.sav', 'rb'))
        res = pipe.predict(np.array([a, b, c, d]).reshape(1, -1))[0]
        st.write('Prediction:  ', dic[res])
        pred = pipe.predict(X_test)
        score = pipe.score(X_test, y_test)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Accuracy: ', score)
        st.write('Confusion Matrix: ', cm)


# title
st.title('Iris')
# show data
if st.checkbox('Show dataframe'):
    st.write(X_test)
# st.write(X_train) # Show the dataset

number = st.text_input('Choose a row of information in the dataset (0~119):', 5)  # Input the index number

test_demo(int(number))  # Run the test function
