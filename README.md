
# CreditDefaultRiskwithDeployment

Goal: Predict the credit repayment capability for each customer.

Data: We used the 'Home Credit Default Risk' competition dataset in this project.

Approach: Turned this problem into a classification problem with hundreds of features where 1 represents 'unable to pay' and 0 represents 'able to pay the loan'. Produced new features using feature extraction and feature interaction techniques. Since the data was highly imbalanced just like the other classification problems, we used 'undersampling' to balance the classes which improved the roc-AUC score a lot. We used LightGBM due to its high speed and high ROC-AUC scores. Finally, we productized our code into a live environment by deploying it into the Streamlit&Heroku platform where customers (in this manner Banks) can enter the most important 9 features and fetch the results. The remaining 445 features have default values and do not seem in the web app but they can be seen in the app.py file where one can find it in the GitHub link provided below. 445 feature can be entered via visiting the app.py file on demand of the customer's request.

Results: %70 ROC-AUC scores on private set. The project deployed into live environment with user-friendly interface.

For the web-app of the application, please click on the link below :

https://home-credit-default-risk-app.herokuapp.com/


