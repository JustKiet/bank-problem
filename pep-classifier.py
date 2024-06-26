import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st

#Load data
data_path = 'bank-data.csv'
df = pd.read_csv(data_path)

#Xử lý dữ liệu
df_processed = df.copy()
region_map = {'INNER_CITY': 0, 'TOWN': 1, 'RURAL': 2, 'SUBURBAN': 3}
sex_map = {'FEMALE': 0, 'MALE': 1}
df_processed['region'] = df_processed['region'].replace(region_map)
df_processed.replace({"YES": 1, "NO": 0}, inplace=True)
df_processed['sex'] = df_processed['sex'].replace(sex_map)
for idx, val in enumerate(df_processed['age']):
    if val <= 30:
        df_processed.at[idx, 'age'] = 0 #Dưới 30 tuổi
    elif 30 < val <= 50:
        df_processed.at[idx, 'age'] = 1 #Khoảng 30 - 50 tuổi
    else:
        df_processed.at[idx, 'age'] = 2 #Trên 50 tuổi

for idx, val in enumerate(df_processed['income']):
    if val <= 20000:
        df_processed.at[idx, 'income'] = 0 #Dưới 20000 (Thu nhập thấp)
    elif 20000 < val <= 40000:
        df_processed.at[idx, 'income'] = 1 #Khoảng 20000 - 40000 (Thu nhập trung)
    else:
        df_processed.at[idx, 'income'] = 2 #Trên 40000 (Thu nhập cao)
df_processed['income'] = df_processed['income'].astype(int)
if 'id' in df_processed:
    df_processed.drop('id', axis=1, inplace=True) 

#Chia bộ dữ liệu thành thuộc tính (X) và kết quả (y)
X = df_processed.drop('pep', axis=1)  # Thuộc tính
y = df_processed['pep']  # Kết quả

# Split data thành training data và test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#Random Forest
rf = RandomForestClassifier(criterion='entropy')

# Train model
rf.fit(X_train, y_train)

# Dự đoán theo X_test
y_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_rf)
print(classification_report(y_test, y_rf))
print(f"Accuracy: {accuracy_rf * 100:.2f}%")


#ID3
id3 = DecisionTreeClassifier(criterion='entropy')

id3.fit(X_train, y_train)

y_id3 = id3.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_id3)
print(confusion_matrix)
accuracy_id3 = accuracy_score(y_test, y_id3)
print(classification_report(y_test, y_id3))
print(f"Accuracy: {accuracy_id3 * 100:.2f}%")


#CART
cart = DecisionTreeClassifier()

cart.fit(X_train, y_train)

y_cart = cart.predict(X_test)

accuracy_cart = accuracy_score(y_test, y_cart)
print(classification_report(y_test, y_cart))
print(f"Accuracy: {accuracy_cart * 100:.2f}%")


def pickModel(rf_acc, cart_acc, id3_acc, inputVar):
    bestModel = max(rf_acc, cart_acc, id3_acc)
    if bestModel == rf_acc:
        y_pred = rf.predict(inputVar)
        return y_pred
    elif bestModel == cart_acc:
        y_pred = cart.predict(inputVar)
        return y_pred
    elif bestModel == id3_acc:
        y_pred = id3.predict(inputVar)
        return y_pred


#Streamlit GUI
form = st.form(key='index')
age = int(form.number_input('age', step=1, format="%i"))
sex = int(form.number_input("sex (FEMALE: 0, MALE: 1)", step=1, format="%i"))
region = int(form.number_input("region ('INNER_CITY': 0, 'TOWN': 1, 'RURAL': 2, 'SUBURBAN': 3)", step=1, format="%i"))
income = int(form.number_input('income', step=1, format="%i"))
married = int(form.number_input('married (YES: 1, NO: 0)', step=1, format="%i"))
children = int(form.number_input('children', step=1, format="%i"))
car = int(form.number_input('car (YES: 1, NO: 0)', step=1, format="%i"))
save_act = int(form.number_input('save_act (YES: 1, NO: 0)', step=1, format="%i"))
current_act = int(form.number_input('current_act (YES: 1, NO: 0)', step=1, format="%i"))
mortgage = int(form.number_input('mortgage (YES: 1, NO: 0)', step=1, format="%i"))
submit = form.form_submit_button('Submit')

if submit:
    X_user = pd.DataFrame(columns=X.columns)

    if age <= 30:
        age = 0
    elif 30 < age <= 50:
        age = 1
    else:
        age = 2

    if income <= 20000:
        income = 0
    elif 20000 < income <= 40000:
        income = 1
    else:
        income = 2

    X_user.loc[len(X_user)] = [age, sex, region, income, married, children, car, save_act, current_act, mortgage]

    result = pickModel(accuracy_rf, accuracy_cart, accuracy_id3, X_user)

    st.write("PEP (YES: 1, NO: 0)", result)