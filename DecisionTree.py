import pandas as pd
df = pd.read_csv("students_adaptability_level_online_education.csv")
df.head()

inputs = df.drop(['Gender','IT Student','Location','Self Lms','Adaptivity Level','Age','Class Duration'], axis='columns')
Age_library = [x.split("-") for x in df["Age"]]
inputs["Age"] = [(sum(list(map(int, x))) / 2) for x in Age_library]
Class_duration_library = [x.split("-") for x in df["Class Duration"]]
inputs["Class Duration"] = [(sum(list(map(int, x))) / 2) for x in Class_duration_library]
target = df['Adaptivity Level']
inputs.head()

from sklearn.preprocessing import LabelEncoder

le_Age = LabelEncoder()
le_Education = LabelEncoder()
le_Institution = LabelEncoder()
le_Load_shedding = LabelEncoder()
le_Financial_Condition = LabelEncoder()
le_Internet_type = LabelEncoder()
le_Network_type = LabelEncoder()
le_Class_duration = LabelEncoder()
le_Device = LabelEncoder()
le_Adaptivity_Level = LabelEncoder()

inputs_n = pd.DataFrame()
target_n = pd.DataFrame()

inputs_n['Education_n'] = le_Education.fit_transform(inputs['Education Level'])
inputs_n['Institution_n'] = le_Institution.fit_transform(inputs['Institution Type'])
inputs_n['Load_shedding_n'] = le_Load_shedding.fit_transform(inputs['Institution Type'])
inputs_n['Financial_Condition_n'] = le_Financial_Condition.fit_transform(inputs['Financial Condition'])
inputs_n['Internet_type_n'] = le_Internet_type.fit_transform(inputs['Internet Type'])
inputs_n['Network_type_n'] = le_Network_type.fit_transform(inputs['Network Type'])
inputs_n['Device_n'] = le_Device.fit_transform(inputs['Device'])
target_n['Adaptivity_Level_n'] = le_Adaptivity_Level.fit_transform(target['Adaptivity Level'])

inputs.head()
target.head()
