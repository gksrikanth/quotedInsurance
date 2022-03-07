import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime
from numpy import cov
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.int` is a deprecated alias')

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from PIL import Image
image = Image.open('C:/Users/srikanthg/GKS/P-NB/Streamlit/Life-Insurance-Concept-Family.jpg')


st.title('Prediction of Customers buying Quoted Insurance Plan!')
st.caption('A simple app that shows where a Customer buys the quoted insurance plan. You can choose the options'
             + ' to verify for a particular Customer. There is also a file upload option where you can pass your quoted insurance ' + 
             'file and by passing the Customer Quote ID, you will get the predicted result.')
    
st.image(image, caption='Quoted Insurance Prediction - Srikanth G K')

DATA_URL = ('C:/Users/srikanthg/GKS/Projects/train_data/train.csv')

@st.cache
def load_data():
    insurance = pd.read_csv(DATA_URL)
    return insurance
	
@st.cache
def data_preprocessing_1(insurance):

	insurance["Field10"] = insurance["Field10"].str.replace(",","").astype(int)

	str_date = insurance['Original_Quote_Date']
	year_lst=[]
	month_lst=[]
	weekend_lst=[]
	date_lst=[]

	for i in str_date:
	  year_lst.append(int(i[0:4]))
	  month_lst.append(int(i[5:7]))
	  date_lst.append(int(i[8:]))
	  d = datetime(int(i[0:4]), int(i[5:7]), int(i[8:]))
	  if d.weekday() > 4:
	  	weekend_lst.append(1)
	  else:
	  	weekend_lst.append(0)

	insurance['Year'] = year_lst
	insurance['Month'] = month_lst
	insurance['Is_Weekend'] = weekend_lst

	insurance.drop(['Original_Quote_Date'], axis = 1, inplace = True)
	return insurance

@st.cache
def data_preprocessing_2(insurance):

	# Get the Categorical Features and its count.
	cols = insurance.columns
	num_cols = insurance._get_numeric_data().columns

	categorical_features = sorted(list(set(cols) - set(num_cols)))
	df_cat_features_lt_str = insurance[categorical_features].nunique().to_frame().to_string()

	#Give names to the output data
	df_cat_features_lt = pd.read_csv(StringIO(df_cat_features_lt_str), sep='\s+', names = ['Field', 'Unique_Count', 'Null_Count'])
	df_cat_features_lt = df_cat_features_lt[df_cat_features_lt.Field != '0']

	df_cat_features_lt['Null_Count'] = insurance[df_cat_features_lt.Field].isnull().sum().values

	# Replace values having only 'Y' or 'N' to 1 and 0 respectively.
	features_only_two = [] 
	features_only_two = df_cat_features_lt[df_cat_features_lt['Unique_Count'] == 2]['Field']

	# Converting categorical features having only 2 values into numerical 1's and 0's.
	for j in df_cat_features_lt.loc[(df_cat_features_lt['Unique_Count'] == 2)]['Field']:
	  insurance[j].replace({'Y' : 1, 'N' : 0}, inplace = True)

	for i in features_only_two:
	  index_name = df_cat_features_lt[df_cat_features_lt['Field'] == i].index
	  df_cat_features_lt.drop(index_name, inplace = True)
	

	# Remove the featues that have more than 90% of "-1" as value.
	total_count_in_dataset = insurance['QuoteNumber'].count()

	insurance_with_minus_1 = list(insurance.columns[insurance.isin([-1]).any()])
	#insurance_with_minus_1

	insurance_with_minus_1_str = insurance[insurance_with_minus_1][insurance[insurance_with_minus_1].isin([-1])].count().to_frame().to_string()

	#Give names to the output data
	insurance_with_minus_1_percent = pd.read_csv(StringIO(insurance_with_minus_1_str), sep='\s+', names = ['Field', 'Minus_1_Count', 'Minus_1_Percent'])
	insurance_with_minus_1_percent = insurance_with_minus_1_percent[insurance_with_minus_1_percent.Field != '0']

	insurance_with_minus_1_percent['Minus_1_Percent'] = (insurance_with_minus_1_percent.Minus_1_Count/total_count_in_dataset)*100

	insurance.drop(columns=insurance_with_minus_1_percent[insurance_with_minus_1_percent['Minus_1_Percent']>90]['Field'], inplace = True)

	# Replacing all -1's to NaN.
	for col in insurance[list(insurance_with_minus_1_percent[insurance_with_minus_1_percent['Minus_1_Percent']<90]['Field'])]:
	  insurance[col] = insurance[col].replace(-1, np.nan)
	  

	# Drop the Standalone features.
	df_str = insurance.nunique().to_frame().to_string()

	#Give names to the output data
	df_new = pd.read_csv(StringIO(df_str), sep='\s+', names = ['Field', 'Unique_Count'])

	#Will ignore the unwanted field with field value '0' from the new DataFrame
	df_drop_features = df_new.loc[1:]

	#List the unique fields having a single value in the entire train dataset
	df_drop_features['Unique_Count'].astype(int)

	#Finally drop the fields from the train dataset
	fieldsToDrop = df_drop_features[df_drop_features['Unique_Count'] == 1]['Field'].values;

	insurance.drop(columns=fieldsToDrop, axis = 1, inplace = True)
	return insurance
	
@st.cache	
def data_preprocessing_3(insurance):

	# Removing high correlated features from the train dataset.
	field_cols = []
	for i in insurance.columns:
	  if i.startswith('Field'):
	  	field_cols.append(i)

	coverage_field_cols = []
	for i in insurance.columns:
	  if i.startswith('CoverageField'):
	  	coverage_field_cols.append(i)

	sales_field_cols = []
	for i in insurance.columns:
	  if i.startswith('SalesField'):
	  	sales_field_cols.append(i)

	personal_field_cols = []
	for i in insurance.columns:
	  if i.startswith('PersonalField'):
	  	personal_field_cols.append(i)

	property_field_cols = []
	for i in insurance.columns:
	  if i.startswith('PropertyField'):
	  	property_field_cols.append(i)

	graphic_field_cols = []
	for i in insurance.columns:
	  if i.startswith('GeographicField'):
	  	graphic_field_cols.append(i)

	all_features_list = ['field_cols', 'coverage_field_cols', 'sales_field_cols', 'personal_field_cols', 'property_field_cols', 'graphic_field_cols']

	total_high_corr_features=[]

	corr_matrix = insurance[field_cols].corr().abs()
	corr_limit = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
	remove_high_corr_features = [col for col in corr_limit.columns if any(corr_limit[col] > 0.9)]
	if (len(remove_high_corr_features) != 0):
	  total_high_corr_features.extend(remove_high_corr_features)

	corr_matrix = insurance[coverage_field_cols].corr().abs()
	corr_limit = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
	remove_high_corr_features = [col for col in corr_limit.columns if any(corr_limit[col] > 0.9)]
	if (len(remove_high_corr_features) != 0):
	  total_high_corr_features.extend(remove_high_corr_features)

	corr_matrix = insurance[sales_field_cols].corr().abs()
	corr_limit = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
	remove_high_corr_features = [col for col in corr_limit.columns if any(corr_limit[col] > 0.9)]
	if (len(remove_high_corr_features) != 0):
	  total_high_corr_features.extend(remove_high_corr_features)

	corr_matrix = insurance[personal_field_cols].corr().abs()
	corr_limit = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
	remove_high_corr_features = [col for col in corr_limit.columns if any(corr_limit[col] > 0.9)]
	if (len(remove_high_corr_features) != 0):
	  total_high_corr_features.extend(remove_high_corr_features)

	corr_matrix = insurance[property_field_cols].corr().abs()
	corr_limit = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
	remove_high_corr_features = [col for col in corr_limit.columns if any(corr_limit[col] > 0.9)]
	if (len(remove_high_corr_features) != 0):
	  total_high_corr_features.extend(remove_high_corr_features)

	corr_matrix = insurance[graphic_field_cols].corr().abs()
	corr_limit = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
	remove_high_corr_features = [col for col in corr_limit.columns if any(corr_limit[col] > 0.9)]
	if (len(remove_high_corr_features) != 0):
	  total_high_corr_features.extend(remove_high_corr_features)

	insurance.drop(columns=total_high_corr_features, inplace = True)

	# Get the featurs which has empty/NaN values and their count and their percentage.
	total_count = insurance.shape[0]

	dfsr = insurance.isnull().sum()
	df_nan = dfsr[dfsr.values != 0].to_frame().to_string()

	#Give names to the output data
	df_nan_new = pd.read_csv(StringIO(df_nan), sep='\s+', names = ['Field', 'Nan_Count', 'Nan_Percent'])

	df_nan_new = df_nan_new[df_nan_new.Field != '0']

	# Get the NaN percentage from the total count.
	df_null = insurance[insurance.columns[insurance.eq(-1).any()]]
	total_null_cols = df_null.columns
	nan_new_count = df_nan_new['Field'].count()

	for i in total_null_cols:
	  nan_new_count += 1
	  df_nan_new.loc[nan_new_count] = [i] + [int(insurance.QuoteNumber[insurance[i] == -1].count())] + [0]

	for ind, j in df_nan_new.iterrows():
	  df_nan_new['Nan_Percent'] = round((df_nan_new['Nan_Count'] / total_count)*100,2)
	

	#Dropping the features that has more than 90% of NULL/NaN values.

	for k in df_nan_new[df_nan_new['Nan_Percent'] >= 90].Field:
	  insurance.drop(columns=k, axis = 1, inplace = True)
	
	return insurance
	  
@st.cache  
def data_preprocessing_4(insurance):
	medianFields = ['SalesField2A',
	'PersonalField7',
	'PersonalField10A',
	'PersonalField84',
	'PropertyField3',
	'PropertyField29',
	'PropertyField32',
	'PropertyField34',
	'PropertyField36',
	'PropertyField38',
	'GeographicField1A',
	'GeographicField5B',
	'GeographicField17A',
	'GeographicField22B',
	'GeographicField37A',
	'GeographicField47A',
	'GeographicField62B',
	'GeographicField31A',
	'GeographicField32A',
	'GeographicField33A',
	'GeographicField35A',
	'GeographicField38A',
	'GeographicField39A',
	'GeographicField49A',
	'GeographicField50A',
	'GeographicField53A',
	'GeographicField55A']

	meanFields = ['CoverageField1A',
	'CoverageField11A',
	'SalesField2B',
	'PersonalField4A',
	'PersonalField10B',
	'PropertyField1A',
	'PropertyField16A',
	'PropertyField21A',
	'PropertyField24A',
	'PropertyField26A',
	'PropertyField39A',
	'GeographicField2A',
	'GeographicField6A',
	'GeographicField6B',
	'GeographicField7A',
	'GeographicField18B',
	'GeographicField19A',
	'GeographicField20A',
	'GeographicField20B',
	'GeographicField21B',
	'GeographicField24A',
	'GeographicField25A',
	'GeographicField26A',
	'GeographicField28A',
	'GeographicField29A',
	'GeographicField30A',
	'GeographicField30B',
	'GeographicField34A',
	'GeographicField36A',
	'GeographicField37B',
	'GeographicField40A',
	'GeographicField41A',
	'GeographicField42A',
	'GeographicField43A',
	'GeographicField44A',
	'GeographicField45A',
	'GeographicField46A',
	'GeographicField47B',
	'GeographicField51A',
	'GeographicField54A',
	'GeographicField56B',
	'GeographicField57A',
	'GeographicField58A',
	'GeographicField59A',
	'GeographicField60B',
	'GeographicField61B']

	for i in medianFields:
	  insurance[i] = insurance[i].fillna(insurance[i].median())

	for i in meanFields:
	  insurance[i] = insurance[i].fillna(insurance[i].mean())
	
	
	# Grouping the features
	field_cols = []
	for i in insurance.columns:
	  if i.startswith('Field'):
	  	field_cols.append(i)

	coverage_field_cols = []
	for i in insurance.columns:
	  if i.startswith('CoverageField'):
	  	coverage_field_cols.append(i)

	sales_field_cols = []
	for i in insurance.columns:
	  if i.startswith('SalesField'):
	  	sales_field_cols.append(i)

	personal_field_cols = []
	for i in insurance.columns:
	  if i.startswith('PersonalField'):
	  	personal_field_cols.append(i)

	property_field_cols = []
	for i in insurance.columns:
	  if i.startswith('PropertyField'):
	  	property_field_cols.append(i)

	geographic_field_cols = []
	for i in insurance.columns:
	  if i.startswith('GeographicField'):
	  	geographic_field_cols.append(i)

	all_features_list = ['field_cols', 'coverage_field_cols', 'sales_field_cols', 'personal_field_cols', 'property_field_cols', 'geographic_field_cols']

	
	#Get the top correlated feature having correlated value > 0.70 and lessthan 1.000000 for each grouped features.
	#Field Feature
	y = insurance['QuoteConversion_Flag']
	X_field = insurance[field_cols]
	X_field['QuoteConversion_Flag'] = insurance['QuoteConversion_Flag']

	corr_mat = X_field.corr()
	top_corr_features_field = corr_mat.index

	final_field_feature = []
	final_field_feature = X_field[top_corr_features_field].corr()[(X_field[top_corr_features_field].corr().values > 0.70) & (X_field[top_corr_features_field].corr().values < 1.000000)].index.unique()

	final_features = []
	final_features.extend(final_field_feature.values)

	#Coverage Feature
	X_cov_field = insurance[coverage_field_cols]
	X_cov_field['QuoteConversion_Flag'] = insurance['QuoteConversion_Flag']

	corr_mat = X_cov_field.corr()
	top_corr_features_cov_field = corr_mat.index

	final_cov_field_feature = []
	final_cov_field_feature = X_cov_field[top_corr_features_cov_field].corr()[(X_cov_field[top_corr_features_cov_field].corr().values > 0.70) & (X_cov_field[top_corr_features_cov_field].corr().values < 1.000000)].index.unique()

	final_features.extend(final_cov_field_feature.values)

	#Sales Feature
	X_sales_field = insurance[sales_field_cols]
	X_sales_field['QuoteConversion_Flag'] = insurance['QuoteConversion_Flag']

	corr_mat = X_sales_field.corr()
	top_corr_features_sales_field = corr_mat.index

	final_sales_field_feature = []
	final_sales_field_feature = X_sales_field[top_corr_features_sales_field].corr()[(X_sales_field[top_corr_features_sales_field].corr().values > 0.70) & (X_sales_field[top_corr_features_sales_field].corr().values < 1.000000)].index.unique()

	final_features.extend(final_sales_field_feature.values)
	
	#Personal Feature
	X_personal_field = insurance[personal_field_cols]
	X_personal_field['QuoteConversion_Flag'] = insurance['QuoteConversion_Flag']

	corr_mat = X_personal_field.corr()
	top_corr_features_personal_field = corr_mat.index

	final_personal_field_feature = []
	final_personal_field_feature = X_personal_field[top_corr_features_personal_field].corr()[(X_personal_field[top_corr_features_personal_field].corr().values > 0.70) & (X_personal_field[top_corr_features_personal_field].corr().values < 1.000000)].index.unique()

	final_features.extend(final_personal_field_feature.values)
	
	#Property Feature
	X_property_field = insurance[property_field_cols]
	X_property_field['QuoteConversion_Flag'] = insurance['QuoteConversion_Flag']

	corr_mat = X_property_field.corr()
	top_corr_features_property_field = corr_mat.index

	final_property_field_feature = []
	final_property_field_feature = X_property_field[top_corr_features_property_field].corr()[(X_property_field[top_corr_features_property_field].corr().values > 0.70) & (X_property_field[top_corr_features_property_field].corr().values < 1.000000)].index.unique()

	final_features.extend(final_property_field_feature.values)
	
	# Geographic Feature
	X_geographic_field = insurance[geographic_field_cols]
	X_geographic_field['QuoteConversion_Flag'] = insurance['QuoteConversion_Flag']

	corr_mat = X_geographic_field.corr()
	top_corr_features_geographic_field = corr_mat.index

	final_geographic_field_feature = []
	final_geographic_field_feature = X_geographic_field[top_corr_features_geographic_field].corr()[(X_geographic_field[top_corr_features_geographic_field].corr().values > 0.70) & (X_geographic_field[top_corr_features_geographic_field].corr().values < 1.000000)].index.unique()

	final_features.extend(final_geographic_field_feature.values)
	
	# Drop the features with high correlation of more than 70.
	insurance.drop(columns = final_features, axis = 1, inplace = True)
	
	return insurance
	
@st.cache	
def data_preprocessing_5(insurance):
	# Get the categorical features.
	cols = insurance.columns
	num_cols = insurance._get_numeric_data().columns

	categorical_features = sorted(list(set(cols) - set(num_cols)))
	df_cat_features_lt_str = insurance[categorical_features].nunique().to_frame().to_string()

	#Give names to the output data
	df_cat_features_lt = pd.read_csv(StringIO(df_cat_features_lt_str), sep='\s+', names = ['Field', 'Unique_Count'])
	df_cat_features_lt = df_cat_features_lt[df_cat_features_lt.Field != '0']
	
	# Using One-Hot Encoding to convert String to Numeric Features.
	for i in df_cat_features_lt['Field']:
	  enc = OneHotEncoder(handle_unknown = 'ignore')
	  enc_df_state = pd.DataFrame(enc.fit_transform(insurance[[i]]).toarray())
	  enc_df_state.columns = enc.get_feature_names([i])
	  insurance = insurance.join(enc_df_state)
	  insurance.drop([i], axis = 1, inplace = True)
	  
	return insurance

@st.cache	  
def split_train_test(insurance):
	X_train_df = insurance.loc[:, insurance.columns != 'QuoteConversion_Flag']
	Y_train_df = insurance['QuoteConversion_Flag']

	X_train, X_test, y_train, y_test = train_test_split(
		X_train_df, Y_train_df, test_size=0.3
	)
	return X_train

@st.cache	
def get_top_features_pca(X_train):
	scaler = StandardScaler()
	X_trained_scaled = scaler.fit_transform(X_train)

	pca = PCA()
	x_pca = pca.fit_transform(X_trained_scaled)

	tot_explained_variance = pca.explained_variance_ratio_.cumsum()
	
	# Get the feature names having 95% of explained variance
	model = PCA(n_components=290).fit(X_train)
	X_pc = model.transform(X_train)

	# number of components
	n_pcs= model.components_.shape[0]

	# get the index of the most important feature on each component
	most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

	initial_feature_names = X_train.columns
	# get the important feature names
	most_important_features = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

	dic = {'PC{}'.format(i): most_important_features[i] for i in range(n_pcs)}

	# build the dataframe
	df_from_PCA = pd.DataFrame(dic.items())
	features_from_PCA = list(df_from_PCA[1])
	return features_from_PCA

@st.cache	
def split_train_test_from_pca(insurance, features_from_PCA):
	
	X_train_df = insurance[features_from_PCA]
	Y_train_df = insurance['QuoteConversion_Flag']

	X_train, X_test, y_train, y_test = train_test_split(
		X_train_df, Y_train_df, test_size=0.3, random_state = 42
	)
	return X_train_df
	
@st.cache	
def model_classifier(insurance, features_from_PCA):
	X_train_df = insurance[features_from_PCA]
	Y_train_df = insurance['QuoteConversion_Flag']
	
	X_train, X_test, y_train, y_test = train_test_split(X_train_df, Y_train_df, stratify=Y_train_df, test_size=0.20)
	# split the train data into train and cross validation by maintaining same distribution of output varaible 'y_train' [stratify=y_train]
	X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.20)

	r_cfl=RandomForestClassifier(random_state=42,n_jobs=-1)
	r_cfl.fit(X_train,y_train)
	sig_clf_rf = CalibratedClassifierCV(r_cfl, method="sigmoid")
	sig_clf_rf.fit(X_train, y_train)
	return sig_clf_rf
	
@st.cache	
def main_process():
	insurance_data = load_data()
	insurance = insurance_data.copy()
	insurance = data_preprocessing_1(insurance)
	insurance = data_preprocessing_2(insurance)
	insurance = data_preprocessing_3(insurance)
	insurance = data_preprocessing_4(insurance)
	insurance = data_preprocessing_5(insurance)
	return insurance
	
@st.cache
def get_features_fromPCA(insurance):	
	X_train = split_train_test(insurance)
	features_from_PCA = get_top_features_pca(X_train)
	return features_from_PCA

@st.cache(allow_output_mutation=True)
def prepare_the_output(insurance, features_from_PCA):
	sig_clf_rfs = model_classifier(insurance, features_from_PCA)
	return sig_clf_rfs
	

def predicted_probability(insurance, input_value, features_from_PCA, sig_clf_rf):
	X_train_df = insurance[features_from_PCA]
	if (int(input_value) in X_train_df.values):
	  input_test_data = X_train_df[X_train_df.QuoteNumber == int(input_value)]
	pred_proba = (sig_clf_rf.predict_proba(input_test_data).max().round(2))*100
	  
	return pred_proba

def predicted_output(insurance, input_value, features_from_PCA, sig_clf_rf):
	X_train_df = insurance[features_from_PCA]
	if (int(input_value) in X_train_df.values):
		input_test_data = X_train_df[X_train_df.QuoteNumber == int(input_value)]
	return sig_clf_rf.predict(input_test_data)[0]

@st.cache
def get_actual_label(get_orig_data, input_value):
	if(int(input_value) in get_orig_data.values):
	  label_data = get_orig_data.QuoteConversion_Flag[get_orig_data.QuoteNumber == int(input_value)]
	else:
	  label_data = 111
	return int(label_data)

insurance = main_process()

get_orig_data = load_data()

features_from_PCA = get_features_fromPCA(insurance)

sig_clf_rf = prepare_the_output(insurance, features_from_PCA)

user_input = st.text_input("Please enter the Quote Number", 349820)

try:
	predicted_proba = predicted_probability(insurance, user_input, features_from_PCA, sig_clf_rf)
	st.text('The predicted probability is: ')
	st.text(predicted_proba)
	try:
		output_p = predicted_output(insurance, user_input, features_from_PCA, sig_clf_rf)
		st.text('The predicted output is: ')
		st.text(output_p)
		st.text('The actual output is: ')
		st.text(get_actual_label(get_orig_data, user_input))
	except:
		st.text('The entered Quote Number does not exist. Please enter the corect Quote Number...')
except:
	st.text('The entered Quote Number does not exist. Please enter the corect Quote Number...')
