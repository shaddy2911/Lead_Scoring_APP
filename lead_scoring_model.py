#supressing warnings
import warnings
warnings.filterwarnings('ignore')
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
import pickle
# visulaisation
from matplotlib.pyplot import xticks
# Data display coustomization
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

#title
st.title('Lead Scoring Model')
image=Image.open('123.png')
st.image(image,use_column_width=True)

def main():
    st.subheader('Lead Scoring Model')
    data1=st.file_uploader('Upload The Dataset',type=['csv'])
    if data1 is not None:
        data=pd.read_csv(data1, encoding='ISO 8859-1')
        st.success('Data Successfully Uploaded')
        st.write('Raw data:',data.head(10))
        p2=st.write('')
        option=('EDA','Train The Model','Batch Predictions')
        side_bar=st.sidebar.radio('Select the Operation',option)
        #data = data.drop(data.loc[:,list(round(100*(data.isnull().sum()/len(data.index)), 2)>70)].columns, 1)
        data = data.drop(['Asymmetrique Activity Index','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Profile Score'],1)
        if side_bar=='EDA':
            if st.button('Shape of The Data'):
                st.write(data.shape)
            if st.button('Describe The Data'):
                st.write(data.describe().T)
            data = data.replace('Select', np.nan)
            data = data.drop(data.loc[:,list(round(100*(data.isnull().sum()/len(data.index)), 2)>70)].columns, 1)
            if st.button('Null Values'):
                st.write(data.isnull().sum())
                st.write('Null Values in Percentage')
                st.write(round(100*(data.isnull().sum()/len(data.index)), 2))
                
                    
            if st.button('Analysis'):
                st.title('Lead Quality Vs Converted')
                data['Lead Quality'] = data['Lead Quality'].replace(np.nan, 'Not Sure')
                fig = plt.figure(figsize=(12, 4))
                sns.countplot(x = "Lead Quality", hue = "Converted", data = data)
                st.pyplot(fig)
                st.write('-'*50)
                
                st.title('City Vs Converted')
                fig1 = plt.figure(figsize=(12, 4))
                sns.countplot(x = "City", hue = "Converted", data = data)
                xticks(rotation = 90)
                st.pyplot(fig1)
                st.write('Mumbai Has The Maximum Number Of Lead Count And Conversion')
                st.write('-'*50)

                st.title('Lead Source Vs Converted')
                data['Lead Source'] = data['Lead Source'].replace(['google'], 'Google')
                data['Lead Source'] = data['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',
                        'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')
                #data['Specialization'] = data['Specialization'].replace(np.nan, 'Others')
                fig2 = plt.figure(figsize=(12, 4))
                sns.countplot(x = "Lead Source", hue = "Converted", data = data)
                xticks(rotation = 90)
                st.pyplot(fig2)
                st.write(""" 
                        1) Google and Direct traffic generates maximum number of leads
                        
                        2)  Conversion Rate of reference leads and leads through welingak website is high """)
                st.write('-'*50)

                st.title('Lead Origin vs Converted')
                fig3 = plt.figure(figsize=(12, 4))
                sns.countplot(x = "Lead Origin", hue = "Converted", data = data)
                xticks(rotation = 90)
                st.pyplot(fig3)
                st.write(""" 
                            1) API and Landing Page Submission have 30-35%  conversion rate but count of lead originated from them are considerable 
                               
                            2) Lead Add Form has more than 90% conversion rate but count of lead are not very high
                            
                            3) Lead Import are very less in count""")
                st.write('-'*50)


                st.title('Last Activity vs Converted')
                data['Last Activity'] = data['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 
                                                       'Visited Booth in Tradeshow', 'Approached upfront',
                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other_Activity')
                fig4 = plt.figure(figsize=(12, 4))
                sns.countplot(x = "Last Activity", hue = "Converted", data = data)
                xticks(rotation = 90)
                st.pyplot(fig4)
                st.write("""
                        1) Most of the lead have their Email opened as their last activity
                        
                        2) Most of the leads are where last activitity is sms sent """)
                st.write('-'*50)

                st.title('Occupation vs Converted')
                data['What is your current occupation'] = data['What is your current occupation'].replace(['Other'], 'Other_Occupation')
                fig5 = plt.figure(figsize=(12, 4))
                sns.countplot(x = "What is your current occupation", hue = "Converted", data = data)
                xticks(rotation = 90)
                st.pyplot(fig5)
                st.write("""
                        1) Working Professionals going for the course have high chances of joining it 
                        
        
                        2) Unemployed leads are the most in numbers but has around 30-35% conversion rate""")
        


        if side_bar=='Train The Model':
            st.subheader('Train The Model')
            data = data.replace('Select', np.nan)
            data = data.drop(data.loc[:,list(round(100*(data.isnull().sum()/len(data.index)), 2)>70)].columns, 1)
            data['Lead Quality'] = data['Lead Quality'].replace(np.nan, 'Not Sure')

            data['City'] = data['City'].replace(np.nan, 'Mumbai')
            data['Specialization'] = data['Specialization'].replace(np.nan, 'Others')
            data['Tags'] = data['Tags'].replace(np.nan, 'Will revert after reading the email')
            data['What matters most to you in choosing a course'] = data['What matters most to you in choosing a course'].replace(np.nan, 'Better Career Prospects')
            data['What is your current occupation'] = data['What is your current occupation'].replace(np.nan, 'Unemployed')
            data['Country'] = data['Country'].replace(np.nan, 'India')
            data.dropna(inplace = True)
            #st.write(round(100*(data.isnull().sum()/len(data.index)), 2))
            data['Lead Source'] = data['Lead Source'].replace(['google'], 'Google')
            data['Lead Source'] = data['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',
            'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')
            data['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])
            sns.boxplot(data['TotalVisits'])
            percentiles = data['TotalVisits'].quantile([0.05,0.95]).values
            data['TotalVisits'][data['TotalVisits'] <= percentiles[0]] = percentiles[0]
            data['TotalVisits'][data['TotalVisits'] >= percentiles[1]] = percentiles[1] 
            percentiles = data['Page Views Per Visit'].quantile([0.05,0.95]).values
            data['Page Views Per Visit'][data['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]
            data['Page Views Per Visit'][data['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]  
            data['Last Activity'] = data['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 
                                                       'Visited Booth in Tradeshow', 'Approached upfront',
                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other_Activity') 
            data['Specialization'] = data['Specialization'].replace(['Others'], 'Other_Specialization')
            data['What is your current occupation'] = data['What is your current occupation'].replace(['Other'], 'Other_Occupation')
            data['Tags'] = data['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized'], 'Other_Tags')
            data = data.drop(['Lead Number','What matters most to you in choosing a course','Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
           'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content',
           'Get updates on DM Content','I agree to pay the amount through cheque','A free copy of Mastering The Interview','Country'],1)
            #st.write(data.shape)
            # Converting some binary variables to 1/0
            varlist =  ['Do Not Email', 'Do Not Call']
            def binary_map(x):
                return x.map({'Yes': 1, "No": 0})
            data[varlist] = data[varlist].apply(binary_map)
            # Creating a dummy variable for some of the categorical variables and dropping the first one
            dummy1 = pd.get_dummies(data[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                              'Tags','Lead Quality','City','Last Notable Activity']], drop_first=True)
            #st.write(dummy1.head())
            data = pd.concat([data, dummy1], axis=1)
            #st.write(data.head()) 
            data = data.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','Tags','Lead Quality','City','Last Notable Activity'], axis = 1)          
            #st.write(data.head())
            from sklearn.model_selection import train_test_split
            X = data.drop(['Prospect ID','Converted'], axis=1)
            #st.write(X.head())
            y = data['Converted']
            #st.write(y.head())
            # Splitting the data into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
            #st.write(X_train.head())
            import statsmodels.api as sm
            # Logistic regression model
            logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
            logm1.fit().summary()
            from sklearn.linear_model import LogisticRegression
            logreg = LogisticRegression()

            from sklearn.feature_selection import RFE
            rfe = RFE(logreg)             
            rfe = rfe.fit(X_train, y_train)
            #st.write(rfe.support_)
            list(zip(X_train.columns, rfe.support_, rfe.ranking_))
            col = X_train.columns[rfe.support_]
            X_train_sm = sm.add_constant(X_train[col])
            logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
            res = logm2.fit()
            #st.write(res.summary())
            col1 = col.drop('Tags_invalid number',1)
            X_train_sm = sm.add_constant(X_train[col1])
            logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
            res = logm2.fit()
            #st.write(res.summary())
            col2 = col1.drop('Tags_wrong number given',1)
            X_train_sm = sm.add_constant(X_train[col2])
            logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
            res = logm2.fit()
            col2 = col1.drop('Tags_wrong number given',1)
            X_train_sm = sm.add_constant(X_train[col2])
            logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
            res = logm2.fit()
            # Getting the predicted values on the train set
            y_train_pred = res.predict(X_train_sm)
            #st.write(y_train_pred[:10])
            y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
            y_train_pred_final['Prospect ID'] = y_train.index
            #st.write(y_train_pred_final.head())
            y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)
             # Let's see the head
            #st.write(y_train_pred_final.head())
            # Let's create columns with different probability cutoffs 
            # Let's check the overall accuracy.
            #st.write((metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)*100))
            numbers = [float(x)/10 for x in range(10)]
            for i in numbers:
                y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
            #y_train_pred_final.head()
            # Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
            cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
            from sklearn.metrics import confusion_matrix

            # TP = confusion[1,1] # true positive 
            # TN = confusion[0,0] # true negatives
            # FP = confusion[0,1] # false positives
            # FN = confusion[1,0] # false negatives

            num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            for i in num:
                cm1 = confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
                total1=sum(sum(cm1))
                accuracy = (cm1[0,0]+cm1[1,1])/total1
    
                speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
                sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
                cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
            #st.write((cutoff_df))
            y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.2 else 0)
            y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))
            k=y_train_pred_final[['Prospect ID','Converted_prob','final_predicted','Lead_Score']]
            st.write(k.tail())
            from sklearn import metrics
            st.write('The Accuracy is (%):',np.round((metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)*100)))
            
            #op=k['Prospect ID'].unique()
            #seb=st.selectbox('Select The ID',(op))
            
            #result = k[k['Prospect ID'] == seb]
            #st.write(result)
            #st.download_button(label='Download File',data=k.to_csv(),file_name='lead_scores.csv',mime='csv')

        if side_bar=='Batch Predictions':
            data1=st.file_uploader('Upload The Dataset',type=['csv'],key=1)
            if data1 is not None:
                data=pd.read_csv(data1, encoding='ISO 8859-1')
                st.success('Data Successfully Uploaded')
                #st.write('Raw data:',data.tail(10))
                data = data.drop(['Asymmetrique Activity Index','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Profile Score'],1)
                data = data.replace('Select', np.nan)
                data = data.drop(data.loc[:,list(round(100*(data.isnull().sum()/len(data.index)), 2)>70)].columns, 1)
                data['Lead Quality'] = data['Lead Quality'].replace(np.nan, 'Not Sure')

                data['City'] = data['City'].replace(np.nan, 'Mumbai')
                data['Specialization'] = data['Specialization'].replace(np.nan, 'Others')
                data['Tags'] = data['Tags'].replace(np.nan, 'Will revert after reading the email')
                data['What matters most to you in choosing a course'] = data['What matters most to you in choosing a course'].replace(np.nan, 'Better Career Prospects')
                data['What is your current occupation'] = data['What is your current occupation'].replace(np.nan, 'Unemployed')
                data['Country'] = data['Country'].replace(np.nan, 'India')
                data.dropna(inplace = True)
                 #st.write(round(100*(data.isnull().sum()/len(data.index)), 2))
                data['Lead Source'] = data['Lead Source'].replace(['google'], 'Google')
                data['Lead Source'] = data['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',
                 'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')
                data['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])
                sns.boxplot(data['TotalVisits'])
                percentiles = data['TotalVisits'].quantile([0.05,0.95]).values
                data['TotalVisits'][data['TotalVisits'] <= percentiles[0]] = percentiles[0]
                data['TotalVisits'][data['TotalVisits'] >= percentiles[1]] = percentiles[1] 
                percentiles = data['Page Views Per Visit'].quantile([0.05,0.95]).values
                data['Page Views Per Visit'][data['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]
                data['Page Views Per Visit'][data['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]  
                data['Last Activity'] = data['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 
                                                        'Visited Booth in Tradeshow', 'Approached upfront',
                                                        'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other_Activity') 
                data['Specialization'] = data['Specialization'].replace(['Others'], 'Other_Specialization')
                data['What is your current occupation'] = data['What is your current occupation'].replace(['Other'], 'Other_Occupation')
                data['Tags'] = data['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                      'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                     'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                     'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                     'University not recognized'], 'Other_Tags')
                data = data.drop(['Lead Number','What matters most to you in choosing a course','Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
                 'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content',
                 'Get updates on DM Content','I agree to pay the amount through cheque','A free copy of Mastering The Interview','Country'],1)
                #st.write(data.shape)
                 # Converting some binary variables to 1/0
                varlist =  ['Do Not Email', 'Do Not Call']
                def binary_map(x):
                    return x.map({'Yes': 1, "No": 0})
                data[varlist] = data[varlist].apply(binary_map)
                 # Creating a dummy variable for some of the categorical variables and dropping the first one
                dummy1 = pd.get_dummies(data[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                               'Tags','Lead Quality','City','Last Notable Activity']], drop_first=True)
                #st.write(dummy1.head())
                data = pd.concat([data, dummy1], axis=1)
                #st.write(data.head()) 
                data = data.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','Tags','Lead Quality','City','Last Notable Activity'], axis = 1)          
                #st.write(data.head())
                from sklearn.model_selection import train_test_split
                X = data.drop(['Prospect ID','Converted'], axis=1)
                #st.write(X.head())
                y = data['Converted']
                #st.write(y.head())
                 # Splitting the data into train and test
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
                #st.write(X_train.head())
                import statsmodels.api as sm
                logm1 = sm.GLM(y_train,(sm.add_constant(X_train)),family = sm.families.Binomial())
                logm1.fit().summary()
                from sklearn.linear_model import LogisticRegression
                logreg = LogisticRegression()

                from sklearn.feature_selection import RFE
                rfe = RFE(logreg)             
                rfe = rfe.fit(X_train, y_train)

                from sklearn.feature_selection import RFE
                rfe = RFE(logreg)             
                rfe = rfe.fit(X_train, y_train)
                #st.write(rfe.support_)
                list(zip(X_train.columns, rfe.support_, rfe.ranking_))
                col = X_train.columns[rfe.support_]
                X_train_sm = sm.add_constant(X_train[col])
                logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
                res = logm2.fit()
                #st.write(res.summary())
                #pickle_in = open('lead_score.pkl', 'rb') 
                #ls = pickle.load(pickle_in)

                col1 = col.drop('Tags_invalid number',1)
                X_train_sm = sm.add_constant(X_train[col1])
                logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
                res = logm2.fit()
                #st.write(res.summary())
                col2 = col1.drop('Tags_wrong number given',1)
                X_train_sm = sm.add_constant(X_train[col2])
                logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
                res = logm2.fit()
                col2 = col1.drop('Tags_wrong number given',1)
                X_train_sm = sm.add_constant(X_train[col2])
                logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
                res = logm2.fit()
               # Getting the predicted values on the train set
                y_train_pred = res.predict(X_train_sm)
                #st.write(y_train_pred[:10])
                y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
                y_train_pred_final['Prospect ID'] = y_train.index
                #st.write(y_train_pred_final.head())
                y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)
                # Let's see the head
                #st.write(y_train_pred_final.head())
                # Let's create columns with different probability cutoffs 
                # Let's check the overall accuracy.
                #st.write((metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)*100))
                numbers = [float(x)/10 for x in range(10)]
                for i in numbers:
                    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
                #y_train_pred_final.head()
                # Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
                cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
                from sklearn.metrics import confusion_matrix

                # TP = confusion[1,1] # true positive 
                # TN = confusion[0,0] # true negatives
                # FP = confusion[0,1] # false positives
                # FN = confusion[1,0] # false negatives

                num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
                for i in num:
                    cm1 = confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
                    total1=sum(sum(cm1))
                    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
                    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
                    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
                    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
                #st.write((cutoff_df))
                y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.2 else 0)
                y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))
                k=y_train_pred_final[['Prospect ID','Converted_prob','final_predicted','Lead_Score']]
                #st.write(k.tail())
                from sklearn import metrics
                #st.write('The Accuracy is (%):',np.round((metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)*100)))
            
                op=k['Prospect ID'].unique()
                seb=st.selectbox('Select The ID',(op))
            
                result = k[k['Prospect ID'] == seb]
                st.write(result[['Prospect ID','Lead_Score']])
                st.download_button(label='Download File',data=k.to_csv(),file_name='lead_scores.csv',mime='csv')
        

        
    

            






                    
                    
                    

            
                



    
    










if __name__ == '__main__':
    main()
