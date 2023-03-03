import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import squarify
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

# 1. Read data
df = pd.read_fwf("CDNOW_master.txt",
                 header=None,
                 names = ["CustomerId", "InvoiceDate", 
                          "Quantity", "TotalSales"])

#--------------
# GUI
st.title("Data Science Project")
st.write("## Customer Segmentation")
# # Upload file
# uploaded_file = st.file_uploader("Choose a file", type=['csv'])
# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file, encoding='latin-1')
#     data.to_csv("data_new.csv", index = False)

# 2. Data information
st.markdown('##### Data Sample')
st.write(pd.DataFrame(df.head()))
df.head(5)
# # df.info()
# # df.describe()

# # Chuyển dữ liệu cột InvoiceDate về đúng định dạng
# df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y%m%d')
# st.markdown('##### Data convert')
# st.write(pd.DataFrame(df.head()))
# df.head(5)
# # df.info()
# # Kiểm tra dữ liêu Null, Nan
# df.isnull().sum()
# df.isna().sum()

# # Kiểm tra dữ liệu trùng
# duplicate = df[df.duplicated()]
# duplicate.head()
# duplicate.shape
# # Xem xét kỹ hơn dữ liệu trùng
# df[df.duplicated()].value_counts()
# # Xóa giá trị trùng
# df = df.drop_duplicates()
# df.shape
# df.describe()
# # Xem xét kỹ hơn dữ liệu cần thao tác
# print('Khoảng thời gian phát sinh dao dịch đầu tiên {} đến giao dịch mới nhất {}'.format(df['InvoiceDate'].min(), df['InvoiceDate'].max()))
# print('{:,} transactions don\'t have a customer id'.format(df[df.CustomerId.isnull()].shape[0]))
# print('{:,} unique customer_id'.format(len(df.CustomerId.unique())))

# df.rename(columns = {'InvoiceDate':'Day'}, inplace = True)
# # Lưu file tiền xử lý
# df.to_csv('CDNOW_master_clean.csv')
# df.head()

# # 3. Tạo RFM phân tích cho từng khách hàng
# # Convert string to date, get max date of dataframe
# max_date = df['Day'].max().date()
# Recency = lambda x : (max_date - x.max().date()).days
# Frequency  = lambda x: len(x.unique())
# Monetary = lambda x : round(sum(x), 2)

# df_RFM = df.groupby('CustomerId').agg({'Day': Recency,
#                                         'Quantity': Frequency,  
#                                         'TotalSales': Monetary })
# df_RFM.head()
# # Rename the columns of DataFrame
# df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
# # Descending Sorting 
# df_RFM = df_RFM.sort_values('Monetary', ascending=False)
# df_RFM.head()
# # Lưu file RFM
# df_RFM.to_csv("Data_RFM.csv")
# ### Visualization
# plt.figure(figsize=(8,10))
# plt.subplot(3, 1, 1)
# sns.distplot(df_RFM['Recency'])# Plot distribution of R
# plt.subplot(3, 1, 2)
# sns.distplot(df_RFM['Frequency'])# Plot distribution of F
# plt.subplot(3, 1, 3)
# sns.distplot(df_RFM['Monetary']) # Plot distribution of M
# plt.show()

# #Define function that will computer the IQR
# def limit(i):
#     Q1 = df_RFM[i].quantile(0.5)
#     Q3 = df_RFM[i].quantile(0.95)
#     IQR = Q3 - Q1
    
#     #Determine the upper and lower limits and the extreme upper and lower limits
#     lower_limit = df_RFM[i].quantile(0.5) - (IQR * 1.5)
#     lower_limit_extreme = df_RFM[i].quantile(0.5) - (IQR * 3)
#     upper_limit = df_RFM[i].quantile(0.95) + (IQR * 1.5)
#     upper_limit_extreme = df_RFM[i].quantile(0.5) + (IQR * 3)
    
#     #Print out the exact upper and lower limits
#     print('Lower Limit:', lower_limit)
#     print('Lower Limit Extreme:', lower_limit_extreme)
#     print('Upper Limit:', upper_limit)
#     print('Upper Limit Extreme:', upper_limit_extreme)


# #Define function that computes the percent outliers in the data    
# def percent_outliers(i):
#     Q1 = df_RFM[i].quantile(0.5)
#     Q3 = df_RFM[i].quantile(0.95)
#     IQR = Q3 - Q1
    
#     #Determine the upper and lower limits and the extreme upper and lower limits
#     lower_limit = df_RFM[i].quantile(0.5) - (IQR * 1.5)
#     lower_limit_extreme = df_RFM[i].quantile(0.5) - (IQR * 3)
#     upper_limit = df_RFM[i].quantile(0.95) + (IQR * 1.5)
#     upper_limit_extreme = df_RFM[i].quantile(0.95) + (IQR * 3)
    
#     #Display the percentage of outliers
#     print('Lower Limit: {} %'.format(df_RFM[(df_RFM[i] >= lower_limit)].shape[0]/ df_RFM.shape[0]*100))
#     print('Lower Limit Extereme: {} %'.format(df_RFM[(df_RFM[i] >= lower_limit_extreme)].shape[0]/df_RFM.shape[0]*100))
#     print('Upper Limit: {} %'.format(df_RFM[(df_RFM[i] >= upper_limit)].shape[0]/ df_RFM.shape[0]*100))
#     print('Upper Limit Extereme: {} %'.format(df_RFM[(df_RFM[i] >= upper_limit_extreme)].shape[0]/df_RFM.shape[0]*100))

# #Check outliers in recency feature
# sns.boxplot(x = df_RFM["Recency"])
# #Check outliers in frequency feature
# sns.boxplot(x = df_RFM["Frequency"])
# #Check outliers in monetary feature
# sns.boxplot(x = df_RFM["Monetary"])

# #Hiển thị giới hạn trên và dưới và giới hạn cực đại của Monetary cùng với tỷ lệ phần trăm
# print(limit('Monetary'))
# print('-'*50)
# print(percent_outliers('Monetary'))
# #Remove outliers from monetary that outside the 95% max limit of the data distribution
# outliers1_drop = df_RFM[(df_RFM['Monetary'] > 883)].index
# df_RFM.drop(outliers1_drop, inplace = True)
# df_RFM.shape
# df_RFM.to_csv("Data_RFM_clean.csv")

# #4. Tính toán RFM quartiles
# #Compute the recency score (the fewer days the better ~~ more recent)
# r_labels = range(4, 0, -1)
# # r_groups = pd.qcut(df_RFM.Recency.rank(method='first'), q = 4, labels = r_labels).astype('int')
# r_groups = pd.qcut(df_RFM.Recency.rank(method='first'), q = 4, labels = r_labels)
# #Compute frequency score
# f_lables = range(1, 5)
# f_groups = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=4, labels=f_lables)
# # f_groups = pd.qcut(df_RFM.Frequency.rank(method = 'first'), 3).astype('str')

# #Compute monetary score
# m_labels = range(1, 5)
# # m_groups = pd.qcut(df_RFM.Monetary.rank(method='first'), q = 4, labels = m_labels).astype('int')
# m_groups = pd.qcut(df_RFM.Monetary.rank(method='first'), q = 4, labels = m_labels)

# [*r_labels]
# #Create a column for each of the scores created
# df_RFM = df_RFM.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)

# #5. Concat RFM quartile values to create RFM Segments
# def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
# df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)
# df_RFM.head()
# df_RFM.tail()

# rfm_count_unique = df_RFM.groupby('RFM_Segment')['RFM_Segment'].nunique()
# print(rfm_count_unique.sum())

# #6. Calculate RFM score and level
# # Calculate RFM_Score
# df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)
# df_RFM.head()
# df_RFM.RFM_Score.unique()

# #7. Modeling wiht KMeans
# df_k = df_RFM[['Recency','Frequency','Monetary']]
# df_k

# from sklearn.cluster import KMeans
# sse = {}
# for k in range(1, 20):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(df_k)
#     sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

# plt.title('The Elbow Method')
# plt.xlabel('k')
# plt.ylabel('SSE')
# sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
# plt.show()

# # Build model with k=4
# model = KMeans(n_clusters=4, random_state=42)
# model.fit(df_k)
# model.labels_.shape

# df_k["Cluster"] = model.labels_
# df_k.groupby('Cluster').agg({
#     'Recency':'mean',
#     'Frequency':'mean',
#     'Monetary':['mean', 'count']}).round(2)

# # Calculate average values for each RFM_Level, and return a size of each segment 
# rfm_agg2 = df_k.groupby('Cluster').agg({
#     'Recency': 'mean',
#     'Frequency': 'mean',
#     'Monetary': ['mean', 'count']}).round(0)

# rfm_agg2.columns = rfm_agg2.columns.droplevel()
# rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
# rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)

# # Reset the index
# rfm_agg2 = rfm_agg2.reset_index()

# # Change thr Cluster Columns Datatype into discrete values
# rfm_agg2['Cluster'] = 'Cluster '+ rfm_agg2['Cluster'].astype('str')

# # Print the aggregated dataset
# rfm_agg2

# #Create our plot and resize it.
# fig = plt.gcf()
# ax = fig.add_subplot()
# fig.set_size_inches(14, 10)
# colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
#                'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}

# squarify.plot(sizes=rfm_agg2['Count'],
#               text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
#               color=colors_dict2.values(),
#               label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg2.iloc[i])
#                       for i in range(0, len(rfm_agg2))], alpha=0.5 )


# plt.title("Customers Segments",fontsize=26,fontweight="bold")
# plt.axis('off')

# plt.savefig('Unsupervised Segments.png')
# plt.show()

# import plotly.express as px

# fig = px.scatter_3d(rfm_agg2, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',
#                     color = 'Cluster', opacity=0.3)
# fig.update_traces(marker=dict(size=20),
                  
#                   selector=dict(mode='markers'))
# fig.show()

# import plotly.express as px

# fig = px.scatter(rfm_agg2, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
#            hover_name="Cluster", size_max=100)
# fig.show()

# # Save model:
# import pickle
# pkl_filename = "customer_segmentation_model.pkl"  
# with open(pkl_filename, 'wb') as file:  
#     pickle.dump(model, file)

# # Đọc model
# import pickle
# with open(pkl_filename, 'rb') as file:  
#     customer_segmentation_model = pickle.load(file)

# GUI
menu = ["Overview", "Model", "Conclusion"]
choice = st.sidebar.selectbox('MENU', menu)
if choice == 'Overview':    
    st.subheader("Overview")
    st.write("""
    ###### Customer segmentation is the process of dividing a customer base into distinct groups of individuals that have similar characteristics. This process makes it easier to target specific groups of customers with tailored products, services, and marketing strategies.
    """)  
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for customer segmentation.""")
    st.image("Customer_Segmentation.jpg")

elif choice == 'Model':
    st.subheader("Model")
    st.write("##### 1. Some data")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))  
    # # Đọc model
    # import pickle
    # with open(pkl_filename, 'rb') as file:  
    #     customer_segmentation_model = pickle.load(file)
    # df.info()
    # df.describe()

    # Chuyển dữ liệu cột InvoiceDate về đúng định dạng
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y%m%d')
    st.markdown('##### Data convert')
    st.write(pd.DataFrame(df.head()))
    df.head(5)
    # df.info()
    # Kiểm tra dữ liêu Null, Nan
    df.isnull().sum()
    df.isna().sum()

    # Kiểm tra dữ liệu trùng
    duplicate = df[df.duplicated()]
    duplicate.head()
    duplicate.shape
    # Xem xét kỹ hơn dữ liệu trùng
    df[df.duplicated()].value_counts()
    # Xóa giá trị trùng
    df = df.drop_duplicates()
    df.shape
    df.describe()
    # Xem xét kỹ hơn dữ liệu cần thao tác
    print('Khoảng thời gian phát sinh dao dịch đầu tiên {} đến giao dịch mới nhất {}'.format(df['InvoiceDate'].min(), df['InvoiceDate'].max()))
    print('{:,} transactions don\'t have a customer id'.format(df[df.CustomerId.isnull()].shape[0]))
    print('{:,} unique customer_id'.format(len(df.CustomerId.unique())))

    df.rename(columns = {'InvoiceDate':'Day'}, inplace = True)
    # Lưu file tiền xử lý
    df.to_csv('CDNOW_master_clean.csv')
    df.head()

    # 3. Tạo RFM phân tích cho từng khách hàng
    # Convert string to date, get max date of dataframe
    max_date = df['Day'].max().date()
    Recency = lambda x : (max_date - x.max().date()).days
    Frequency  = lambda x: len(x.unique())
    Monetary = lambda x : round(sum(x), 2)

    df_RFM = df.groupby('CustomerId').agg({'Day': Recency,
                                            'Quantity': Frequency,  
                                            'TotalSales': Monetary })
    df_RFM.head()
    # Rename the columns of DataFrame
    df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
    # Descending Sorting 
    df_RFM = df_RFM.sort_values('Monetary', ascending=False)
    df_RFM.head()
    # Lưu file RFM
    df_RFM.to_csv("Data_RFM.csv")
    ### Visualization
    plt.figure(figsize=(8,10))
    plt.subplot(3, 1, 1)
    sns.distplot(df_RFM['Recency'])# Plot distribution of R
    plt.subplot(3, 1, 2)
    sns.distplot(df_RFM['Frequency'])# Plot distribution of F
    plt.subplot(3, 1, 3)
    sns.distplot(df_RFM['Monetary']) # Plot distribution of M
    plt.show()

    #Define function that will computer the IQR
    def limit(i):
        Q1 = df_RFM[i].quantile(0.5)
        Q3 = df_RFM[i].quantile(0.95)
        IQR = Q3 - Q1
        
        #Determine the upper and lower limits and the extreme upper and lower limits
        lower_limit = df_RFM[i].quantile(0.5) - (IQR * 1.5)
        lower_limit_extreme = df_RFM[i].quantile(0.5) - (IQR * 3)
        upper_limit = df_RFM[i].quantile(0.95) + (IQR * 1.5)
        upper_limit_extreme = df_RFM[i].quantile(0.5) + (IQR * 3)
        
        #Print out the exact upper and lower limits
        print('Lower Limit:', lower_limit)
        print('Lower Limit Extreme:', lower_limit_extreme)
        print('Upper Limit:', upper_limit)
        print('Upper Limit Extreme:', upper_limit_extreme)


    #Define function that computes the percent outliers in the data    
    def percent_outliers(i):
        Q1 = df_RFM[i].quantile(0.5)
        Q3 = df_RFM[i].quantile(0.95)
        IQR = Q3 - Q1
        
        #Determine the upper and lower limits and the extreme upper and lower limits
        lower_limit = df_RFM[i].quantile(0.5) - (IQR * 1.5)
        lower_limit_extreme = df_RFM[i].quantile(0.5) - (IQR * 3)
        upper_limit = df_RFM[i].quantile(0.95) + (IQR * 1.5)
        upper_limit_extreme = df_RFM[i].quantile(0.95) + (IQR * 3)
        
        #Display the percentage of outliers
        print('Lower Limit: {} %'.format(df_RFM[(df_RFM[i] >= lower_limit)].shape[0]/ df_RFM.shape[0]*100))
        print('Lower Limit Extereme: {} %'.format(df_RFM[(df_RFM[i] >= lower_limit_extreme)].shape[0]/df_RFM.shape[0]*100))
        print('Upper Limit: {} %'.format(df_RFM[(df_RFM[i] >= upper_limit)].shape[0]/ df_RFM.shape[0]*100))
        print('Upper Limit Extereme: {} %'.format(df_RFM[(df_RFM[i] >= upper_limit_extreme)].shape[0]/df_RFM.shape[0]*100))

    #Check outliers in recency feature
    sns.boxplot(x = df_RFM["Recency"])
    #Check outliers in frequency feature
    sns.boxplot(x = df_RFM["Frequency"])
    #Check outliers in monetary feature
    sns.boxplot(x = df_RFM["Monetary"])

    #Hiển thị giới hạn trên và dưới và giới hạn cực đại của Monetary cùng với tỷ lệ phần trăm
    print(limit('Monetary'))
    print('-'*50)
    print(percent_outliers('Monetary'))
    #Remove outliers from monetary that outside the 95% max limit of the data distribution
    outliers1_drop = df_RFM[(df_RFM['Monetary'] > 883)].index
    df_RFM.drop(outliers1_drop, inplace = True)
    df_RFM.shape
    df_RFM.to_csv("Data_RFM_clean.csv")

    #4. Tính toán RFM quartiles
    #Compute the recency score (the fewer days the better ~~ more recent)
    r_labels = range(4, 0, -1)
    # r_groups = pd.qcut(df_RFM.Recency.rank(method='first'), q = 4, labels = r_labels).astype('int')
    r_groups = pd.qcut(df_RFM.Recency.rank(method='first'), q = 4, labels = r_labels)
    #Compute frequency score
    f_lables = range(1, 5)
    f_groups = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=4, labels=f_lables)
    # f_groups = pd.qcut(df_RFM.Frequency.rank(method = 'first'), 3).astype('str')

    #Compute monetary score
    m_labels = range(1, 5)
    # m_groups = pd.qcut(df_RFM.Monetary.rank(method='first'), q = 4, labels = m_labels).astype('int')
    m_groups = pd.qcut(df_RFM.Monetary.rank(method='first'), q = 4, labels = m_labels)

    [*r_labels]
    #Create a column for each of the scores created
    df_RFM = df_RFM.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)

    #5. Concat RFM quartile values to create RFM Segments
    def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
    df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)
    df_RFM.head()
    df_RFM.tail()

    rfm_count_unique = df_RFM.groupby('RFM_Segment')['RFM_Segment'].nunique()
    print(rfm_count_unique.sum())

    #6. Calculate RFM score and level
    # Calculate RFM_Score
    df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)
    df_RFM.head()
    df_RFM.RFM_Score.unique()

    #7. Modeling wiht KMeans
    df_k = df_RFM[['Recency','Frequency','Monetary']]
    df_k

    from sklearn.cluster import KMeans
    sse = {}
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_k)
        sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

    plt.title('The Elbow Method')
    plt.xlabel('k')
    plt.ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    plt.show()

    # Build model with k=4
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(df_k)
    model.labels_.shape

    df_k["Cluster"] = model.labels_
    df_k.groupby('Cluster').agg({
        'Recency':'mean',
        'Frequency':'mean',
        'Monetary':['mean', 'count']}).round(2)

    # Calculate average values for each RFM_Level, and return a size of each segment 
    rfm_agg2 = df_k.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg2.columns = rfm_agg2.columns.droplevel()
    rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)

    # Reset the index
    rfm_agg2 = rfm_agg2.reset_index()

    # Change thr Cluster Columns Datatype into discrete values
    rfm_agg2['Cluster'] = 'Cluster '+ rfm_agg2['Cluster'].astype('str')

    # Print the aggregated dataset
    rfm_agg2

    #Create our plot and resize it.
    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(14, 10)
    colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
                'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}

    squarify.plot(sizes=rfm_agg2['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict2.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg2.iloc[i])
                        for i in range(0, len(rfm_agg2))], alpha=0.5 )


    plt.title("Customers Segments",fontsize=26,fontweight="bold")
    plt.axis('off')

    plt.savefig('Unsupervised Segments.png')
    plt.show()

    import plotly.express as px

    fig = px.scatter_3d(rfm_agg2, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',
                        color = 'Cluster', opacity=0.3)
    fig.update_traces(marker=dict(size=20),
                    
                    selector=dict(mode='markers'))
    fig.show()

    import plotly.express as px

    fig = px.scatter(rfm_agg2, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
            hover_name="Cluster", size_max=100)
    fig.show()

    # Save model:
    import pickle
    pkl_filename = "customer_segmentation_model.pkl"  
    with open(pkl_filename, 'wb') as file:  
        pickle.dump(model, file)

    # Đọc model
    import pickle
    with open(pkl_filename, 'rb') as file:  
        customer_segmentation_model = pickle.load(file)

    st.write("##### 5. Summary: This model is good enough for customer segmentation.")

elif choice == 'Conclusion':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)
            # st.write(lines.columns)
            lines = lines[0]     
            flag = True       
    if type=="Input":        
        email = st.text_area(label="Input your content:")
        if email!="":
            lines = np.array([email])
            flag = True
    
    if flag:
        st.write("Content:")
        # if len(lines)>0:
        #     st.code(lines)        
        #     x_new = count_model.transform(lines)        
        #     y_pred_new = ham_spam_model.predict(x_new)       
        #     st.code("New predictions (0: Ham, 1: Spam): " + str(y_pred_new))
    

