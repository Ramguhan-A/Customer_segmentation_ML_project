import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("Customer Data.csv")
print(df.head())

print(df.info())

print(df.describe())

#Finding NAN values
missing_values = df.isnull().sum()
print("Missing_values", missing_values)

#Filling NAN's
df['MINIMUM_PAYMENTS']=df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS']).mean()
df['CREDIT_LIMIT']=df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT']).mean()
print(df.isnull().sum().sum())

#Dropping the column
df.drop("CUST_ID",axis=1,inplace=True)

#checking the distribtuion
plt.figure(figsize(7,5))
sns.histplot(df["BALANCE"])
plt.show()

plt.figure(figsize(7,5))
sns.histplot(df["PURCHASES_FREQUENCY"])
plt.show()

#standadicing the dataset
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

print(type(df_scaled))
print(type(df))

df_scaled = pd.DataFrame(df_scaled,columns=df.columns)

#Applying PCA
pca = PCA(n_components=6)
Principle_component = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(Principle_component, columns=["PC1","PC2","PC3","PC4","PC5","PC6"])
print(df_pca.head())

#Finding cluster using k means Elbow_method
inertia=[]
range_v = range(1,15)
for i in range_v:
    kmeans = KMeans(n_clusters=i,random_state=42)
    kmeans.fit_predict(df_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range_v,inertia,'rx-')
plt.title("Elbow_method")
plt.xlabel("range_v")
plt.ylabel("inertia")
plt.show()

#Clustering models using kmean
k_means_model = KMeans(4,random_state=42)
k_means_model.fit_predict(df_pca)

#Concatinating PCA data with kmean label
df_pca_clustered = pd.concat([df_pca,pd.DataFrame({"Clusters":k_means_model.labels_})],axis=1)
print(df_pca_clustered.head())

plt.figure(figsize=(10,8))

#Scatterplot for k means clustering
Cluster_plot = sns.scatterplot(x="PC1",y='PC2',data=df_pca_clustered,hue="Clusters",palette="viridis")
plt.title("KMeans Cluster plot")
plt.show()

#concatinating the labels with original data
cluster_center = pd.concat([df,pd.DataFrame({"Cluster":k_means_model.labels_})],axis=1)
print(cluster_center)


#Checking PCA loading
# Create a DataFrame for PCA component contributions
pca_components = pd.DataFrame(pca.components_, columns=df.columns, index=["PC1","PC2","PC3","PC4","PC5","PC6"])
pd.set_option('display.max_columns', None)
print(pca_components.T)


print("PCA Explained variance ratio: ",pca.explained_variance_ratio_)

print("PCA Cumulative Explained variance ratio: ",np.cumsum(pca.explained_variance_ratio_))

# plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), 'bo-')
# plt.axhline(y=0.8, color='r', linestyle='--')  # 80% variance threshold
# plt.xlabel("Number of Principal Components")
# plt.ylabel("Cumulative Explained Variance")
# plt.title("Scree Plot (Variance Explained by PCs)")
# plt.show()

print("_----------------------------------------")
# Compute mean values of each feature per cluster
cluster_summary = cluster_center.groupby("Cluster").mean()
print(cluster_summary)

cluster_summary.to_csv("cluster_summary.csv")

cluster_0 = cluster_center[cluster_center["Cluster"]==0]
print("Cluster 0",cluster_0)

# EDA - Understanding how cluster separated
sns.barplot(x="Cluster",y="BALANCE",data=cluster_center)
plt.show()
sns.barplot(x="Cluster",y="PURCHASES",data=cluster_center)
plt.show()
sns.barplot(x="Cluster",y="CASH_ADVANCE",data=cluster_center)
plt.show()
sns.barplot(x="Cluster",y="PURCHASES_FREQUENCY",data=cluster_center)
plt.show()
sns.barplot(x="Cluster",y="PAYMENTS",data=cluster_center)
plt.show()

#Saving Clustered CSV file
cluster_center.to_csv("Clustered_Customer_Dataset.csv")

X = cluster_center.drop(["Cluster"],axis=1)
y = cluster_center["Cluster"]

#Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Decision tree model
model_decision_tree = DecisionTreeClassifier(criterion="entropy")
model_decision_tree.fit(X_train,y_train)

#Predicting y
y_pred = model_decision_tree.predict(X_test)

#Checking the score
print(classification_report(y_test,y_pred))

#Deploying the model (Converting into pickle file)
import pickle
with open('model_decision_tree.pkl','wb') as obj:
    pickle.dump(model_decision_tree, obj)