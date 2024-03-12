# CryptoClustering
## Find the Best Value for k by Using the Original Data (15 points)
    - Created the elbow curve so that we could identify the best value for k by visualizing it on a plot

            k = list(range(1, 11))
            inertia = []
            for i in k:
                k_model = KMeans(n_clusters=i, random_state=0)
                k_model.fit(df_data_transformed)
                inertia.append(k_model.inertia_)
            elbow_data = {"k": k, "inertia": inertia}
            df_elbow = pd.DataFrame(elbow_data)
            elbow1 = df_elbow.hvplot.line(
                x="k",
                y="inertia",
                title="Elbow Curve",
                xticks=k
            )
            elbow1


    - What's the best value for k? 
        - The best value for k based on the elbow curve is 4 because that's where we start seeing diminishing returns on the elbow curve. 
    - **Code for this part of the assignment was taken from activities performed in class**

## Cluster the Cryptocurrencies with K-Means by Using the Original Data (10 points)
    - Initialize the K-means model with four clusters by using the best value for k. (1 point)
    - Fit the K-means model by using the original data. (1 point)
    - cPredict the clusters for grouping the cryptocurrencies by using the original data. Review the resulting array of cluster values. (3 points)
    - Create a copy of the original data, and then add a new column of the predicted clusters. (1 point)
    - Using hvPlot, create a scatter plot by setting x="price_change_percentage_24h" and y="price_change_percentage_7d". Color the graph points with the labels that you found by using K-means. Then add the crypto name to the hover_cols parameter to identify the cryptocurrency that each data point represents. (4 points)

            model = KMeans(n_clusters=4, random_state=0)
            model.fit(df_data_transformed)
            kmeans_predictions = model.predict(df_data_transformed)
            print(kmeans_predictions)
            df_data_transformed_predictions = df_data_transformed.copy()
            df_data_transformed_predictions["PredictedData"] = kmeans_predictions
            df_data_transformed_predictions.head()
            scatter1 = df_data_transformed_predictions.hvplot.scatter(
                x="price_change_percentage_24h",
                y="price_change_percentage_7d",
                by="PredictedData",
                hover_cols=["coin_id"]
            )
            scatter1

    - **Code for this part of the assignment was taken from activities performe din class

## Optimize the Clusters with Principal Component Analysis (10 points)
    - Create a PCA model instance, and set n_components=3. (1 point)
    - Use the PCA model to reduce the features to three principal components. Then review the first five rows of the DataFrame. (2 points)
    - Get the explained variance to determine how much information can be attributed to each principal component. (2 points)
    - Create a new DataFrame with the PCA data. Be sure to set the coin_id index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame. (2 points)

            pca = PCA(n_components=3)
            data_transformed_pca = pca.fit_transform(df_data_transformed)
            data_transformed_pca[:5]
            pca.explained_variance_ratio_
            data_transformed_pca_df = pd.DataFrame(
                data_transformed_pca,
                columns=["PC1", "PC2", "PC3"]
            )
            data_transformed_pca_df["coin_id"] = df_market_data.index
            data_transformed_pca_df = data_transformed_pca_df.set_index("coin_id")


    - What is the total explained variance of the three principal components?
        - The total explained variance of the 3 principal components is roughly .895 taken from the 3 values of PCA. 
    - **Code for this portion of the assignment was taken from activities perofrmed in class.

## Find the Best Value for k by Using the PCA Data (10 points)
    - Code the elbow method algorithm, and use the PCA data to find the best value for k. Use a range from 1 to 11. (2 points)
    - To visually identify the optimal value for k, plot a line chart of all the inertia values computed with the different values of k. (5 points)

            k2 = list(range(1, 11))
            inertia2 = []
            for i in k2:
                k_model = KMeans(n_clusters=i, random_state=1)
                k_model.fit(data_transformed_pca_df)
                inertia2.append(k_model.inertia_)
            elbow2_data = {"k": k2, "inertia": inertia2}
            df_elbow2 = pd.DataFrame(elbow2_data)
            elbow2 = df_elbow2.hvplot.line(
                x="k",
                y="inertia",
                title="Elbow Curve 2",
                xticks=k
            )
            elbow2

    - Answer the following questions: What’s the best value for k when using the PCA data? Does it differ from the best value for k that you found by using the original data? (3 points)
        - I would still say the best value for k would be 4 because that's when you start to get diminishing returns.
        - No. I identified as the optimal k value for the origial data as well. However, the inertia values are lower on the 2nd elbow curve compared to the first one
    - **Code for this part of the assignment was taken from activities performed in class.**

## Cluster the Cryptocurrencies with K-means by Using the PCA Data (10 points)
    - Initialize the K-means model with four clusters by using the best value for k. (1 point)
    - Fit the K-means model by using the PCA data. (1 point)
    - Predict the clusters for grouping the cryptocurrencies by using the PCA data. Review the resulting array of cluster values. (3 points)
    - Create a copy of the DataFrame with the PCA data, and then add a new column to store the predicted clusters. (1 point)
    - Using hvPlot, create a scatter plot by setting x="PC1" and y="PC2". Color the graph points with the labels that you found by using K-means. Then add the crypto name to the hover_cols parameter to identify the cryptocurrency that each data point represents. (4 points)

            model = KMeans(n_clusters=4, random_state=0)
            model.fit(data_transformed_pca_df)
            k_cluster = model.predict(data_transformed_pca_df)
            pca_cluster_df = data_transformed_pca_df.copy()
            pca_cluster_df["Predictions2"] = k_cluster
            scatter2 = pca_cluster_df.hvplot.scatter(
                x="PC1",
                y="PC2",
                by="Predictions2",
                hover_cols=["coin_id"]
            )
            scatter2

    - **Code for this part of the assignment was taken from activities performed in class.**

## Visualize and Compare the Results
    - Create a composite plot by using hvPlot and the plus sign (+) operator to compare the elbow curve that you created from the original data with the one that you created from the PCA data. (5 points)
    - Create a composite plot by using hvPlot and the plus (+) operator to compare the cryptocurrency clusters that resulted from using the original data with those that resulted from the PCA data. (5 points)

            elbow1 + elbow2
            scatter1 + scatter2

    - After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
        - In this scenario, the impact of using fewer features to cluster the data is minimal. It does more clearly distinguish the different clusters on the plot, but overall the outcome is not much different. 
    - **Code for this part of the assignment was taken from activities performed in class.**





