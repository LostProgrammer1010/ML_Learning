# Experrimenting with Different Models
- Test different models to see which one give you the best results
- scikit-learn ([Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html))
    - Tree Depth: Number of splits before coming to a predication
    - Many other decision tree option shown in documentation
- You have to find a sweet spot for the best number of leaf nodes to get the best results

# Overfitting
- Increasing the number of split would be 2^(#splits) meaning the data will get spread across more leafs as you add more splits
- Having a leaf contain a small group of data can lead to unaccurate predications with new data because there aren't many things to compare to
- Model become very good with the specific data given but poor with new data


# Underfitting
- Split only into a couple of leafs leaving a large group of data in each leaf
- This will lead to very inaccurate predication for new data and the trained data
- Does give the model enough time to find important distinctions or patterns in the data