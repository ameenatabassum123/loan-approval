import pickle

# Load the model you saved
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# See what the model knows about its features
print("What the model is:", type(model))
if hasattr(model, 'feature_names_in_'):
    print("Features the model expects:", model.feature_names_in_)
else:
    print("Couldn’t find the feature list. It might be in a pipeline.")