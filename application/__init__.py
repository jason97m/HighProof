from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- Load data from CSV ---
df = pd.read_csv("HIghProof/Meta-CriticWhiskeyDB.csv")

# Encode categorical variables
categorical_cols = ['Cost', 'Class', 'Super Cluster', 'Cluster', 'Country', 'Type']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Normalize features
features = ['Meta Critic', 'STDEV', '#', 'Cost', 'Class', 'Super Cluster', 'Cluster', 'Country', 'Type']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
similarity_matrix = cosine_similarity(X_scaled)

# Recommendation function
def recommend_whiskey(whisky_name, top_n=3):
    try:
        idx = df[df['Whisky'].str.contains(whisky_name, case=False, regex=False)].index[0]
    except IndexError:
        return []
    similarity_scores = similarity_matrix[idx]
    similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    results = df.iloc[similar_indices][['Whisky', 'Meta Critic']]
    return results.to_dict(orient='records')

# JSON API route
@app.route("/api", methods=["GET", "POST"])
def api_recommend():
    data = request.get_json()
    print("📩 Received JSON:", data)

    if not data or "whiskey_name" not in data:
        return jsonify({"error": "Missing 'whiskey_name' in request"}), 400

    whiskey_name = data["whiskey_name"]
    recommendations = recommend_whiskey(whiskey_name)

    if not recommendations:
        response = {
            "whiskey_name": whiskey_name,
            "recommendations": [],
            "message": "No matches found."
        }
        print("📤 Sending response JSON:", response)
        return jsonify(response)

    response = {
        "whiskey_name": whiskey_name,
        "recommendations": recommendations
    }
    print("📤 Sending response JSON:", response)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
