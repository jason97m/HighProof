from flask import Flask, request, render_template_string
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
    return df.iloc[similar_indices][['Whisky', 'Meta Critic']].values.tolist()

# HTML template
HTML_TEMPLATE = '''
<!doctype html>
<title>Whiskey Recommender</title>
<h2>Whiskey Recommendation System</h2>
<form action="/" method="post">
  Enter a whiskey name: <input type="text" name="whiskey_name">
  <input type="submit" value="Get Recommendations">
</form>

{% if recommendations %}
<h3>Recommendations for "{{ whiskey_name }}"</h3>
<ul>
  {% for whisky, score in recommendations %}
    <li><strong>{{ whisky }}</strong> (Meta Critic: {{ score }})</li>
  {% endfor %}
</ul>
{% elif whiskey_name %}
<p>No results found for "<strong>{{ whiskey_name }}</strong>"</p>
{% endif %}
'''

@app.route("/", methods=["GET", "POST"])
def home():
    whiskey_name = ""
    recommendations = []
    if request.method == "POST":
        whiskey_name = request.form["whiskey_name"]
        recommendations = recommend_whiskey(whiskey_name)

    return render_template_string(HTML_TEMPLATE, whiskey_name=whiskey_name, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
