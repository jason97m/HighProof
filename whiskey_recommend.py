import csv
import difflib
import streamlit as st
import pandas as pd

def load_csv(filename):
    with open(filename, newline='', encoding='utf-8') as file:
        reader = list(csv.reader(file))
        header = reader[0]
        data = reader[1:]
    return header, data

def find_best_match(user_input, whiskey_list):
    matches = difflib.get_close_matches(user_input, whiskey_list, n=1, cutoff=0.5)
    return matches[0] if matches else None

def main():
    st.title("Whiskey Recommender")

    filename = 'Meta-CriticWhiskeyDB.csv'  # Make sure this file is in your directory
    header, data = load_csv(filename)

    whiskey_names = [row[0] for row in data]

    user_input = st.text_input("What whiskey do you like?")

    if user_input:
        best_match = find_best_match(user_input, whiskey_names)

        if not best_match:
            st.warning("Sorry, no close match found.")
            return

        st.success(f"Best match found: {best_match}")

        matched_row = next(row for row in data if row[0] == best_match)

        rating = float(matched_row[1])
        type_whiskey = matched_row[5]
        flavor_profile = matched_row[7]
        region = matched_row[8]
        specific_type = matched_row[9]

        st.markdown(f"**Type:** {type_whiskey}")
        st.markdown(f"**Flavor Profile:** {flavor_profile}")
        st.markdown(f"**Region:** {region}")
        st.markdown(f"**Specific Type:** {specific_type}")

        # Filter and sort
        filtered = [
            row for row in data
            if row[5] == type_whiskey and row[7] == flavor_profile and
               row[8] == region and row[9] == specific_type
        ]

        filtered_sorted = sorted(filtered, key=lambda x: float(x[1]), reverse=True)

        st.subheader("Top 3 Recommended Whiskeys:")
        for row in filtered_sorted[:3]:
            st.write(f"{row[0]} (Rating: {row[1]})")

if __name__ == "__main__":
    main()
