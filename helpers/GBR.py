from sklearn.ensemble import GradientBoostingRegressor
import os

def train_GBR(training_df):
    """ Train a Gradient Boosting Regression model. """
    features = training_df[['score_bm25', 'score_lp', 'score_jm']].fillna(0)  # Handle missing scores
    labels = training_df['rel']

    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(features, labels)

    return gbr


def predict_with_gbr(testing_df, model):
    """ 
    - Predict with testing data. 
    - Write predicion score back to testing_df.
    """
    features = testing_df[['score_bm25', 'score_lp', 'score_jm']].fillna(0)
    testing_df['predicted_score'] = model.predict(features)
    return testing_df


def format_to_trec(testing_df, output_txt_path, top_k=1000):
    """ Format testing_df to trec format for evaluation. """
    formatted_data = []

    # sort by query_id and predicted_score, then assign rank
    for query_id, group in testing_df.groupby('query_id'):
        group = group.sort_values('predicted_score', ascending=False).reset_index()
        group['rank'] = range(1, len(group) + 1)

        # limit to top_k results
        group = group.head(top_k)

        for _, row in group.iterrows():
            formatted_data.append(f"{row['query_id']} Q0 {row['doc_id']} {row['rank']} {row['predicted_score']} GBR\n")

    # Write to output file
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, 'w') as f:
        f.writelines(formatted_data)

