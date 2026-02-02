# Transformer-Based Sequential Movie Recommendation System

## Overview
An end-to-end sequential recommendation system employing a decoder-only transformer to predict the next item a user is likely to interact with. The model is trained on the MovieLens 1M dataset and evaluated on metrics such as Hit@K, NDCG@K, and MRR.

The system includes a FastAPI inference service that accepts a sequence of historical item interactions and returns top-K recommendations produced by the trained model.

## Model Architecture
The model is a decoder-only transformer trained for autoregressive next-item prediction. Given a padded sequence of item IDs, the model predicts the distribution over the next item at each timestep. During evaluation and inference, only the final non-padding timestep is used.

## Results
Note that during evaluation, only the final interaction of each user sequence is evaluated in order to minimize data leakage.

Final Test Results:
- Hit@50: 0.4072
- NDCG@50: 0.1568
- MRR: 0.0985

These results are consistent with an ID-only baseline. Performance is limited by the size of the dataset and lack of movie metadata.

# API / Inference
The trained model is served via a FastAPI inference service that accepts a sequence of item IDs and returns the top-K recommendations as determined by the trained model. 

### Example Inference Request
POST /recommendation
(Comments shown for clarity; not part of the actual JSON payload)
```json
{
  "interaction_history": [1, 2, 3, 4],  // historical item IDs
  "k": 5                               // number of recommendations to return
}
```

Example Response:
```json
{
  "recommendations": [5, 6, 7, 8, 9],
  "translated": [
    "Toy Story:Animation",
    "Jumanji:Adventure",
    " . . . omitted for brevity"
    ]      
}
```
## Future Improvements
The next steps would be to train the model on more data, and increase the expressivity of features by embedding movie metadata alongside the movie IDs. These improvements would likely lead to a significant increase in model performance.
