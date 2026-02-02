# Transformer Based Sequential Movie Recommendation System

## Overview
An end-to-end sequential recommendation system employing a Transformer decoder to predict the next item a user is likely to interact with. The model is trained on the MovieLens 1M dataset and evaluated on metrics such as Hit@K, NDCG@K, and MRR.

Employs FastAPI for inference, accepting a sequence if item IDs and returning the top-K recommendations as determined by the model.

## Model Architecture
The model employs a transformer decoder module for autoregressive next-token prediction.

## Results
Note that during evaluation, only the final interaction of each user sequence is evaluated in order to minimize data leakage.

Final Test Results:
- Hit@50: 0.4072
- NDCG@50: 0.1568
- MRR: 0.0985

These results are sub-optimal, but expected due to the limited size of the dataset and relatively inexpressive features.

# API / Inference
Trained model is served via a FastAPI inference service that accepts a sequence of item IDs and returns the top-K recommendations as determined by the trained model. 

### Example Inference Request
POST /recommendation

```json
{
  "interaction_history": [1, 2, 3, 4],  // historical item IDs
  "k": 5                               // number of recommendations to return
}
```

Example Response:
```json
{
  "recommendations": [5, 6, 7, 8, 9],                               // recommendation item IDs
  "translated": ["Toy Story:Animation", "Jumanji:Adventure" . . . ] // Corresponding metadata for each movie (omitted some for brevity)        
}
```
## Future Improvements
The obvious next steps would be to train the model on more data, and increase the expressivity of features by embedding movie metadata alongside the movie ids. These improvements would likely lead to a significant increase in model performance.
