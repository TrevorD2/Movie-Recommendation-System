import fastapi

from api.recommend_request import RecommendRequest
from api.recommender import Recommender

app = fastapi.FastAPI()
recommender = Recommender()

@app.post("/recommendation")
def get_recommendation(recommend_request: RecommendRequest):
    interaction_history = recommend_request.interaction_history
    k = recommend_request.k
    
    recommendations = recommender.recommend(interaction_history, k)

    return {"recommendations": recommendations}
