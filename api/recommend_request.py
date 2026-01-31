import pydantic

class RecommendRequest(pydantic.BaseModel):
    interaction_history: list[int]
    k: int
