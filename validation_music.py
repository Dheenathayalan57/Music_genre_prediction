from pydantic import BaseModel, Field

class MusicFeatures(BaseModel):
    Popularity:         int   = Field(..., ge=0, le=100)
    danceability:       float = Field(..., ge=0.0, le=1.0)
    energy:             float = Field(..., ge=0.0, le=1.0)
    key:                int   = Field(..., ge=0, le=11)
    loudness:           float = Field(..., ge=-60.0, le=0.0)
    mode:               int   = Field(..., ge=0, le=1)
    speechiness:        float = Field(..., ge=0.0, le=1.0)
    acousticness:       float = Field(..., ge=0.0, le=1.0)
    instrumentalness:   float = Field(..., ge=0.0, le=1.0)
    liveness:           float = Field(..., ge=0.0, le=1.0)
    valence:            float = Field(..., ge=0.0, le=1.0)
    tempo:              float = Field(..., ge=0.0)
    duration_in_min_ms: float = Field(..., gt=0, alias="duration_in min/ms")
    time_signature:     int   = Field(..., ge=1, le=7)

    class Config:
        populate_by_name = True  
