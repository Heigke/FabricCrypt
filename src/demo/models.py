"""Pydantic request/response models for the demo server."""
from pydantic import BaseModel
from typing import List, Optional


class ChallengeRequest(BaseModel):
    nonce_hex: Optional[str] = None  # if None, server picks a fresh one


class SignResponse(BaseModel):
    nonce_hex: str
    sig: List[float]   # 64 dims (32 phys + 32 nonce_emb)
    host: str
    elapsed_ms: float


class VerifyRequest(BaseModel):
    nonce_hex: str
    sig: List[float]
    expected_host: Optional[str] = None  # for diagnostic only


class VerifyResponse(BaseModel):
    accept: bool
    plan_score: float
    plan_thresh: float
    classifier_p0: Optional[float] = None
    elapsed_ms: float
