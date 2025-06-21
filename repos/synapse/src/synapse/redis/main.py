#!/usr/bin/env python3
"""
Script to list all folders in a GCS bucket and create Redis entries for each.
Optimized for handling 100k+ folders efficiently.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import Enum
from functools import lru_cache

import pandas as pd
import redis
from pydantic import BaseModel, field_serializer, field_validator
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GCSYoutubeVideo(BaseModel):
    video_id: str

    @field_serializer("date_added")
    def serialize_created_at(self, value: datetime) -> str:
        return value.isoformat()

    @field_validator("date_added", mode="before")
    def validate_date_added(cls, value: str | datetime) -> datetime:
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        elif isinstance(value, datetime):
            return value
        else:
            raise ValueError("date_added must be a string or datetime object")

    date_added: datetime

    @field_validator(
        "time_started_processing", "time_finished_processing", mode="before"
    )
    def validate_time_fields(cls, value: str | datetime | None) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        elif isinstance(value, datetime):
            return value
        else:
            raise ValueError("time fields must be a string or datetime object")

    @field_serializer("time_started_processing", "time_finished_processing")
    def serialize_time_fields(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.isoformat()

    time_started_processing: datetime | None = None
    time_finished_processing: datetime | None = None


class GCSVideoQueue(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ERROR = "error"


VIDEO_RECORDINGS_PARQUET = (
    "gs://induction-labs/jonathan/video_recordings_downloaded_100k.parquet"
)


def put_video_data_into_redis(redis_client: redis.Redis):
    df = pd.read_parquet(VIDEO_RECORDINGS_PARQUET)
    BATCH_SIZE = 1000
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i : i + BATCH_SIZE]
        with redis_client.pipeline() as pipe:
            for _, row in batch.iterrows():
                video_id = row["video_id"]
                date_added = datetime.now(UTC)

                video_data = GCSYoutubeVideo(
                    video_id=video_id,
                    date_added=date_added,
                    time_started_processing=None,
                    time_finished_processing=None,
                )

                pipe.lpush(
                    GCSVideoQueue.NOT_STARTED,
                    video_data.model_dump_json(exclude_none=True, exclude_unset=True),
                )
            pipe.execute()


def get_video_next_unprocessed(
    redis_client: redis.Redis, queue: GCSVideoQueue = GCSVideoQueue.NOT_STARTED
) -> GCSYoutubeVideo | None:
    """
    Get the next unprocessed video from the specified Redis queue.
    """
    video_data_json = redis_client.rpop(queue)
    assert isinstance(video_data_json, str), (
        f"Expected video data to be a JSON string {video_data_json!r}"
    )
    if video_data_json:
        return GCSYoutubeVideo.model_validate_json(video_data_json)
    return None


def put_video_in_queue(
    redis_client: redis.Redis,
    video_data: GCSYoutubeVideo,
    queue: GCSVideoQueue,
):
    """
    Put a video data object into the specified Redis queue.
    """
    video_data_json = video_data.model_dump_json(exclude_none=True, exclude_unset=True)
    redis_client.lpush(queue.value, video_data_json)
    logger.info(f"Video {video_data.video_id} added to {queue.value} queue.")


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    redis_client = redis.Redis(
        host="jeffrey",
        port=6379,
        db=0,
        password=None,  # Set if your Redis requires authentication
        decode_responses=True,  # Decode responses to strings
    )
    if not redis_client.ping():
        raise ConnectionError("Could not connect to Redis server")
    logger.info("Connected to Redis server")
    return redis_client


if __name__ == "__main__":
    put_video_data_into_redis(get_redis_client())
