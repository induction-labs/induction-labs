#!/usr/bin/env python3
"""
Video Duration Analyzer for GCS Action Capture Data

Counts video files for each user from the GCS structure:
gs://induction-labs/action_capture/<user>/<timestamp>/*.mp4
Each video is 30 seconds long.

Usage:
    python video_duration_analyzer.py
"""

from __future__ import annotations

from pathlib import Path

import gcsfs
import pandas as pd


class VideoDurationAnalyzer:
    VIDEO_DURATION_SECONDS = 30.0

    def __init__(
        self,
        bucket_path: str = "gs://induction-labs-data-ext-passive-mangodesk/action_capture/",
    ):
        self.bucket_path = bucket_path
        self.fs = gcsfs.GCSFileSystem()

    def list_users(self) -> list[str]:
        """List all users in the action capture directory"""
        try:
            users = []
            for path in self.fs.ls(self.bucket_path.replace("gs://", "")):
                user = Path(path).name
                if user and not user.startswith("."):
                    users.append(user)
            return sorted(users)
        except Exception as e:
            print(f"Error listing users: {e}")
            return []

    def count_user_videos(self, user: str) -> dict[str, str | int | float | list[dict]]:
        """Count MP4 files for a specific user"""
        user_path = f"{self.bucket_path}{user}/"
        video_count = 0
        sessions = []

        try:
            user_path_clean = user_path.replace("gs://", "")
            for timestamp_dir in self.fs.ls(user_path_clean):
                timestamp = Path(timestamp_dir).name
                timestamp_path_clean = timestamp_dir

                session_videos = 0
                for file_path in self.fs.ls(timestamp_path_clean):
                    if file_path.endswith(".mp4"):
                        session_videos += 1
                        video_count += 1

                if session_videos > 0:
                    sessions.append(
                        {
                            "timestamp": timestamp,
                            "video_count": session_videos,
                            "duration_seconds": session_videos
                            * self.VIDEO_DURATION_SECONDS,
                        }
                    )

        except Exception as e:
            print(f"Error counting videos for user {user}: {e}")

        total_duration = video_count * self.VIDEO_DURATION_SECONDS

        return {
            "user": user,
            "total_videos": video_count,
            "total_sessions": len(sessions),
            "total_duration_seconds": total_duration,
            "total_duration_minutes": total_duration / 60,
            "total_duration_hours": total_duration / 3600,
            "sessions": sessions,
        }

    def analyze_all_users(self) -> dict[str, dict]:
        """Count video files for all users"""
        users = self.list_users()

        if not users:
            print("No users found in the bucket")
            return {}

        print(f"Found {len(users)} users: {', '.join(users)}")

        results = {}
        for user in users:
            print(f"Counting videos for user: {user}")
            results[user] = self.count_user_videos(user)

        return results

    def generate_summary_report(self, results: dict[str, dict]) -> pd.DataFrame:
        """Generate a summary report as a DataFrame"""
        summary_data = []

        for user, data in results.items():
            summary_data.append(
                {
                    "User": user,
                    "Total Videos": data["total_videos"],
                    "Total Sessions": data["total_sessions"],
                    "Total Duration (seconds)": int(data["total_duration_seconds"]),
                    "Total Duration (minutes)": round(
                        data["total_duration_minutes"], 1
                    ),
                    "Total Duration (hours)": round(data["total_duration_hours"], 2),
                    "Avg Videos per Session": round(
                        data["total_videos"] / data["total_sessions"], 1
                    )
                    if data["total_sessions"] > 0
                    else 0,
                }
            )

        return pd.DataFrame(summary_data)

    def save_detailed_report(
        self, results: dict[str, dict], filename: str = "video_duration_analysis.csv"
    ):
        """Save detailed per-session analysis to CSV"""
        detailed_data = []

        for user, data in results.items():
            for session in data["sessions"]:
                detailed_data.append(
                    {
                        "User": user,
                        "Session Timestamp": session["timestamp"],
                        "Video Count": session["video_count"],
                        "Duration (seconds)": int(session["duration_seconds"]),
                        "Duration (minutes)": round(
                            session["duration_seconds"] / 60, 1
                        ),
                    }
                )

        df = pd.DataFrame(detailed_data)
        return df


def main():
    analyzer = VideoDurationAnalyzer()

    print("Starting video count analysis...")
    print("=" * 50)

    # Count videos for all users
    results = analyzer.analyze_all_users()

    if not results:
        print("No results to display")
        return

    # Generate and display summary
    print("\n" + "=" * 50)
    print("SUMMARY REPORT")
    print("=" * 50)

    summary_df = analyzer.generate_summary_report(results)
    print(summary_df.to_string(index=False))

    # Save detailed report
    print("\n" + "=" * 50)
    print("SAVING DETAILED REPORT")
    print("=" * 50)

    analyzer.save_detailed_report(results)

    # Overall statistics
    total_videos = summary_df["Total Videos"].sum()
    total_sessions = summary_df["Total Sessions"].sum()
    total_duration_hours = summary_df["Total Duration (hours)"].sum()

    print("\nOVERALL STATISTICS:")
    print(f"Total users: {len(results)}")
    print(f"Total sessions: {total_sessions}")
    print(f"Total videos: {total_videos}")
    print(f"Total duration: {total_duration_hours:.2f} hours")
    print(f"Average videos per user: {total_videos / len(results):.1f}")
    print(f"Average sessions per user: {total_sessions / len(results):.1f}")


if __name__ == "__main__":
    main()
