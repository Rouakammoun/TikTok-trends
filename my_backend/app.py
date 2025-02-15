from flask import Flask, jsonify
from flask_cors import CORS
import asyncio
from TikTokApi import TikTokApi
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)  # Enable CORS immediately after app creation

load_dotenv()
ms_token = os.getenv("ms_token")

# Retrieve ms_token correctly
async def get_tiktok_trends():
    trends = []
    try:
        async with TikTokApi() as api:
            await api.create_sessions(headless=False, ms_tokens=[ms_token], num_sessions=1, sleep_after=3)

            # Scrape the top 10 trending videos
            async for video in api.trending.videos(count=10):
                data = video.as_dict
                print("Full data structure:", data)  # Add this line to inspect the structure

                if 'id' not in data or 'desc' not in data:
                    continue

                # Check if the video URL is available in the 'video' field or another field
                video_url = data.get('video', {}).get('urls', [None])[0]  # This might be where it's located
                if not video_url:
                    video_url = data.get('video', {}).get('downloadAddr', None)  # Check another potential location for the URL

                trend = {
                    "video_id": data['id'],
                    "caption": data['desc'],
                    "likes": data.get('stats', {}).get('diggCount', 0),
                    "comments": data.get('stats', {}).get('commentCount', 0),
                    "shares": data.get('stats', {}).get('shareCount', 0),
                    "plays": data.get('stats', {}).get('playCount', 0),
                    "music": data.get('music', {}),
                    "video_url": video_url,  # Store the URL here
                    "created_time": data.get('createTime', None)
                }
                trends.append(trend)
    except Exception as e:
        print(f"Error fetching TikTok trends: {e}")

    return trends


@app.route('/', methods=['GET'])
def home():
    return "Welcome to the TikTok Trends API!"

@app.route('/api/tiktok-trends', methods=['GET'])
def fetch_tiktok_trends():
    trends = asyncio.run(get_tiktok_trends())  # Run async function inside sync route
    if not trends:
        return jsonify({"message": "No trends found"}), 404
    return jsonify(trends)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
