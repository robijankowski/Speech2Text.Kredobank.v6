import aiohttp
import asyncio


async def transcribe_audio_aiohttp(audio_content, filename, model: str = "whisper-1"):
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }
    data = aiohttp.FormData()
    data.add_field("file",
                   audio_content,
                   filename=filename,
                   content_type="audio/mpeg")
    data.add_field("model", model)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data) as response:
            if response.status == 200:
                return await response.json()  # Successful response
            else:
                return await response.text()  # Error handling

# Example usage
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    audio_file_path = "./Sources/2 380976776096 Кобець Т. 06.02.wav"
    with open(audio_file_path, "rb") as audio_file:
        audio_content = audio_file.read()
        filename = audio_file_path.split('/')[-1]
    result = asyncio.run(transcribe_audio_aiohttp(audio_content, filename))

    print(result)

