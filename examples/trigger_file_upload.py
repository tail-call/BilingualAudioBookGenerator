import dotenv

dotenv.load_dotenv(".env")

import bookaudio.app as app
from fastapi import UploadFile


async def upload_file():
    filename = "saint_exupery_antoine-the_little_prince.pdf"
    with open(filename, "rb") as f:
        uploadfile = UploadFile(f, filename=filename)
        await app.create_job(uploadfile, "English", "Russian")

await upload_file()
