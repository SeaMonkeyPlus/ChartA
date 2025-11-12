import os
from supabase import create_client, Client
from dotenv import load_dotenv


load_dotenv()  # ✅ .env 파일 자동 로드

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supa: Client = create_client(SUPABASE_URL, SUPABASE_KEY)