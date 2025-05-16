from google.oauth2 import service_account
from google.cloud import aiplatform

service_account_path = 'c:/Users/mlzengo/Documents/TJGO/SPRINT IV/br-tjgo-cld-02-09b1b22e65b3.json'

credentials = service_account.Credentials.from_service_account_file(
    service_account_path
)

aiplatform.init(
    project='cluster-residentes',
    location='us-central1',
    credentials=credentials
)

print("✅ Conexão estabelecida com o Google Cloud Platform (GCP)")
