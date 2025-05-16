from google.oauth2 import service_account
import google.auth

credentials, project = google.auth.default()
print(f"✅ Projeto autenticado: {project}")
print(f"✅ Credenciais: {credentials}")
print(f"✅ Tipo de credencial: {type(credentials)}")