import os
import joblib
import string
from nltk.corpus import stopwords
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words


SCOPES = ['https://www.googleapis.com/auth/gmail.modify']


def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)


def main():
    model = joblib.load('svc_spam.joblib')
    vectorizer = joblib.load('vectorizer.joblib')

    service = get_gmail_service()

    results = service.users().messages().list(userId='me', q='is:unread').execute()
    messages = results.get('messages', [])

    if not messages:
        print("nu sunt mesaje noi.")
        return


    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        email_body = msg_data.get('snippet', '')
        features = vectorizer.transform([email_body])
        prediction = model.predict(features)[0]
        status = "SPAM" if prediction == 1 else "HAM"
        print(f"Email: {email_body[:70]}...")
        print(f"rezultat: {status}")


if __name__ == '__main__':
    main()