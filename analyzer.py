from PIL import Image
import pytesseract
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions, EntitiesOptions

# Set the path to tesseract.exe on your machine
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Step 1: OCR - extract text from image
image_path = "documents/s2.png"
extracted_text = pytesseract.image_to_string(Image.open(image_path))
print("\n📝 Extracted Text from Image:\n")
print(extracted_text)

# Step 2: IBM Watson NLU - analyze text
api_key = "U0eUK2PyFxFAMoeH7kouJsbaSJZ-D02wYa5jTSZ9tRGI"
service_url = "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/ef9739eb-ebab-4cef-a42c-b13561a34afc"

authenticator = IAMAuthenticator(api_key)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url(service_url)

# Analyze extracted text
response = nlu.analyze(
    text=extracted_text,
    features=Features(
        keywords=KeywordsOptions(limit=5),
        entities=EntitiesOptions(limit=5)
    )
).get_result()

print("\n🔍 Keywords and Entities:\n")
for keyword in response['keywords']:
    print(f"Keyword: {keyword['text']} (Relevance: {keyword['relevance']})")

for entity in response['entities']:
    print(f"Entity: {entity['text']} (Type: {entity['type']}, Relevance: {entity['relevance']})")
