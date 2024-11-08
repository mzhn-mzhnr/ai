import requests

# URL для API
url = 'https://iam.api.cloud.yandex.net/iam/v1/tokens'

# Данные для запроса
data = {
    "yandexPassportOauthToken": ""
}

# Выполнение POST-запроса
response = requests.post(url, json=data)

# Печать результата
print(response.json())
