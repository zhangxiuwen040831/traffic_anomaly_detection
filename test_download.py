import requests

urls = [
    "https://raw.githubusercontent.com/khhuang47/UNSW-NB15/master/UNSW_NB15_training-set.csv",
    "https://gitcode.net/mirrors/khhuang47/UNSW-NB15/raw/master/UNSW_NB15_training-set.csv",
    "https://hub.fastgit.org/khhuang47/UNSW-NB15/raw/master/UNSW_NB15_training-set.csv"
]

for url in urls:
    try:
        r = requests.get(url, timeout=10, stream=True)
        print(f"{url}: {r.status_code}")
    except Exception as e:
        print(f"{url}: {e}")