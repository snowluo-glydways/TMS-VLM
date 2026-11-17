"""Log in to catalog.tbdrobotics.com (Cognito hosted UI + PKCE) and print tokens JSON.
Usage: CATALOG_USER=... CATALOG_PASS=... python catalog_login.py > token.json
"""
import base64, hashlib, json, os, re, secrets, sys, urllib.parse
import requests

BASE = "https://catalog.tbdrobotics.com"
s = requests.Session()
meta = s.post(BASE + "/api/auth/metadata", json="").json()
verifier = secrets.token_urlsafe(64)
challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
params = dict(response_type="code", client_id=meta["client_id"], scope="email openid profile",
              redirect_uri=meta["redirection_url"], code_challenge_method="S256", code_challenge=challenge)
login_url = f"https://{meta['environment_prefix']}-catalog-user-pool.auth.{meta['aws_region_id']}.amazoncognito.com/login?" + urllib.parse.urlencode(params)
r = s.get(login_url)
csrf = re.search(r'name="_csrf" value="([^"]+)"', r.text)
if not csrf:
    sys.exit("could not find csrf; page:\n" + r.text[:2000])
r = s.post(login_url, data={"_csrf": csrf.group(1), "username": os.environ["CATALOG_USER"],
                            "password": os.environ["CATALOG_PASS"], "signInSubmitButton": "Sign in"},
           allow_redirects=False)
loc = r.headers.get("Location", "")
code = urllib.parse.parse_qs(urllib.parse.urlparse(loc).query).get("code", [None])[0]
if not code:
    sys.exit(f"login failed: status={r.status_code} loc={loc}\n{r.text[:1500]}")
tok = s.post(BASE + "/api/auth/code", json={"code": code, "verifier": verifier}).json()
print(json.dumps(tok))
