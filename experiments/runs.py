"""List recent official runs (needs DRYFT_TOKEN). usage: python runs.py [n] ; python runs.py rerun SUBMISSION_ID KEY"""
import json, os, sys, urllib.request
BASE = "https://htn.dryft.ai/api/v1"
def call(path, data=None, key=None):
    h = {"Authorization": "Bearer " + os.environ["DRYFT_TOKEN"], "Content-Type": "application/json"}
    if key: h["Idempotency-Key"] = key
    req = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data is not None else None, headers=h)
    return json.load(urllib.request.urlopen(req, timeout=60))
if len(sys.argv) > 1 and sys.argv[1] == "rerun":
    print(call(f"/submissions/{sys.argv[2]}/runs", {"mode": "official"}, sys.argv[3])["run"]["id"]); sys.exit()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
for r in call(f"/runs?limit={n}")["items"]:
    res = call("/runs/" + r["id"])["run"].get("result") or {}
    sh = res.get("shapes") or []
    print(r["commitSha"][:7], r["id"][:8], r["state"], r.get("errorCode") or "", round(res.get("score") or 0, 1),
          [(s["id"][-1], s["caseStatus"][:4], round(s.get("tokensPerSecond") or 0)) for s in sh])
