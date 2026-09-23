"""Bot-account check, stage 2 (local): query Lichess POST /api/users for every
username in groups.json and count games with a BOT-titled player per group."""
import json, sys, time, urllib.request
G = json.load(open(sys.argv[1]))
names = sorted({n for g in G["groups"].values() for x in g["games"] for n in (x["White"], x["Black"]) if n})
UA = "cs-thesis-bot-check/1.0 (undergraduate thesis research; bulk BOT-title check, one-off)"
users = {}
for i in range(0, len(names), 300):
    chunk = names[i:i + 300]
    req = urllib.request.Request("https://lichess.org/api/users", data=",".join(chunk).encode(),
                                 headers={"User-Agent": UA, "Content-Type": "text/plain", "Accept": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                for u in json.load(r):
                    users[u["id"]] = u
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(65); continue
            raise
    time.sleep(2)
out = {"n_names": len(names), "n_returned": len(users), "groups": {}}
for gname, g in G["groups"].items():
    games = g["games"]; bot_games = 0; bot_accounts = set(); accounts = set(); missing = set()
    for x in games:
        hit = False
        for n in (x["White"], x["Black"]):
            accounts.add(n.lower()); u = users.get(n.lower())
            if u is None: missing.add(n.lower()); continue
            if u.get("title") == "BOT": bot_accounts.add(n.lower()); hit = True
        bot_games += hit
    out["groups"][gname] = {"n_games": len(games), "games_with_bot": bot_games,
        "n_accounts": len(accounts), "bot_accounts": len(bot_accounts), "accounts_not_returned": len(missing),
        "err_min": min(x["err"] for x in games), "err_max": max(x["err"] for x in games)}
json.dump(out, open(sys.argv[2], "w"), indent=1); print(json.dumps(out, indent=1))
