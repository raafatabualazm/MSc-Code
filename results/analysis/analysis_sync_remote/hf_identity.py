import json
import sys

from huggingface_hub import HfApi


token = sys.stdin.readline().strip()
if not token:
    raise SystemExit("missing token on stdin")

identity = HfApi(token=token).whoami()
print(
    json.dumps(
        {
            "name": identity.get("name"),
            "fullname": identity.get("fullname"),
            "type": identity.get("type"),
            "organizations": [
                {
                    "name": organization.get("name"),
                    "role": organization.get("roleInOrg"),
                }
                for organization in identity.get("orgs", [])
            ],
        },
        sort_keys=True,
    )
)
