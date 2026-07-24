"""ETL entry point: Go America PAC (Brevo). New candidate as of 2026-07-22 — see
etl_common.py for the shared data-fetching methodology (Brevo opens/axis + backend
chat_votepayload/chat_userreply for emoji clicks/conversations). Schema/client_id
confirmed 2026-07-24 against eli_client and information_schema.schemata."""

import os
import etl_common as common

CANDIDATE        = "Go America PAC"
SCHEMA           = "client_122_Go_America_PAC"
CLIENT_ID        = 122
BREVO_KEY        = os.environ["BREVO_API_KEY"]
CAMPAIGN_PREFIX  = "eli-go_america_pac-"

if __name__ == "__main__":
    common.run_brevo(CANDIDATE, SCHEMA, BREVO_KEY, CAMPAIGN_PREFIX, client_id=CLIENT_ID)
