"""ETL entry point: John Czajka. See etl_common.py for the shared data-fetching
methodology (MailChimp opens/axis + backend chat_votepayload/chat_userreply for
emoji clicks/conversations, matching eliworks-mailchimp-dashboard/app.py)."""

import os
import etl_common as common

CANDIDATE = "John Czajka"
SCHEMA    = "client_52_John_Czajka"
CLIENT_ID = 52
MC_KEY    = os.environ["CZAJKA_MC_KEY"]
MC_DC     = "us4"

if __name__ == "__main__":
    common.run(CANDIDATE, SCHEMA, MC_KEY, MC_DC, client_id=CLIENT_ID)
