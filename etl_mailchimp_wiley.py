"""ETL entry point: James Wiley. See etl_common.py for the shared data-fetching
methodology (MailChimp opens/axis + backend chat_votepayload/chat_userreply for
emoji clicks/conversations, matching eliworks-mailchimp-dashboard/app.py)."""

import os
import etl_common as common

CANDIDATE = "James Wiley"
SCHEMA    = "client_253_James_Wiley"
CLIENT_ID = 253
MC_KEY    = os.environ["WILEY_MC_KEY"]
MC_DC     = "us16"

if __name__ == "__main__":
    common.run(CANDIDATE, SCHEMA, MC_KEY, MC_DC, client_id=CLIENT_ID)
